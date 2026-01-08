import torch
import copy
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Callable

class ModelPartitioner(ABC):
    """Abstract interface for splitting model into aggregatable parts"""
    
    @abstractmethod
    def get_backbone_params(self, model: nn.Module) -> List[torch.Tensor]:
        """Returns parameters that use standard FedAvg aggregation"""
        pass
    
    @abstractmethod
    def get_adaptive_params(self, model: nn.Module) -> List[torch.Tensor]:
        """Returns parameters that use ALA-weighted aggregation"""
        pass
    
    @abstractmethod
    def freeze_backbone(self, model: nn.Module):
        """Freeze backbone during ALA adaptation"""
        pass
    
    @abstractmethod
    def unfreeze_all(self, model: nn.Module):
        """Unfreeze all parameters for training"""
        pass


class GINPartitioner(ModelPartitioner):
    # Partitioner for GIN models: backbone=convs+BN, head=lin1+BN1+lin2
    
    def get_backbone_params(self, model: nn.Module) -> List[torch.Tensor]:
        return list(model.convs.parameters()) + list(model.batch_norms.parameters())
    
    def get_adaptive_params(self, model: nn.Module) -> List[torch.Tensor]:
        return (
            list(model.lin1.parameters()) +
            list(model.batch_norm1.parameters()) +
            list(model.lin2.parameters())
        )
    
    def freeze_backbone(self, model: nn.Module):
        for p in model.convs.parameters():
            p.requires_grad = False
        for p in model.batch_norms.parameters():
            p.requires_grad = False
    
    def unfreeze_all(self, model: nn.Module):
        for p in model.parameters():
            p.requires_grad = True


class ALAWeightUpdater:
    # Handles adaptive weight computation and updates (Eq. 5 in paper)
    
    def __init__(self, eta: float = 1.0, normalize_grad: bool = True):
        self.eta = eta
        self.normalize_grad = normalize_grad
    
    def initialize_weights(self, param_shapes: List[torch.Size], device) -> List[torch.Tensor]:
        """Initialize ALA weights to ones"""
        return [torch.ones(shape, device=device) for shape in param_shapes]
    
    def update_weights(
        self,
        weights: List[torch.Tensor],
        params_temp: List[torch.Tensor],
        params_local: List[torch.Tensor],
        params_global: List[torch.Tensor]
    ) -> Dict[str, float]:
        """
        Update ALA weights using gradient: w <- clip(w - eta * grad_w)
        Returns telemetry statistics
        """
        grad_norm_sum = 0.0
        delta_norm_sum = 0.0
        n_params = 0
        
        with torch.no_grad():
            for w, pt, pl, pg in zip(weights, params_temp, params_local, params_global):
                if pt.grad is None:
                    continue
                
                # Telemetry
                grad_norm_sum += pt.grad.norm().item()
                delta_norm_sum += (pg - pl).norm().item()
                n_params += 1
                
                # Gradient w.r.t. weight: grad_w = grad_theta_t * (theta_g - theta_l)
                grad_w = pt.grad * (pg - pl)
                
                # Optional normalization for stability
                if self.normalize_grad:
                    grad_norm = grad_w.norm()
                    if grad_norm > 1e-6:
                        grad_w = grad_w / grad_norm
                
                # Update and clip
                w.sub_(self.eta * grad_w)
                w.clamp_(0, 1)
        
        return {
            "grad_norm_head": grad_norm_sum / max(1, n_params),
            "delta_norm": delta_norm_sum / max(1, n_params)
        }
    
    def apply_mixing(
        self,
        params_target: List[torch.Tensor],
        params_local: List[torch.Tensor],
        params_global: List[torch.Tensor],
        weights: List[torch.Tensor]
    ):
        """Apply Eq. 4: theta_t = theta_l + (theta_g - theta_l) * w"""
        with torch.no_grad():
            for pt, pl, pg, w in zip(params_target, params_local, params_global, weights):
                pt.copy_(pl + (pg - pl) * w)
    
    def weight_statistics(self, weights: List[torch.Tensor]) -> Dict[str, float]:
        """Compute summary statistics for weights"""
        if not weights:
            return {}
        
        with torch.no_grad():
            flat = torch.cat([w.flatten() for w in weights])
            return {
                "w_mean": flat.mean().item(),
                "w_min": flat.min().item(),
                "w_max": flat.max().item(),
                "w_frac_0": (flat <= 1e-6).float().mean().item(),
                "w_frac_1": (flat >= 0.999999).float().mean().item()
            }


class ALAAdapter:
    """Orchestrates the ALA adaptation process"""
    
    def __init__(
        self,
        partitioner: ModelPartitioner,
        weight_updater: ALAWeightUpdater,
        data_ratio: float = 0.8,
        max_init_epochs: int = 20,
        converge_eps: float = 1e-3,
        converge_patience: int = 2
    ):
        self.partitioner = partitioner
        self.weight_updater = weight_updater
        self.data_ratio = data_ratio
        self.max_init_epochs = max_init_epochs
        self.converge_eps = converge_eps
        self.converge_patience = converge_patience
        
        self.weights: Optional[List[torch.Tensor]] = None
    
    def adapt(
        self,
        local_model: nn.Module,
        global_model: nn.Module,
        data_loader,
        loss_fn: Callable,
        round_id: int,
        device
    ) -> Dict[str, float]:
        """
        Perform ALA adaptation and return adapted model + statistics
        
        Args:
            local_model: Client's local model from previous round
            global_model: New global model from server
            data_loader: DataLoader for ALA adaptation subset
            loss_fn: Loss function
            round_id: Current communication round (0-indexed)
            device: Torch device
        
        Returns:
            Dictionary of telemetry statistics
        """
        # Round 0: just use global model
        if round_id == 0:
            return {}
        
        # Get parameter groups
        params_local = self.partitioner.get_adaptive_params(local_model)
        params_global = self.partitioner.get_adaptive_params(global_model)
        
        # Create temporary model for gradient computation
        temp_model = copy.deepcopy(global_model).to(device)
        params_temp = self.partitioner.get_adaptive_params(temp_model)
        self.partitioner.freeze_backbone(temp_model)
        
        # Initialize weights if needed
        if self.weights is None:
            shapes = [p.shape for p in params_temp]
            self.weights = self.weight_updater.initialize_weights(shapes, device)
        
        # Adaptation: round 1 = converge, round 2+ = one epoch
        if round_id == 1:
            stats = self._adapt_until_convergence(
                temp_model, params_temp, params_local, params_global,
                data_loader, loss_fn, device
            )
        else:
            stats = self._adapt_one_epoch(
                temp_model, params_temp, params_local, params_global,
                data_loader, loss_fn, device
            )
        
        # Apply final mixing to create adapted model
        self.weight_updater.apply_mixing(params_temp, params_local, params_global, self.weights)
        
        return stats
    
    def _adapt_one_epoch(
        self, temp_model, params_temp, params_local, params_global,
        data_loader, loss_fn, device
    ) -> Dict[str, float]:

        temp_model.eval()
        stats = {"ala_batches": 0, "ala_loss": 0.0}
        
        for batch in data_loader:
            batch = batch.to(device)
            
            # Apply current weights
            self.weight_updater.apply_mixing(params_temp, params_local, params_global, self.weights)
            
            # Forward + backward
            temp_model.zero_grad()
            _, logits = temp_model(batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            
            # Update weights
            update_stats = self.weight_updater.update_weights(
                self.weights, params_temp, params_local, params_global
            )
            
            stats["ala_loss"] += loss.item()
            stats["ala_batches"] += 1
            stats.update(update_stats)
        
        # Average and add weight statistics
        stats["ala_loss"] /= max(1, stats["ala_batches"])
        stats.update(self.weight_updater.weight_statistics(self.weights))
        
        return stats
    
    def _adapt_until_convergence(
        self, temp_model, params_temp, params_local, params_global,
        data_loader, loss_fn, device
    ) -> Dict[str, float]:
        # Run ALA until weights converge (for initial round)
        stable_count = 0
        
        for epoch in range(self.max_init_epochs):
            # Save previous weights
            prev_weights = [w.clone() for w in self.weights]
            
            # One epoch update
            stats = self._adapt_one_epoch(
                temp_model, params_temp, params_local, params_global,
                data_loader, loss_fn, device
            )
            
            # Check convergence
            with torch.no_grad():
                max_delta = max((w - wp).abs().max().item() 
                               for w, wp in zip(self.weights, prev_weights))
            
            if max_delta < self.converge_eps:
                stable_count += 1
                if stable_count >= self.converge_patience:
                    stats["converged_at_epoch"] = epoch + 1
                    break
            else:
                stable_count = 0
        
        return stats