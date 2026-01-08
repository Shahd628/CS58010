import torch
import copy
import torch
from openfgl.flcore.fedalare.client import FedALAreClient
from openfgl.flcore.fedalare.components import ALAAdapter
from typing import List, Dict, Optional, Callable

class FedALAGHNClient(FedALAreClient):
    """
    FedALAGHN Client: Inherits FedALA's adaptive aggregation and adds
    graph hypernetwork-based collaborative head aggregation.
    """
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super().__init__(args, client_id, data, data_dir, message_pool, device)
        
        # [NEW] GHN-specific: store alpha history for server
        self.current_ala_alphas = None
    
    def execute(self):
        """
        Execute FedALAGHN:
        1. Receive GHN-predicted alphas from server (if available)
        2. Receive collaborative head aggregation from server
        3. Perform ALA adaptation with GHN-enhanced alphas
        4. Standard training
        """
        round_id = self.message_pool.get("round", 0) # HA!
        global_state = self.message_pool["server"]["state"] #O(1) I think
        

        global_model = copy.deepcopy(self.task.model)
        global_model.load_state_dict(global_state, strict=True)
        
        # Get GHN-predicted alphas if available
        ghn_alphas = self.message_pool["server"].get("ghn_alphas", {}).get(self.client_id, None)
        
        # Get collaborative head aggregation if available
        collaborative_heads = self.message_pool["server"].get("collaborative_heads", {}).get(self.client_id, None)
        
        if round_id > 0 and self.local_model is not None: # would be nice to have some warmup rounds. GHN is expensive so we may not want to apply it initially
            # Perform ALA adaptation
            ala_loader = self._build_ala_loader()
            self.ala_stats = self.adapter.adapt(
                local_model=self.local_model,
                global_model=global_model,
                data_loader=ala_loader,
                loss_fn=self.task.default_loss_fn,
                round_id=round_id,
                device=self.device
            )
            
            # Apply GHN-predicted alphas if available
            if ghn_alphas is not None:
                self._apply_ghn_alphas(global_model, ghn_alphas)
            
            # Integrate collaborative heads if available
            if collaborative_heads is not None:
                self._integrate_collaborative_heads(global_model, collaborative_heads, ghn_alphas)
            
            self.task.model.load_state_dict(global_model.state_dict())
        else:
            # Round 0: just use global model
            self.task.model.load_state_dict(global_state, strict=True)
            self.ala_stats = {}
        
        # Store current ALA alphas for server's graph construction
        self.current_ala_alphas = copy.deepcopy(self.adapter.weights) if self.adapter.weights else None
        
        # Standard training
        self.task.train()
        
        # Save local model for next round
        self.local_model = copy.deepcopy(self.task.model)
    
    def _apply_ghn_alphas(self, model, ghn_alphas: List[torch.Tensor]):
        """
        Override ALA's local alphas with GHN-predicted personalized alphas.
        """
        # if self.adapter.weights is not None and len(ghn_alphas) == len(self.adapter.weights):
        #     for i, alpha in enumerate(ghn_alphas):
        #         self.adapter.weights[i] = alpha.to(self.device).clone()
        if self.adapter.weights is None or len(ghn_alphas) != len(self.adapter.weights):
            return
    
        for i, alpha in enumerate(ghn_alphas): # ghn_alphas is a list so we shape it into a tensor

            if isinstance(alpha, (list, tuple)):
                if len(alpha) == 0:
                    continue
                alpha = alpha[0] 

            target_shape = self.adapter.weights[i].shape
            
            # Convert to tensor if needed
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)
            else:
                alpha = alpha.to(self.device)
            
            # Reshape/broadcast to match target
            if alpha.shape != target_shape:
                if alpha.numel() == 1:  # Scalar - broadcast
                    alpha = alpha.expand(target_shape).clone()
                else:  # Try to reshape
                    try:
                        alpha = alpha.view(target_shape)
                    except RuntimeError:
                        print(f"Warning: Cannot reshape alpha from {alpha.shape} to {target_shape}")
                        continue
            
            self.adapter.weights[i] = alpha.clone()
    
    def _integrate_collaborative_heads(self, model, collaborative_heads: List[torch.Tensor], ghn_alphas: Optional[List[torch.Tensor]]):
        """
        Integrate collaboratively aggregated head parameters.
        Uses GHN alphas to blend local, global, and collaborative heads.
        
        Formula: θ_final = θ_local + α * (θ_collaborative - θ_local)
        """
        head_params = self.adapter.partitioner.get_adaptive_params(model)
        
        # DEBUG
        print(f"\n=== Integration Debug ===")
        print(f"head_params: {len(head_params)} tensors")
        for i, p in enumerate(head_params):
            print(f"  [{i}] shape: {p.shape}")
        
        print(f"collaborative_heads: {len(collaborative_heads)} tensors")
        for i, p in enumerate(collaborative_heads):
            print(f"  [{i}] shape: {p.shape}")
        
        if ghn_alphas:
            print(f"ghn_alphas: {len(ghn_alphas)} tensors")
            for i, a in enumerate(ghn_alphas):
                print(f"  [{i}] shape: {a.shape}")
        print("=" * 30)

        if ghn_alphas is None:
            # Fallback: use uniform blending
            alpha = 0.5
            with torch.no_grad():
                for local_p, collab_p in zip(head_params, collaborative_heads):
                    local_p.data.mul_(1 - alpha).add_(collab_p.to(local_p.device), alpha=alpha)
        else:
            # Use GHN-predicted alphas for blending
            with torch.no_grad():
                for local_p, collab_p, alpha in zip(head_params, collaborative_heads, ghn_alphas):
                    # alpha_val = alpha.item() if alpha.dim() == 0 else alpha.mean().item()
                    alpha=alpha.to(local_p.device)
                    collab_p=collab_p.to(local_p.device)
                    # element-wise blending: theta = theta_local * (1 - α) + theta_collab * α ALT+233 for theta symbol: Θ
                    local_p.data.mul_(1 - alpha).add_(collab_p*alpha)
    
    def send_message(self):
        """Send model state, ALA stats, and current alphas to server"""
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "state": copy.deepcopy(self.task.model.state_dict()),
            "ala_stats": copy.deepcopy(self.ala_stats),
            "ala_alphas": copy.deepcopy(self.current_ala_alphas) if self.current_ala_alphas else None
        }
