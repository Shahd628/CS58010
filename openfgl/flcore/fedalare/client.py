import copy
import random
import torch
from torch_geometric.loader import DataLoader
from openfgl.flcore.base import BaseClient
from .components import ALAAdapter, GINPartitioner, ALAWeightUpdater


class FedALAreClient(BaseClient):
    """
    refactored FedALA client with more abstraction for extensibility (should now run with subgraph-fl tasks as well as other GNN models than gin, once relvant classes/functions get implemented in components).
    """
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super().__init__(args, client_id, data, data_dir, message_pool, device)
        
        # Initialize components
        self.partitioner = GINPartitioner()  # Can be swapped for other architectures, but make it an args in config
        self.weight_updater = ALAWeightUpdater(
            eta=getattr(args, 'ala_eta', 1.0),
            normalize_grad=True
        )
        
        self.adapter = ALAAdapter(
            partitioner=self.partitioner,
            weight_updater=self.weight_updater,
            data_ratio=getattr(args, 'ala_data_ratio', 0.8),
            max_init_epochs=getattr(args, 'ala_max_init_epochs', 20),
            converge_eps=getattr(args, 'ala_converge_eps', 1e-3),
            converge_patience=getattr(args, 'ala_converge_patience', 2)
        )
        
        self.local_model = None  # Store previous round's model
        self.ala_stats = None
    
    def execute(self):
        """Execute FedALA: adapt global model, then train"""
        global_state = self.message_pool["server"]["state"]
        round_id = self.message_pool.get("round", 0)
        
        # Load global model
        global_model = copy.deepcopy(self.task.model)
        global_model.load_state_dict(global_state, strict=True)
        
        # Perform ALA adaptation
        if round_id > 0 and self.local_model is not None:
            ala_loader = self._build_ala_loader()
            self.ala_stats = self.adapter.adapt(
                local_model=self.local_model,
                global_model=global_model,
                data_loader=ala_loader,
                loss_fn=self.task.default_loss_fn,
                round_id=round_id,
                device=self.device
            )
            # Copy adapted parameters back to task model
            self.task.model.load_state_dict(global_model.state_dict())
        else:
            # Round 0: just use global model
            self.task.model.load_state_dict(global_state, strict=True)
            self.ala_stats = {}
        
        # Standard local training
        self.task.train()
        
        # Save local model for next round
        self.local_model = copy.deepcopy(self.task.model)
    
    def _build_ala_loader(self):
        """Build DataLoader for ALA adaptation (sampled subset)"""
        train_idx = self.task.train_mask.nonzero(as_tuple=False).view(-1).tolist()
        if not train_idx:
            return None
        
        m = max(1, int(self.adapter.data_ratio * len(train_idx)))
        sampled_idx = random.sample(train_idx, m) if m < len(train_idx) else train_idx
        sampled_graphs = [self.task.data[i] for i in sampled_idx]
        
        return DataLoader(
            sampled_graphs,
            batch_size=self.args.batch_size,
            shuffle=False
        )
    
    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "state": copy.deepcopy(self.task.model.state_dict()),
            "ala_stats": copy.deepcopy(self.ala_stats)
        }
