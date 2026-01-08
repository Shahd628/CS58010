from openfgl.flcore.base import BaseServer
import copy
import torch

class FedALAreServer(BaseServer):
    """
    FedALA server - identical to FedAvg server.
    All ALA-specific logic is on the client side.
    """
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super().__init__(args, global_data, data_dir, message_pool, device)
    
    def execute(self):
        """Standard FedAvg aggregation"""
        with torch.no_grad():
            sampled = self.message_pool["sampled_clients"]
            num_tot = sum(self.message_pool[f"client_{cid}"]["num_samples"] 
                         for cid in sampled)
            
            # Weighted average aggregation
            agg_state = {k: torch.zeros_like(v) if v.is_floating_point() else v.clone()
                        for k, v in self.task.model.state_dict().items()}
            
            for cid in sampled:
                weight = self.message_pool[f"client_{cid}"]["num_samples"] / num_tot
                client_state = self.message_pool[f"client_{cid}"]["state"]
                
                for k in agg_state:
                    if agg_state[k].is_floating_point():
                        agg_state[k].add_(client_state[k].to(agg_state[k].device), 
                                         alpha=float(weight))
            
            self.task.model.load_state_dict(agg_state, strict=True)
    
    def send_message(self):
        """Broadcast global model to clients"""
        self.message_pool["server"] = {
            "state": copy.deepcopy(self.task.model.state_dict())
        }