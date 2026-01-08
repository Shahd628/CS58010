import copy
import torch
from openfgl.flcore.base import BaseServer
from .components import CollaborationGraph, GraphHypernetwork, AlphaDecoder, CollaborativeHeadAggregator
from openfgl.flcore.fedalare.components import GINPartitioner
from typing import List, Dict, Optional, Callable

class FedALAGHNServer(BaseServer):
    """
    FedALAGHN Server: Manages collaboration graphs, graph hypernetwork inference,
    and collaborative head aggregation.
    """
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super().__init__(args, global_data, data_dir, message_pool, device)
        
        # GHN hyperparameters
        self.ghn_node_dim = getattr(args, 'ghn_node_dim', 64)
        self.ghn_hidden_dim = getattr(args, 'ghn_hidden_dim', 128)
        self.ghn_num_layers = getattr(args, 'ghn_num_layers', 2)
        self.top_k_neighbors = getattr(args, 'ghn_top_k', 5)
        
        # Initialize components (lazy initialization after first round)
        self.collab_graph = None
        self.graph_hypernetwork = None
        self.alpha_decoders = None  # One per head layer
        self.head_aggregator = None
        
        self.num_head_params = None  # Determined from first client message

        # GHN results stored
        self.ghn_alphas = {}
        self.collaborative_heads = {}
    
    def execute(self):
        """
        Server execution:
        1. Standard FedAvg aggregation of backbone
        2. Update collaboration graphs based on alpha deltas
        3. Run GHN to predict personalized alphas
        4. Aggregate heads from top-k similar clients
        5. Send results to clients
        """
        sampled = self.message_pool["sampled_clients"]
        
        # Step 1: Standard FedAvg aggregation
        self._fedavg_aggregation(sampled)
        
        # Step 2-4: GHN-based personalization
        round_id = self.message_pool.get("round", 0)
        if round_id > 0:  # Skip round 0 (no previous alphas)
            self._ghn_personalization(sampled)
    
    def _fedavg_aggregation(self, sampled_clients):
        """Standard weighted averaging of all parameters"""
        with torch.no_grad():
            num_tot = sum(self.message_pool[f"client_{cid}"]["num_samples"] 
                         for cid in sampled_clients)
            
            agg_state = {k: torch.zeros_like(v) if v.is_floating_point() else v.clone()
                        for k, v in self.task.model.state_dict().items()}
            
            for cid in sampled_clients:
                weight = self.message_pool[f"client_{cid}"]["num_samples"] / num_tot
                client_state = self.message_pool[f"client_{cid}"]["state"]
                
                for k in agg_state:
                    if agg_state[k].is_floating_point():
                        agg_state[k].add_(client_state[k].to(agg_state[k].device), 
                                         alpha=float(weight))
            
            self.task.model.load_state_dict(agg_state, strict=True)
    
    def _ghn_personalization(self, sampled_clients):
        """
        GHN-based personalization:
        1. Collect alpha weights from clients
        2. Update collaboration graphs
        3. Run GHN to predict personalized alphas
        4. Aggregate heads from similar clients
        """
        # Collect alphas from all clients
        current_alphas = {}
        all_client_heads = {}
        
        for cid in sampled_clients:
            client_msg = self.message_pool[f"client_{cid}"]
            if client_msg.get("ala_alphas") is not None:
                current_alphas[cid] = client_msg["ala_alphas"]
            
            # Extract head parameters
            client_state = client_msg["state"]
            all_client_heads[cid] = self._extract_head_params(client_state)
        
        if not current_alphas:
            return  # No alphas available yet
        
        # Initialize components if first time
        if self.collab_graph is None:
            self._initialize_ghn_components(current_alphas)
        
        # Update collaboration graphs
        adjacency_matrices = self.collab_graph.update_graphs(current_alphas)
        # [num_clients, num_head_layers, num_clients]
        
        # Run GHN and predict alphas for each client
        ghn_alphas = {}
        collaborative_heads = {}
        
        for cid in sampled_clients:
            client_alphas = []
            client_collab_heads = []
            
            for param_idx in range(self.num_head_params):
                # Get collaboration graph for this client and layer
                adj_matrix = adjacency_matrices[cid, min(param_idx, adjacency_matrices.shape[1]-1), :]  # [num_clients]
                adj_matrix_2d = adj_matrix.unsqueeze(0).repeat(self.collab_graph.num_clients, 1)  # Broadcast to [num_clients, num_clients] for GNN

                embedding = self.graph_hypernetwork(adj_matrix_2d, cid)  # [hidden_dim]
                
                # Decode to alpha
                alpha = self.alpha_decoders[param_idx](embedding)  # [param_shape]
                # alpha=alpha_list[0] ################################## what about 1,2,...? -> we now have an assert in AlphaDecoder for this
                client_alphas.append(alpha)
                
                # Aggregate heads from top-k neighbors ####TUNABLE
                collab_param = self.head_aggregator.aggregate_heads(
                    client_id=cid,
                    all_client_heads=all_client_heads,
                    adjacency_matrix=adj_matrix,
                    param_idx=param_idx
                )
                client_collab_heads.append(collab_param) # .append adds one tensor while .extend appends all tensors so from [1,2],[3,4]->[1,2,[3,4]] vs [1,2,3,4]
            
            ghn_alphas[cid] = client_alphas
            collaborative_heads[cid] = client_collab_heads
        
        # self.message_pool["server"]["ghn_alphas"] = ghn_alphas
        # self.message_pool["server"]["collaborative_heads"] = collaborative_heads
        self.ghn_alphas = ghn_alphas
        self.collaborative_heads = collaborative_heads
    
    def _initialize_ghn_components(self, current_alphas):
        """Initialize GHN components based on first alpha batch"""
        num_clients = len(current_alphas)
        self.num_head_params = len(next(iter(current_alphas.values())))
        
        # Get parameter shapes from first client's alphas
        first_client_alphas = next(iter(current_alphas.values()))
        param_shapes = [alpha.shape for alpha in first_client_alphas]

        self.collab_graph = CollaborationGraph(
            num_clients=num_clients,
            num_head_layers=self.num_head_params
        )
        
        self.graph_hypernetwork = GraphHypernetwork(
            num_clients=num_clients,
            node_feature_dim=self.ghn_node_dim,
            hidden_dim=self.ghn_hidden_dim,
            num_gnn_layers=self.ghn_num_layers
        ).to(self.device)
        
        self.alpha_decoders = torch.nn.ModuleList([
            AlphaDecoder(
                embedding_dim=self.ghn_hidden_dim,
                param_shape=param_shapes[layer_idx], # passing shapes
                hidden_dim=64
            ).to(self.device)
            for layer_idx in range(self.num_head_params)
        ])
        
        self.head_aggregator = CollaborativeHeadAggregator(top_k=self.top_k_neighbors)
    
    def _extract_head_params(self, state_dict) -> List[torch.Tensor]:
        # Assuming GIN architecture: lin1, batch_norm1, lin2 # Needs extensibility
        # head_keys = [k for k in state_dict.keys() 
        #             if k.startswith('lin1.') or k.startswith('batch_norm1.') or k.startswith('lin2.')]
        # return [state_dict[k] for k in sorted(head_keys)]
        dummy_model = copy.deepcopy(self.task.model)
        dummy_model.load_state_dict(state_dict, strict=True)
        return GINPartitioner().get_adaptive_params(dummy_model)
    
    def send_message(self):
        self.message_pool["server"] = {
            "state": copy.deepcopy(self.task.model.state_dict()),
            "ghn_alphas": self.ghn_alphas,  # Now reading from instance variable
            "collaborative_heads": self.collaborative_heads  # Now reading from instance variable
        }