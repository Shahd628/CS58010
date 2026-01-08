import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class CollaborationGraph:
    """
    Manages collaboration graph construction based on ALA weight deltas.
    Each client has a graph connecting it to all other clients.
    """

    def __init__(self, num_clients: int, num_head_layers: int):
        self.num_clients = num_clients
        self.num_head_layers = num_head_layers
        
        # Store previous alpha weights for delta computation
        # Shape: [num_clients, num_head_layers, param_dim]
        self.prev_alphas: Optional[Dict[int, List[torch.Tensor]]] = None
        
        # Collaboration graphs: G_i^p for client i, layer p
        # Shape: [num_clients, num_head_layers, num_clients] (adjacency matrices)
        self.adjacency_matrices = None

    def update_graphs( self, current_alphas: Dict[int, List[torch.Tensor]] ) -> torch.Tensor:
        """
        Compute collaboration graphs based on cosine similarity of alpha deltas.
        
        Args:
            current_alphas: Dict[client_id -> List[alpha_weights per layer]]
        
        Returns:
            adjacency_matrices: [num_clients, num_head_layers, num_clients]
        """
        if self.prev_alphas is None:
            # First round: no deltas, use uniform weights or identity
            self.prev_alphas = current_alphas
            self.adjacency_matrices = self._initialize_uniform_graphs()
            return self.adjacency_matrices
        
        # Compute deltas: Δα_i^p = α_i^p(t) - α_i^p(t-1)
        deltas = {}
        for client_id, alphas in current_alphas.items():
            if client_id in self.prev_alphas:
                deltas[client_id] = [
                    (curr - prev).flatten()
                    for curr, prev in zip(alphas, self.prev_alphas[client_id])
                ]
        
        # Build adjacency matrices via cosine similarity
        self.adjacency_matrices = self._compute_similarity_graphs(deltas)
        
        # Update previous alphas
        self.prev_alphas = {k: [a.clone() for a in v] for k, v in current_alphas.items()}
        
        return self.adjacency_matrices
    
    def _initialize_uniform_graphs(self) -> torch.Tensor:
        """Initialize with uniform weights (all clients equally connected)"""
        graphs = torch.ones(self.num_clients, self.num_head_layers, self.num_clients)
        # Zero out self-loops
        for i in range(self.num_clients):
            graphs[i, :, i] = 0
        # Normalize
        graphs = graphs / (self.num_clients - 1)
        return graphs
    
    def _compute_similarity_graphs( self, deltas: Dict[int, List[torch.Tensor]] ) -> torch.Tensor:
        """
        Compute cosine similarity between all client pairs for each layer.
        
        Returns:
            graphs: [num_clients, num_head_layers, num_clients]
        """
        client_ids = sorted(deltas.keys())
        num_layers = len(next(iter(deltas.values())))
        
        graphs = torch.zeros(self.num_clients, num_layers, self.num_clients)
        
        for layer_idx in range(num_layers):
            # Collect all deltas for this layer: [num_clients, delta_dim]
            layer_deltas = torch.stack([
                deltas[cid][layer_idx] for cid in client_ids
            ])  # [num_clients, delta_dim]
            
            # Compute pairwise cosine similarity
            # Normalize vectors
            layer_deltas_norm = F.normalize(layer_deltas, p=2, dim=1)
            similarity_matrix = torch.mm(layer_deltas_norm, layer_deltas_norm.t())
            # [num_clients, num_clients]
            
            # Apply ReLU to keep only positive similarities
            similarity_matrix = F.relu(similarity_matrix)
            
            # Zero out diagonal (no self-loops)
            similarity_matrix.fill_diagonal_(0)
            
            # Normalize rows (each client's edge weights sum to 1)
            row_sums = similarity_matrix.sum(dim=1, keepdim=True)
            row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
            similarity_matrix = similarity_matrix / row_sums
            
            # Store for all clients
            for i, cid in enumerate(client_ids):
                graphs[cid, layer_idx, :] = similarity_matrix[i, :]
        
        return graphs


class GraphHypernetwork(nn.Module):
    """
    Graph Hypernetwork that processes collaboration graphs to produce
    client-specific embeddings for personalized aggregation.
    """
    
    def __init__( self, num_clients: int, node_feature_dim: int = 64, hidden_dim: int = 128, num_gnn_layers: int = 2, aggregation: str = 'mean' ):
        super().__init__()
        self.num_clients = num_clients
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        
        # Learnable node embeddings for each client
        self.node_embeddings = nn.Embedding(num_clients, node_feature_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(
                node_feature_dim if i == 0 else hidden_dim,
                hidden_dim,
                aggregation=aggregation
            )
            for i in range(num_gnn_layers)
        ])
        
        # Output dimension = hidden_dim (embedding for decoder)
        self.output_dim = hidden_dim
    
    def forward( self, adjacency_matrix: torch.Tensor, client_id: int ) -> torch.Tensor:
        """
        Process collaboration graph for a specific client.
        
        Args:
            adjacency_matrix: [num_clients, num_clients] edge weights
            client_id: Target client ID
        
        Returns:
            embedding: [hidden_dim] representation for this client
        """
        # Get node features for all clients
        node_ids = torch.arange(self.num_clients, device=adjacency_matrix.device)
        node_features = self.node_embeddings(node_ids)  # [num_clients, node_feature_dim]
        
        # Apply GNN layers
        h = node_features
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, adjacency_matrix)  # [num_clients, hidden_dim]
        
        # Extract embedding for target client
        client_embedding = h[client_id]  # [hidden_dim]
        
        return client_embedding


class GNNLayer(nn.Module):
    """Single GNN layer with message passing and aggregation"""
    
    def __init__(self, in_dim: int, out_dim: int, aggregation: str = 'mean'):
        super().__init__()
        self.aggregation = aggregation
        
        # Message transformation
        self.message_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        # Update transformation
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU()
        )
    
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, in_dim]
            adjacency: [num_nodes, num_nodes] normalized edge weights
        
        Returns:
            updated_features: [num_nodes, out_dim]
        """
        # Compute messages
        messages = self.message_mlp(node_features)  # [num_nodes, out_dim]
        
        # Aggregate messages from neighbors (weighted by adjacency)
        aggregated = torch.mm(adjacency, messages)  # [num_nodes, out_dim]
        
        # Update node features
        combined = torch.cat([node_features, aggregated], dim=1)  # [num_nodes, in_dim + out_dim]
        updated = self.update_mlp(combined)  # [num_nodes, out_dim]
        
        return updated


class AlphaDecoder(nn.Module):
    """
    Decodes GHN embeddings into personalized alpha coefficients.
    One decoder per head layer parameter group.
    """
    
    def __init__(self, embedding_dim: int, param_shape: torch.Size, hidden_dim: int = 64):
        super().__init__()
        # assert len(param_shape) == 1, "Each decoder should handle one parameter shape" # Having a list of tensors caused error so now we just pass one tesnor
        self.param_shape = param_shape
        # self.num_head_params = len(param_shapes)
        
        # Separate decoder for each head parameter
        self.decoders = \
            nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.param_shape.numel()),
                nn.Sigmoid()  # α ∈ [0, 1]
            )
            # for shape in param_shapes # param_shapes no longer a list
    
    def forward(self, embedding: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            embedding: [embedding_dim] from GHN
        
        Returns:
            alphas: List of [param_shape] alpha coefficients
        """
        # alphas = []
        # for decoder,shape in zip(self.decoders, self.param_shapes):
        alpha_flat = self.decoders(embedding)  # type: ignore # equivalent to [numel]
        alpha_shaped = alpha_flat.view(self.param_shape) # reshaping
        # alphas.append(alpha_shaped)
        
        return alpha_shaped


class CollaborativeHeadAggregator:
    """
    Aggregates head parameters from top-k similar clients based on
    collaboration graph edge weights.
    """
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
    
    def aggregate_heads(self, client_id: int, all_client_heads: Dict[int, List[torch.Tensor]], adjacency_matrix: torch.Tensor, param_idx: int) -> torch.Tensor:
        """
        Aggregate a single head parameter from top-k similar clients.
        
        Args:
            client_id: Target client
            all_client_heads: Dict[client_id -> List[head params per layer]]
            adjacency_matrix: [num_clients] edge weights from client_id to others
            param_idx: Which paramtere to aggregate
        
        Returns:
            aggregated_params: single torch.Tensor aggregated for this parameters
        """
        # Get top-k neighbors by edge weight
        edge_weights = adjacency_matrix.clone()
        edge_weights[client_id] = -1  # Exclude self
        
        top_k = min(self.top_k, (edge_weights > 0).sum().item())
        if top_k == 0:
            # No valid neighbors, return zeros
            target_param = all_client_heads[client_id][param_idx]
            return torch.zeros_like(target_param)
            # return [torch.zeros_like(p) for p in all_client_heads[client_id]]
        
        top_k_indices = torch.topk(edge_weights, top_k).indices
        top_k_weights = edge_weights[top_k_indices]
        
        # Normalize weights
        top_k_weights = top_k_weights / top_k_weights.sum()
        
        # Weighted aggregation
        aggregated = None
        for idx, weight in zip(top_k_indices, top_k_weights):
            idx = int(idx.item())
            if idx not in all_client_heads or param_idx >= len(all_client_heads[idx]):
                continue
            
            param = all_client_heads[idx][param_idx]
            aggregated = weight * param.clone() if aggregated is None else aggregated + weight * param

            # if aggregated is None:
            #     aggregated = weight * param.clone()
            # else:
            #     # for i, p in enumerate(client_params):
            #     aggregated += weight * param
        if aggregated is None:
            target_param = all_client_heads[client_id][param_idx]
            return torch.zeros_like(target_param)
        
        return aggregated #if aggregated is not None else [torch.zeros_like(p) for p in all_client_heads[client_id]]
