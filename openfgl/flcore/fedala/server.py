import copy
import torch
from openfgl.flcore.base import BaseServer
from collections import OrderedDict
import torch.nn.functional as F

def _is_bn_buffer(k: str) -> bool:
    return ("running_mean" in k) or ("running_var" in k) or ("num_batches_tracked" in k)

def _is_backbone(k: str) -> bool:
    return k.startswith("convs.") or k.startswith("batch_norms.")

def _is_neck_head(k: str) -> bool:
    return k.startswith("lin1.") or k.startswith("batch_norm1.") or k.startswith("lin2.")

class FedALAServer(BaseServer):
    """
    FedALAServer implements the server-side logic for the Federated Adaptive Local Aggregation (FedALA) algorithm,
    as introduced in the paper "FedALA: Adaptive Local Aggregation for Personalized Federated Learning"
    by Zhang et al. (2022). This class is responsible for adaptively aggregating local model updates from clients
    and sending the updated global model to the server in the federated learning process.

    Attributes:
        None (inherits attributes from BaseServer)
    """
    
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the FedALAServer.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): Global dataset accessible by the server.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between server and clients.
            device (torch.device): Device to run the computations on.
            lambda_graph (float): Weight controlling the contribution of model parameter similarity versus graph-embedding similarity when computing client aggregation weights (Tunable hyperparameter)
            ala_temperature (float): Softmax temperature controlling how sharply client similarity scores influence aggregation weights  (Tunable hyperparameter)
            ala_warmup_rounds (int): Number of initial training rounds during which graph-structure similarity is incorporated into aggregation weight computation  (Tunable hyperparameter)
        """
        super(FedALAServer, self).__init__(args, global_data, data_dir, message_pool, device)

        # ============================
        # [NEW] Hyperparameters
        # ============================
        # self.lambda_graph = args.lambda_graph
        self.ala_temperature = args.ala_temperature
        self.ala_warmup_rounds = args.ala_warmup_rounds
    
    def execute(self):
        """
        Executes the server-side operations. This method aggregates model updates from the 
        clients by computing a weighted average of the model parameters, based on the number 
        of samples each client used for training.
        """
        with torch.no_grad(): # Disable gradient tracking since server-side aggregation does not require backprop
            sampled = self.message_pool["sampled_clients"] # Retrieve the list of client IDs sampled in the current federated round
            num_tot = sum(self.message_pool[f"client_{cid}"]["num_samples"] for cid in sampled)

            global_state = self.task.model.state_dict() # Get the current global model parameters

            # ============================
            # [CHANGED] Use neck/head parameters for distance computation
            # Since local training is only 1 epoch, backbone changes are minimal.
            # Head parameters are more meaningful because ALA adapts them to local data.
            # ============================
            # dist_keys = []
            # for k, v in global_state.items():
            #     if _is_neck_head(k) and torch.is_floating_point(v):
            #         if _is_bn_buffer(k):
            #             continue
            #         dist_keys.append(k)

            # if len(dist_keys) == 0:
            #     raise RuntimeError("dist_keys is empty. Check neck/head key filters and model state_dict keys.")
            # print(f"[Distance Keys] Using neck/head parameters: {dist_keys}")

            # eps = 1e-8

            # # ======================================
            # # [NEW] Collect graph embeddings
            # # ======================================
            # graph_embs = [
            #     self.message_pool[f"client_{cid}"]["graph_emb"].to(self.device) # Retrieve and move each client's graph embedding to the server device
            #     for cid in sampled
            # ]
            # # ======================================
            # # [NEW] Compute the global graph prototype by averaging client graph embeddings
            # # ======================================
            # h_global = torch.stack(graph_embs).mean(dim=0)
            # h_global_norm = h_global.norm() + eps

            # Get the current training round index
            round_id = self.message_pool.get("round", 0)

            # lambda_t = min( ##### NEW 2
            #     self.lambda_graph,
            #     round_id / max(1, self.ala_warmup_rounds) * self.lambda_graph
            # )

            # # ======================================
            # # [NEW] FIRST PASS: compute param distances
            # # ======================================
            # param_dists = [] # List to store parameter distances for each client
            # graph_dists = [] # List to store graph embedding distances (used in early rounds only)

            # for cid in sampled:
            #     client_state = self.message_pool[f"client_{cid}"]["state"]

            #     # Layer-wise normalized distance, then average
            #     per_key = []
            #     for k in dist_keys:
            #         diff = client_state[k].to(self.device) - global_state[k]
            #         global_norm = global_state[k].norm() + eps
            #         relative_diff = (diff.norm() / global_norm).pow(2)
            #         per_key.append(relative_diff)
            #     param_dist = torch.sqrt(torch.stack(per_key).mean())
            #     param_dists.append(param_dist)
                    
            #     if round_id < self.ala_warmup_rounds:
            #         client_emb = self.message_pool[f"client_{cid}"]["graph_emb"].to(self.device)
            #         graph_dist = (client_emb - h_global).norm() / (h_global_norm + eps)
            #         graph_dists.append(graph_dist)

            # # ======================================
            # # [NEW] Normalize param distances ACROSS clients (Z-score)
            # # ======================================
            # param_dists = torch.stack(param_dists) # Convert parameter distance list to tensor
            # print(f"[Round {round_id}] Raw param_dists (neck/head): {param_dists}")

            # # param_dists = param_dists / (param_dists.mean() + 1e-8) # Normalize distances to stabilize scale across clients
            # # Z-score normalization - preserves distribution, better separation
            # p_std = param_dists.std()
            # if p_std > eps:
            #     param_dists = (param_dists - param_dists.mean()) / p_std
            # else:
            #     # All distances are the same
            #     param_dists = torch.zeros_like(param_dists)


            # if round_id < self.ala_warmup_rounds: # Normalize graph distances only when they are used
            #     graph_dists = torch.stack(graph_dists) # Convert graph distance list to tensor
            #     # graph_dists = graph_dists / (graph_dists.mean() + 1e-8) # Normalize graph distances to comparable scale
            #     # Z-score normalization for graph distances
            #     g_std = graph_dists.std()
            #     if g_std > eps:
            #         graph_dists = (graph_dists - graph_dists.mean()) / g_std
            #     else:
            #         graph_dists = torch.zeros_like(graph_dists)

            # # ======================================
            # # [NEW] Build logits
            # # ======================================
            # logits = [] # List to store similarity-based aggregation logits
            # for i, cid in enumerate(sampled): # Iterate through clients with index
            #     if round_id < self.ala_warmup_rounds: # Use hybrid (parameter + graph) distance in early rounds
            #         total_dist = (
            #             lambda_t * param_dists[i] # Weight parameter distance
            #             + (1 - lambda_t) * graph_dists[i] # Weight graph structure distance
            #         )
            #     else:
            #         total_dist = param_dists[i] # After warm-up, rely only on parameter similarity

            #     logit = -total_dist # Convert distance to similarity score (closer → higher weight)
            #     logits.append(logit) # Store logit for this client
 
            #     print( # Debug output to monitor distances and logits
            #         f"[Round {round_id}] Client {cid} | "
            #         f"param_dist(norm)={param_dists[i].item():.6f}, "
            #         f"logit={logit.item():.6f}"
            #     )

            # # ======================================
            # # [NEW] Softmax aggregation weights
            # # ======================================
            # logits = torch.stack(logits) # Stack logits into a tensor
            # print(f"[Round {round_id}] Logits: min={logits.min().item():.4f}, "
            #       f"max={logits.max().item():.4f}, std={logits.std().item():.4f}")

            # alphas = torch.softmax(logits / self.ala_temperature, dim=0) # Convert logits into normalized aggregation weights using temperature scaling
            # print(f"[Round {round_id}] alpha std = {alphas.std().item():.6f}") # Print standard deviation of weights to assess weighting diversity

            # # ======================================
            # # [NEW] Similarity-weighted aggregation
            # # ======================================
            backbone_keys = [k for k, v in global_state.items()
                             if _is_backbone(k) and torch.is_floating_point(v)]
            neck_head_keys = [k for k, v in global_state.items()
                              if _is_neck_head(k) and torch.is_floating_point(v)]
            agg_state = OrderedDict() # Initialize ordered dictionary for aggregated model parameters


            for k, v in global_state.items(): # Iterate over global model parameters
                agg_state[k] = torch.zeros_like(v) if torch.is_floating_point(v) else v.clone() # Initialize accumulators for floating-point parameters

            # Backbone: similarity-weighted aggregation (alpha)
            # for k in agg_state:
            #     agg_state[k].zero_()

            # for alpha, cid in zip(alphas, sampled): # Aggregate each client's contribution weighted by alpha
            #     client_state = self.message_pool[f"client_{cid}"]["state"] # Retrieve client model parameters
            #     a=float(alpha)
            #     for k in backbone_keys:
            #         agg_state[k].add_(client_state[k].to(self.device), alpha=a)   # Weighted sum of client parameters

            # Neck+Head: FedAvg (sample-weighted aggregation)
            # ns = torch.tensor(
            #     [float(self.message_pool[f"client_{cid}"]["num_samples"]) for cid in sampled],
            #     device=self.device,
            #     dtype=torch.float32
            # )
            # ws = ns / (ns.sum() + 1e-8)

            # for w, cid in zip(ws, sampled):
            #     client_state = self.message_pool[f"client_{cid}"]["state"]
            #     for k in agg_state:
            #         if torch.is_floating_point(agg_state[k]):
            #             agg_state[k].add_(client_state[k].to(self.device), alpha=float(w))
            # # for k in neck_head_keys:
            # #     agg_state[k].zero_()

            # for w, cid in zip(ws, sampled):
            #     client_state = self.message_pool[f"client_{cid}"]["state"]
            #     ww = float(w)
            #     for k in neck_head_keys:
            #         agg_state[k].add_(client_state[k].to(self.device), alpha=ww)
            # 2) aggregate only floating tensors
            for cid in sampled:
                weight = self.message_pool[f"client_{cid}"]["num_samples"] / num_tot
                client_state = self.message_pool[f"client_{cid}"]["state"]

                for k in agg_state.keys():
                    if torch.is_floating_point(agg_state[k]):
                        # ensure same device
                        ck = client_state[k].to(agg_state[k].device)
                        agg_state[k].add_(ck, alpha=float(weight))
                    else:
                        # keep as-is (or copy from first client if you prefer)
                        pass

            self.task.model.load_state_dict(agg_state, strict=True) # Update the global model with the similarity-weighted aggregated parameters


    def send_message(self):
        """
        Sends a message to the clients containing the updated global model parameters after 
        aggregation.
        """
        self.message_pool["server"] = {
            "state": copy.deepcopy(self.task.model.state_dict())
        }