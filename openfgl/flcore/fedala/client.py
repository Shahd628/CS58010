import numpy as np
import torch
from openfgl.flcore.base import BaseClient
import random
from torch_geometric.loader import DataLoader
import torch.nn as nn
import copy
import torch.nn.functional as F

class FedALAClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the FedAvgClient.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
        """
        super(FedALAClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.start_phase =  True
        self.ala_weights = None  # Adaptive weights for top layers (head)
        self.ala_reset_each_round = False # do not initialize weights to ones at each round
        self.ala_data_ratio = 0.8 # ratio of local training graphs used for ALA
        self.ala_eta = 1.0 # learning rate for ALA weight updates
        self.max_init_epochs = 20 # max epochs for initial ALA weight convergence
        self.converge_eps = 1e-3 # if max |Δw| < eps, then consider converged
        self.converge_patience = 2 # eps sağlanmasını ardışık kaç epoch isteyelim
        self.device = device
        self._ala_epoch_stats = None  # telemetry: stats from last ALA epoch
        self.last_ala_stats = None  # telemetry: stats to send to server
        
    def execute(self):
        """
        Executes the local training process. This method first gets the global model from the message pool,
        then performs local initialization including the ALA adaptation, and finally trains the model
        on the client's local data.
        """
        global_state = self.message_pool["server"]["state"]
        self.local_initialization(global_state)
        self.task.train()


    def send_message(self):
        """
        Sends a message to the server containing the model parameters after training
        and the number of samples in the client's dataset.
        """
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "state": copy.deepcopy(self.task.model.state_dict()),
                "ala": copy.deepcopy(self.last_ala_stats)
        }

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _head_params(model):
        """
        This is a helper function to get the top layer (head) parameters of the GIN model.
        For other GNN models, this function should be modified accordingly."""
        # Head-only (GIN): lin1 + batch_norm1 (affine params) + lin2
        return (
            list(model.lin1.parameters()) +
            list(model.batch_norm1.parameters()) +
            list(model.lin2.parameters())
        )
            
    def _w_stats(self):
        """Compute summary statistics for current ALA weights."""
        if self.ala_weights is None or len(self.ala_weights) == 0:
            return {}
        with torch.no_grad():
            flat = torch.cat([w.detach().flatten() for w in self.ala_weights])
            eps = 1e-6
            return {
                "w_mean": float(flat.mean().item()),
                "w_min": float(flat.min().item()),
                "w_max": float(flat.max().item()),
                "w_frac_0": float((flat <= eps).float().mean().item()),
                "w_frac_1": float((flat >= 1.0 - eps).float().mean().item()),
            }

    def _build_ala_loader(self):
        """
        Builds a DataLoader for the ALA adaptation process by sampling a subset of the local training graphs.
        The size of the subset is determined by the ala_data_ratio attribute."""
        # Graph-FL: ALA datası = train graph'ların s% subset'i
        train_idx = self.task.train_mask.nonzero(as_tuple=False).view(-1)
        train_idx = train_idx.detach().cpu().tolist()

        if len(train_idx) == 0:
            return None

        m = max(1, int(self.ala_data_ratio * len(train_idx)))
        sampled_idx = random.sample(train_idx, m) if m < len(train_idx) else train_idx

        # self.task.data graph list/dataset; BaseTask'te zaten device'a taşınmış olur
        sampled_graphs = [self.task.data[i] for i in sampled_idx]

        # ALA için küçük batch_size daha stabil olabilir; burada args.batch_size kullanıyorum
        return DataLoader(sampled_graphs, batch_size=self.args.batch_size, shuffle=False)

    def _apply_head_mix_(self, params_t, params_l, params_g):
        """
        Applies the adaptive layer aggregation (ALA) mixing to the top layer parameters.
        """
        
        """
        Eq(4) head kısmı: theta_t = theta_l + (theta_g - theta_l) * w
        w: self.ala_weights
        """
        # overwrite the parameters
        with torch.no_grad():
            for pt, pl, pg, w in zip(params_t, params_l, params_g, self.ala_weights):
                pt.data.copy_(pl.data + (pg.data - pl.data) * w)

    def _ala_one_epoch_update(self, temp_model, params_t, params_l, params_g, ala_loader):
        """
        Paper Eq(5) benzeri: w <- clip(w - eta * (grad_theta_t * (theta_g - theta_l)))
        Graph-FL: loss = CE(logits, batch.y)

        NOTE: Model update logic is unchanged; we only collect telemetry for debugging/W&B.
        """
        temp_model.eval()
        total_loss = 0.0
        n_batches = 0

        loss_start = None
        loss_end = None
        grad_norm_acc = 0.0
        delta_norm_acc = 0.0
        n_terms = 0

        for batch in ala_loader:
            batch = batch.to(self.device)

            # 1) temp head'i w ile yeniden kur
            self._apply_head_mix_(params_t, params_l, params_g)

            # 2) forward + loss
            temp_model.zero_grad(set_to_none=True)
            _, logits = temp_model(batch)
            loss = self.task.default_loss_fn(logits, batch.y)

            # 3) backward => grad(theta_t)
            loss.backward()

            if loss_start is None:
                loss_start = float(loss.item())
            loss_end = float(loss.item())

            # 4) w update + clip (same as before) + telemetry norms
            with torch.no_grad():
                for pt, pl, pg, w in zip(params_t, params_l, params_g, self.ala_weights):
                    if pt.grad is None:
                        continue
                    grad_norm_acc += float(pt.grad.detach().norm().item())
                    delta_norm_acc += float((pg.data - pl.data).detach().norm().item())
                    n_terms += 1

                    grad_w = pt.grad * (pg.data - pl.data)
                    grad_norm = grad_w.norm()
                    if grad_norm > 1e-6:  # sıfıra bölme kontrolü
                        grad_w = grad_w / grad_norm  # normalize
                    w.data.sub_(self.ala_eta * grad_w)
                    w.data.clamp_(0, 1)

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        stats = self._w_stats()
        stats: dict[str, float] = self._w_stats()
        stats.update({
            "ala_loss": float(avg_loss),
            "ala_loss_start": float(loss_start) if loss_start is not None else float("nan"),
            "ala_loss_end": float(loss_end) if loss_end is not None else float("nan"),
            "grad_norm_head": float(grad_norm_acc / max(1, n_terms)),
            "delta_norm": float(delta_norm_acc / max(1, n_terms)),
            "ala_batches": float(n_batches),
        })

        self._ala_epoch_stats = stats

        return avg_loss
    def local_initialization(self, global_state_dict):

        # first, load global model weights to the local model(s)
        # store global model copy
        # store local model copy
        # sample a random subset of training nodes for ALA
        # create a subgraph Data object for these nodes
        # get global and local model parameters
        # if first round, skip adaptation (at the first round, global and local models are identical)
        # if not first round: start ALA
        # create a temporary model copy # Aradığımız şey w’yi öğrenmek, local parametreleri güncellemek değil. bu yüzden temp model gerekli.
        # get the top layer parameters from global, local, and temp models
        # freeze lower layers of temp model
        # initialize adaptive weights with ones
        # forward pass on temp model with subgraph data
        # compute loss
        # backpropagate to get gradients (loss.backward())
        # update adaptive weights with Eq(5) --> weight = weight - eta * (param_t.grad * (param_g - param_l))
        # reconstruct the top layers of the temp model with --> param_l + (param_g - param_l) * weight
        # check convergence criteria: if converged, break the loop
        # at the end, get the top layers of the temp_model and write it to self.task.model
        round_id = int(self.message_pool.get("round", 0))
        local_model = copy.deepcopy(self.task.model).to(self.device) # get the local model which is from the last round
        
        global_model = copy.deepcopy(self.task.model).to(self.device)
        global_model.load_state_dict(global_state_dict, strict=True)

       # Round 0 (t=1): ALA skip, sadece global ile başla
        if round_id == 0:
            self.last_ala_stats = None
            self._ala_epoch_stats = None
            self.task.model.load_state_dict(global_state_dict, strict=True)
            return

        # Prepare top layers for adaptation
        temp_model = copy.deepcopy(global_model).to(self.device) # Temporary model for gradient updates
        
        params_l = self._head_params(local_model) # local head
        params_g = self._head_params(global_model) # global head
        params_t = self._head_params(temp_model) # temp head (to be updated)
        
        # freeze the backbone
        for p in temp_model.convs.parameters():
            p.requires_grad = False
        for p in temp_model.batch_norms.parameters():
            p.requires_grad = False

        # make sure that temp head is being tracked
        for p in params_t:
            p.requires_grad = True

            
        
        # 6) ALA data loader (s% local train graphs)
        ala_loader = self._build_ala_loader()


        # if ala_loader is None:
        #     # train graph yoksa global ile başla
        #     self.task.model.load_state_dict(global_state_dict, strict=True)
        #     return

        if self.ala_reset_each_round or (self.ala_weights is None):
            self.ala_weights = [torch.ones_like(p.data, device=self.device) for p in params_t]


        # 8) t=2 (round_id==1) initial stage: converge until
        #    t>2 (round_id>=2): one epoch update
        if round_id == 1:
            stable_cnt = 0
            for _ in range(self.max_init_epochs):
                w_prev = [w.detach().clone() for w in self.ala_weights]
                _ = self._ala_one_epoch_update(temp_model, params_t, params_l, params_g, ala_loader)

                # converge check: max |Δw|
                with torch.no_grad():
                    max_delta = 0.0
                    for w, wp in zip(self.ala_weights, w_prev):
                        max_delta = max(max_delta, float((w - wp).abs().max().item()))

                if max_delta < self.converge_eps:
                    stable_cnt += 1
                    if stable_cnt >= self.converge_patience:
                        break
                else:
                    stable_cnt = 0
        else:
            # t>2: one epoch
            _ = self._ala_one_epoch_update(temp_model, params_t, params_l, params_g, ala_loader)

        # telemetry: save last ALA stats (from the last ALA epoch)
        self.last_ala_stats = copy.deepcopy(self._ala_epoch_stats)

        # 9) Eq(4): hat{Theta}_i^t oluştur ve self.task.model'e yaz
        #    lower layers global, head = local + (global-local)*w
        self.task.model.load_state_dict(global_state_dict, strict=True)
        # self.task.model.load_state_dict(temp_model.state_dict(), strict=True)
        
        # self.task.model head paramlarını w ile karıştır
        model_params_t = self._head_params(self.task.model)
        self._apply_head_mix_(model_params_t, params_l, params_g)
        for p in self.task.model.convs.parameters():
            p.requires_grad = True
        for p in self.task.model.batch_norms.parameters():
            p.requires_grad = True