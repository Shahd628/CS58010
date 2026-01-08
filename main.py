import random
import numpy as np
import torch
import io
import os
import csv
import time
import sys
# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import openfgl.config as config


from openfgl.flcore.trainer import FGLTrainer

args = config.args

# args.root = "D:/CySec/CS58010/project/OpenFGL/data" #"/home/ceren/Desktop/scalable/OpenFGL-main/data"
args.root = "./data"
args.scenario = "graph_fl"
args.task = "graph_cls"

args.fl_algorithm = "fedalaghn"
args.model = ["gin"]
CSV_FILE = args.fl_algorithm+"_seed_results.csv"
write_header = not os.path.exists(CSV_FILE)

# BELOW 3 ADDED [NEW]
args.lambda_graph = 0.6
args.ala_temperature = 0.09
args.ala_warmup_rounds = 0

args.dataset = ["MUTAG"] # choose multiple datasets for data heterogeneity setting.
args.simulation_mode = "graph_fl_label_skew"
args.skew_alpha = 1
args.num_clients = 10
args.lr = 0.001
args.num_epochs = 1
args.num_rounds = 100
args.batch_size = 128
args.weight_decay = 5e-4
args.dropout = 0.5
args.optim = "adam"
args.seed = 42
args.use_wandb = False
args.wandb_project = "pFGL"
args.wandb_entity = "scalable-group2"
args.wandb_name="you_think_you_funny"
args.dirichlet_alpha=0.5


args.metrics = ["accuracy"]

# Patch torch.load for PyTorch 2.6+ issue
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    # Ensure backward compatibility with older model loading code
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load # Override torch.load with patched version

run_name="shahd_"+ args.fl_algorithm + "_" + args.dataset[0] + "_seed"+ str(args.seed)
args.wandb_name=run_name

with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    if write_header:
        writer.writerow([
            "dataset",
            "seed",
            "best_round",
            "best_val_accuracy",
            "best_test_accuracy"
        ])

trainer = FGLTrainer(args)

# trainer.train()
start=time.time()
# Start federated training, evaluate the global model and save results
# Capture printed output from evaluation
# old_stdout = sys.stdout
# sys.stdout = mystdout = io.StringIO()
trainer.train()
#trainer.evaluate()  # This prints metrics to console
# sys.stdout = old_stdout
# end=time.time()

# fname="results_"+args.dataset[0]+'_'+args.fl_algorithm+".txt" # "results_" + str(args.seed) + '_' + ds + '_' + args.fl_algorithm + ".txt"
# with open(fname, "w", encoding="utf-8") as f:
#     f.write(mystdout.getvalue())

print(f"Evaluation results saved to {fname}")
print("Time taken was", end-start)

result = trainer.evaluation_result

with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        args.dataset[0],
        args.seed,
        result["best_round"],
        result["best_val_accuracy"],
        result["best_test_accuracy"],
    ])
print(f"\nResults saved to {CSV_FILE}")

print(
    f"[Seed {args.seed}] "
    f"best_round={result['best_round']} | "
    f"best_val_acc={result['best_val_accuracy']:.4f} | "
    f"best_test_acc={result['best_test_accuracy']:.4f}"
)