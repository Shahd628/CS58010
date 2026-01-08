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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

SEED_GRID = [42, 21, 7]   # ← 3 runs

args = config.args

# args.root = "D:/CySec/CS58010/project/OpenFGL/data" #"/home/ceren/Desktop/scalable/OpenFGL-main/data"
args.root = "./data"
args.scenario = "graph_fl"
args.task = "graph_cls"


args.fl_algorithm = "fedala"
args.model = ["gin"]
CSV_FILE = args.fl_algorithm+"_seed_results.csv"

# Fedala/Fedalaghn-Specific
args.lambda_graph = 0.6
args.ala_temperature = 0.09
args.ala_warmup_rounds = 10

# args.dataset = ["IMDB-MULTI"] # choose multiple datasets for data heterogeneity setting.
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

args.use_wandb = True
args.wandb_project = "pFGL"
args.wandb_entity = "scalable-group2"
args.wandb_name="shahd_fedala_BZR_seed42"


args.metrics = ["accuracy"]

# Patch torch.load for PyTorch 2.6+ issue
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    # Ensure backward compatibility with older model loading code
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load # Override torch.load with patched version

write_header = not os.path.exists(CSV_FILE)
CSV_FILE = args.fl_algorithm+"_seed_results.csv"
datasets=["ENZYMES", "DD", "PROTEINS", "IMDB-BINARY", "IMDB-MULTI"]
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
    for ds in datasets:
        print(f"\n===== Running dataset {ds} =====")
        args.dataset = [ds]
        if args.dataset[0]=='Cox2':
            args.dirichlet_alpha=5
        else:
            args.dirichlet_alpha=0.5
        start_seed_grid_time=time.time()
        for seed in SEED_GRID:
            print(f"\n===== Running seed {seed} =====")

            set_seed(seed)
            args.seed = seed

            run_name="shahd_"+ args.fl_algorithm + "_" + ds + "_seed"+ str(args.seed)
            args.wandb_name=run_name

            trainer = FGLTrainer(args)

            start_time=time.time()
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()
            trainer.train()
            sys.stdout = old_stdout
            end_time=time.time()

            fname="results_" + str(args.seed) + '_' + ds + '_' + args.fl_algorithm + ".txt"
            with open(fname, "w", encoding="utf-8") as f:
                f.write(mystdout.getvalue())

            print(f"Evaluation results saved to {fname}")
            print(f"Time taken to run trainer was {end_time-start_time}")

            result = trainer.evaluation_result

            writer.writerow([
                args.dataset[0],
                seed,
                result["best_round"],
                result["best_val_accuracy"],
                result["best_test_accuracy"],
            ])

            print(
                f"[Seed {seed}] "
                f"best_round={result['best_round']} | "
                f"best_val_acc={result['best_val_accuracy']:.4f} | "
                f"best_test_acc={result['best_test_accuracy']:.4f}"
            )

        print(f"\nResults saved to {CSV_FILE}")
        end_seed_grid_time=time.time()
        print(f"Time for seed grid to run was {end_seed_grid_time-start_seed_grid_time}")

# trainer = FGLTrainer(args)

# trainer.train()

# Start federated training, evaluate the global model and save results
# Capture printed output from evaluation
# old_stdout = sys.stdout
# sys.stdout = mystdout = io.StringIO()
# trainer.train()
# #trainer.evaluate()  # This prints metrics to console
# sys.stdout = old_stdout

# fname="results_"+args.dataset[0]+'_'+args.fl_algorithm+".txt"
# with open(fname, "w", encoding="utf-8") as f:
#     f.write(mystdout.getvalue())

# print(f"Evaluation results saved to {fname}")