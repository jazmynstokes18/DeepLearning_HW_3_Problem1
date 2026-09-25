# Step 1 — Setup: imports, device selection, reproducibility seed, plot defaults.
# Everything downstream (data loading, model, training loop, plots) depends on the
# `device` variable and the fixed seed defined here, and this cell must be run first.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import matplotlib.pyplot as plt

# Pick the fastest device available on this machine.
# Order matters: CUDA (NVIDIA GPU) first, then Apple Silicon's "mps" backend
# (what this Mac uses), then CPU as the universal fallback so the notebook still
# runs anywhere.
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Fix the random seed once, up front, so the dataset split, weight initialization,
# and DataLoader shuffling are all reproducible on every rerun of this notebook.
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Matplotlib defaults applied notebook-wide, so every later figure (e.g. the
# training/validation accuracy curves) is consistently sized and legible without
# repeating these settings near each plot.
plt.rc("font", size=12)
plt.rc("axes", labelsize=12, titlesize=14)
plt.rc("legend", fontsize=12)
plt.rc("figure", figsize=(8, 5))

print(f"PyTorch version : {torch.__version__}")
print(f"Selected device : {device}")


# Sanity check for the setup cell: confirms torchmetrics imported correctly,
# a tensor can actually be moved to the selected device, and the seed is doing its
# job (two manual_seed(42) calls in a row must reproduce the same values).
assert isinstance(device, str) and device in {"cuda", "mps", "cpu"}

_probe = torch.randn(3, 3).to(device)
assert _probe.device.type == device
print(f"Tensor successfully placed on '{device}', shape {tuple(_probe.shape)}")

torch.manual_seed(SEED)
_a = torch.rand(5)
torch.manual_seed(SEED)
_b = torch.rand(5)
assert torch.equal(_a, _b), "Seeding is not reproducible!"
print("Seed reproducibility check passed.")

# torchmetrics import check: build a trivial metric object (no data yet).
_metric_check = torchmetrics.Accuracy(task="multiclass", num_classes=10)
print(f"torchmetrics OK: {_metric_check}")

del _probe, _a, _b, _metric_check  # scratch variables only, not needed later
