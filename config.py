import os
import torch

# Image
IMG_SIZE = 224

# Training
BATCH_SIZE   = 32
EPOCHS       = 10
FREEZE_EPOCHS = 5      # train only the head for this many epochs, then unfreeze all
LR_HEAD      = 1e-3    # learning rate while backbone is frozen
LR_FULL      = 1e-4    # learning rate after unfreezing backbone
TRAIN_RATIO  = 0.85
RANDOM_SEED  = 42
NUM_WORKERS  = 2

# Device — prefer CUDA, then Apple MPS, then CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Paths
OUTPUT_DIR = "outputs"
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
