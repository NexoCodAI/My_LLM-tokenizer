import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Model
VOCAB_SIZE = None
D_MODEL     = 384
N_LAYERS    = 6
N_HEADS     = 4
BLOCK_SIZE  = 128

# Data
VALIDATION_SPLIT = 0.1

# Hyperparams
BATCH_SIZE      = 128
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-2
LABEL_SMOOTHING = 0.1

# Scheduler → plateau‑aware
SCHEDULER = {
    'type': 'ReduceLROnPlateau',
    'mode': 'min',
    'factor': 0.5,
    'patience': 1,
    'min_lr': 1e-6,
    'verbose': True
}

# Training
EPOCHS               = 3    # you can crank this up after plateau break
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRADIENT_ACCUM_STEPS = 8

# Early‑stop / eval
EVAL_INTERVAL       = 1000
VALID_LOSS_THRESHOLD = 1.0
PATIENCE             = 6

CHECKPOINT_DIR = "checkpoints"
