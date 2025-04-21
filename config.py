import torch

# Model configuration
VOCAB_SIZE = None     # to be set after tokenizer initialization
D_MODEL = 256         # embedding dimension
N_LAYERS = 3          # number of transformer blocks
N_HEADS = 4           # attention heads per block
BLOCK_SIZE = 128      # context/window size

# Data split for validation
VALIDATION_SPLIT = 0.1  # fraction of tokens reserved for validation

# Training hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
epochs_to_run = 1
EPOCHS = epochs_to_run  # total epochs (can override via CLI)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRADIENT_ACCUM_STEPS = 4  # accumulate gradients for larger effective batch size

# Early-stopping & evaluation
EVAL_INTERVAL = 1000        # run validation every N batches
VALID_LOSS_THRESHOLD = 1.0  # stop training when val loss < this
PATIENCE = 3                # max validation checks without improvement

# Checkpointing
CHECKPOINT_DIR = "checkpoints"  # where to save model checkpoints
