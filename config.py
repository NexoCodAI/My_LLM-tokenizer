import torch

# Model configuration
VOCAB_SIZE = None     # to be set after tokenizer initialization
D_MODEL = 384         # embedding dimension (increased from 256)
N_LAYERS = 6          # number of transformer blocks (increased from 3)
N_HEADS = 4           # attention heads per block
BLOCK_SIZE = 256      # context/window size (increased from 128)

# Data split for validation
VALIDATION_SPLIT = 0.1  # fraction of tokens reserved for validation

# Training hyperparameters
BATCH_SIZE = 64       # reduced to fit larger model
LEARNING_RATE = 3e-4  # base learning rate
WEIGHT_DECAY = 1e-2   # for optimizer regularization
LABEL_SMOOTHING = 0.1 # for CrossEntropyLoss

# Scheduler parameters for OneCycleLR
SCHEDULER = {
    'type': 'OneCycleLR',     # learning rate schedule
    'max_lr': LEARNING_RATE,  # peak learning rate
    'pct_start': 0.1,         # fraction of total steps for warmup
    'anneal_strategy': 'cos'  # cosine decay after warmup
}

# Training loop control
EPOCHS = 5            # total epochs (increased from 1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRADIENT_ACCUM_STEPS = 4  # accumulate gradients for larger effective batch size

# Early-stopping & evaluation
EVAL_INTERVAL = 1000        # run validation every N batches
VALID_LOSS_THRESHOLD = 1.0  # stop training when val loss < this
PATIENCE = 6               # max validation checks without improvement (increased from 3)

# Checkpointing
CHECKPOINT_DIR = "checkpoints"  # where to save model checkpoints
