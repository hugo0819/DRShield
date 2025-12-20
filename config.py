import torch

# ============================================================
# GNN+PPO Reinforcement Learning Hyperparameters
# ============================================================

# --- GNN Network Parameters ---
GNN_TYPE = 'gcn'           # GNN type: 'gcn' or 'gat'
GNN_HIDDEN_DIM = 128       # GNN hidden layer dimension
GNN_NUM_LAYERS = 3         # Number of GNN layers
GNN_DROPOUT = 0.1          # Dropout ratio
NODE_FEATURE_DIM = 7       # Node feature dimension

# --- PPO Reinforcement Learning Hyperparameters ---
PPO_LR = 3e-4              # Learning rate
PPO_GAMMA = 0.99           # Discount factor
PPO_EPS_CLIP = 0.2         # PPO clipping parameter
PPO_K_EPOCHS = 10          # Number of iterations per update
PPO_GAE_LAMBDA = 0.95      # GAE λ parameter
PPO_ENTROPY_COEF = 0.01    # Entropy regularization coefficient
PPO_VALUE_LOSS_COEF = 0.5  # Value loss coefficient
PPO_MAX_GRAD_NORM = 0.5    # Gradient clipping norm
PPO_UPDATE_INTERVAL = 10   # Update interval (every N episodes)

# --- Reward Function Weights (MR/FPR/RU/LTD) ---
W_MR = 0.4    # Mitigation Rate weight
W_FPR = 0.2   # False Positive Rate weight
W_RU = 0.2    # Resource Utilization weight
W_LTD = 0.2   # Latency weight

# --- Normalization Parameters ---
T_FPR = 0.05           # FPR tolerance upper limit (5%)
TAU_LATENCY = 10.0     # Expected latency (ms), used for LTD normalization
ALPHA_FPR = 20.0       # FPR exponential decay coefficient

# ============================================================
# DDPG Reinforcement Learning Hyperparameters (for compatibility)
# ============================================================
GAMMA = 0.99
TAU = 1e-2
LEARNING_RATE_ACTOR = 1e-3
LEARNING_RATE_CRITIC = 1e-3
REPLAY_MEM_CAPACITY = int(1e5)
MINI_BATCH_SIZE = 1024

# --- Exploration Noise Parameters ---
INITIAL_NOISE_SCALE = 0.1
NOISE_DECAY = 0.99

# ============================================================
# Training Loop Parameters
# ============================================================
NUM_EPISODES = 2000        # Total number of training episodes
EPISODE_LENGTH = 10        # Duration of each traffic episode (seconds)
MODEL_SAVE_INTERVAL = 500  # Model save interval

# ============================================================
# Network Topology Configuration
# ============================================================
EXPECTED_SWITCHES = 7      # Expected number of switches
EXPECTED_HOSTS = 8         # Expected number of hosts

# ============================================================
# Network and Device Configuration
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MININET_API_URL = "http://127.0.0.1:5001"
RYU_CONTROLLER_API_URL = "http://localhost:8080"

# ============================================================
# Attack Traffic Definition (for traffic counting)
# ============================================================
SPOOFED_SRC_IP = '10.1.1.1'
DEST_NAME = 'h8'
DEST_IP = '10.0.0.8'

# ============================================================
# Model Save Path
# ============================================================
MODEL_SAVE_PATH = "saved_models"

# ============================================================
# Algorithm Selection
# ============================================================
USE_GNN_PPO = True  # True: Use GNN+PPO, False: Use original PPO/DDPG
