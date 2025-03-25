import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model architecture parameters
LATENT_DIM = 4  # Reduced latent dimension for better compression
INPUT_DIM = (1, 28, 28)  # MNIST image shape

# Training hyperparameters
BATCH_SIZE = 64  # Reduced batch size for better stability
LEARNING_RATE = 0.0003  # Reduced learning rate
EPOCHS = 5  # Reduced epochs since we have a simpler model
VALIDATION_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 10  # Increased patience

# VAE-specific parameters
BETA_BASE = 1  # Beta weight for base VAE KL divergence term
BETA_GOAL = 0.25  # Beta weight for goal VAE KL divergence term
BETA_GMM = 5   # Beta weight for GMM VAE KL divergence term

# Dataset configuration
EXPERT_GOALS = [6]  # Target numbers for shifting the latent space
MNIST_PATH = "data"

# Model save paths
BASEPATH = "models/base.pth"  # Path to save/load base VAE model
GOALPATH = "models/goal.pth"  # Path to save/load goal VAE model
GMMPATH = "models/gmm.pth"  # Path to save/load GMM model


SAVE_IMAGES = True # Whether to save images to the image path
IMAGE_PATH = "images" # Path to save images

# Training control flags
FORCE_RETRAIN_BASE = False  # Force retrain base VAE with new architecture
FORCE_RETRAIN_GOAL = False  # Force retrain goal VAE
FORCE_RETRAIN_GMM = False   # Force retrain GMM VAE

# GMM parameters
GMM_N_COMPONENTS = 4  # Number of Gaussian components
GMM_COVARIANCE_TYPE = 'full'  # Type of covariance matrix for GMM
