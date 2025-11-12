# Your PyTorch model setup and training logic
# install_einops.py
import random  # For random operations
import numpy as np  # For numerical operations on arrays
import subprocess
import sys

# Ensure einops is available before importing its submodules. If missing,
# install it automatically (best-effort).
try:
    import einops
    from einops.layers.torch import Rearrange  # For reshaping tensors in neural networks
    from einops import rearrange  # For elegant tensor reshaping operations
except Exception:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "einops"])
        # Try import again after installation
        import einops
        from einops.layers.torch import Rearrange
        from einops import rearrange
        print("Installed and imported 'einops'.")
    except Exception as e:
        # If installation fails, raise a clear error to the user
        raise ImportError("Required package 'einops' is missing and could not be installed automatically: " + str(e))

# --- Early deterministic setup -------------------------------------------------
# Set the global seed and PYTHONHASHSEED before other imports/operations that
# --- System utilities ---
import os  # For operating system interactions (used for CPU count)
SEED = 42  # Universal seed value for reproducibility
os.environ["PYTHONHASHSEED"] = str(SEED)


# -- Core PyTorch libraries ---
import torch  # Main deep learning framework
import torch.nn.functional as F  # Neural network functions like activation functions
import torch.nn as nn  # Neural network building blocks (layers)
from torch.optim import Adam  # Optimization algorithm for training

# --- Data handling ---
from torch.utils.data import Dataset, DataLoader, random_split  # For organizing and loading our data
try:
    import torchvision  # Library for computer vision datasets and models
    import torchvision.transforms as transforms  # For preprocessing images
except Exception:
    raise ImportError(
        "Missing required package 'torchvision'. Please install it, e.g. `pip install torchvision`, "
        "or run `pip install -r model/requirements.txt` from the repository root."
    )

# --- Visualization tools ---
import matplotlib.pyplot as plt  # For plotting images and graphs
from PIL import Image  # For image processing
from torchvision.utils import save_image, make_grid  # For saving and displaying image grids
from importlib.metadata import version as pkg_version
from typing import Any

# Set up device (GPU or CPU) and print helpful diagnostics
# 1) Get PyTorch package version string
print(torch.__version__)

# 3) Alternative: query installed package metadata (no runtime import of torch required)
print("pkg metadata version:", pkg_version("torch"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Reproducibility and Device Config
# Step 4: Set random seeds for reproducibility
def set_random_seed(seed: int):
    """
    Set the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The random seed value to set.
    """
    torch.manual_seed(seed)          # PyTorch random number generator
    np.random.seed(seed)             # NumPy random number generator
    random.seed(seed)                # Python's built-in random number generator
    print(f"Random seeds set to {SEED} for reproducible results")
    
    # Configure CUDA for GPU operations if available
    if torch.cuda.is_available():
        # GPU path: seed GPU RNGs and print memory diagnostics
        torch.cuda.manual_seed(seed)       # GPU random number generator
        torch.cuda.manual_seed_all(seed)   # All GPUs random number generator
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            # Check available GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
            print(f"Available GPU Memory: {gpu_memory:.1f} GB")
            # Add recommendation based on memory
            if gpu_memory < 4:
                print("Warning: Low GPU memory. Consider reducing batch size if you encounter OOM errors.")
        except Exception as e:
            print(f"Could not check GPU memory: {e}")
    else:
        # CPU path
        print("No GPU detected. Training will be much slower on CPU.")
        print("If you're using Colab, go to Runtime > Change runtime type and select GPU.")

# Added default image size and channels for consistency
IMG_SIZE = 512
IMG_CH = 3
