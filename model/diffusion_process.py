# SDXL Diffusion Process Implementation
# This module defines the SDXL diffusion process, including noise scheduling
import torch

# This module provides utility functions for the diffusion process. It no
# longer performs dataset or RNG initialization at import time. Callers should
# initialize randomness and datasets explicitly (for example, from
# `train_model.py`).

# Module-level placeholders for precomputed noise schedule tensors. Call
# `init_diffusion()` to populate these before using `add_noise`/`denoise`.
beta = None
alpha = None
alpha_cumprod = None
sqrt_alpha_cumprod = None
sqrt_one_minus_alpha_cumprod = None


def init_diffusion(n_steps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02, device_override=None):
    """Initialize the noise schedule tensors used by add_noise/denoise.

    This must be called before using add_noise/denoise so module-level
    schedule tensors are available.

    Args:
        n_steps: number of diffusion timesteps.
        beta_start: start of beta schedule.
        beta_end: end of beta schedule.
        device_override: optional torch.device to place tensors on. If None,
            uses the module-level `device` imported from setup_env.
    """
    global beta, alpha, alpha_cumprod, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod
    dev = device_override if device_override is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check GPU memory usage only if the device is CUDA
    if dev.type == "cuda" and torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(dev).total_memory
        reserved_memory = torch.cuda.memory_reserved(dev)
        available_memory = total_memory - reserved_memory

        # If available memory is less than a threshold, force CPU execution
        if available_memory < 2 * 1024**3:  # Less than 2GB available
            print("Insufficient GPU memory. Forcing CPU execution.")
            dev = torch.device("cpu")
    else:
        print("CUDA not available or using CPU for diffusion initialization.")

    # Use torch.no_grad to prevent gradient tracking
    with torch.no_grad():
        try:
            beta = torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float16).clamp(max=0.999).to(dev)
            alpha = (1.0 - beta).to(dev)
            alpha_cumprod = torch.cumprod(alpha, dim=0).to(dev)
            sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod).to(dev)
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod).to(dev)
        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory in init_diffusion. Falling back to CPU.")
            dev = torch.device("cpu")
            beta = torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float16).clamp(max=0.999).to(dev)
            alpha = (1.0 - beta).to(dev)
            alpha_cumprod = torch.cumprod(alpha, dim=0).to(dev)
            sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod).to(dev)
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod).to(dev)

# Add noise to the input image
def add_noise(x0, t):
    """Add noise to the input tensor x0 at timestep t."""
    if sqrt_alpha_cumprod is None or sqrt_one_minus_alpha_cumprod is None:
        raise ValueError("Global variables not initialized. Call `init_diffusion` first.")
    noise = torch.randn_like(x0)
    # Ensure `t` is on the same device as the tensors being indexed
    t = t.to(sqrt_alpha_cumprod.device)
    x_t = sqrt_alpha_cumprod[t] * x0 + sqrt_one_minus_alpha_cumprod[t] * noise
    return x_t, noise