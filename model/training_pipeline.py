import sys, os, json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

# These imports are only needed for training and are lazy-loaded below
torch = None
nn = None
plt = None
DiffusionPipeline = None
InferenceClient = None
Adam = None
SummaryWriter = None
GradScaler = None
autocast = None
Image = None
set_random_seed = None
device = None
SEED = None
validate_model_parameters = None
enable_cpu_offload = None
save_image = None
open_image_devcontainer = None
benchmark_metrics = None
log_pipeline_run = None
create_train_loader = None
init_diffusion = None
add_noise = None
SDXL = None

from .utils import generate_image

def _lazy_load_training_deps():
    """Lazily load training-specific dependencies only when needed."""
    global torch, nn, plt, DiffusionPipeline, InferenceClient, Adam, SummaryWriter
    global GradScaler, autocast, Image, set_random_seed, device, SEED
    global validate_model_parameters, enable_cpu_offload, save_image
    global open_image_devcontainer, benchmark_metrics, log_pipeline_run, create_train_loader
    global init_diffusion, add_noise, SDXL
    
    if torch is not None:
        return  # Already loaded
    
    import torch as torch_
    import torch.nn as nn_
    import matplotlib.pyplot as plt_
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline as DP
    from huggingface_hub import InferenceClient as IC
    from torch.optim import Adam as Adam_
    from torch.utils.tensorboard import SummaryWriter as SW
    from torch.amp.grad_scaler import GradScaler as GS
    from torch.amp.autocast_mode import autocast as autocast_
    from PIL import Image as Image_
    
    from .setup_env import set_random_seed as srs, device as d, SEED as s
    from .utils import (
        validate_model_parameters as vmp,
        enable_cpu_offload as eco,
        save_image as si,
        open_image_devcontainer as oid,
        benchmark_metrics as bm,
        log_pipeline_run as lpr,
        create_train_loader as ctl,
    )
    from .diffusion_process import init_diffusion as id_, add_noise as an_
    from .model import SDXL as SDXL_
    
    torch = torch_
    nn = nn_
    plt = plt_
    DiffusionPipeline = DP
    InferenceClient = IC
    Adam = Adam_
    SummaryWriter = SW
    GradScaler = GS
    autocast = autocast_
    Image = Image_
    set_random_seed = srs
    device = d
    SEED = s
    validate_model_parameters = vmp
    enable_cpu_offload = eco
    save_image = si
    open_image_devcontainer = oid
    benchmark_metrics = bm
    log_pipeline_run = lpr
    create_train_loader = ctl
    init_diffusion = id_
    add_noise = an_
    SDXL = SDXL_
    
    set_random_seed(SEED)

# Define refiner_model_id globally
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

# Image configuration
IMG_SIZE = 512
IMG_CH = 3

# Initialize base and refiner pipelines separately
base = None
refiner = None

def load_pipeline():
    _lazy_load_training_deps()  # Load training dependencies on first call
    global base, refiner
    # Ensure SERVICEACCOUNT is loaded and provide a fallback error message
    api_key = os.getenv('SERVICEACCOUNT')
    if not api_key:
        raise RuntimeError("SERVICEACCOUNT is not set. Please check your secrets.env file.")

    client = InferenceClient(
        provider="auto",
        api_key=api_key,
    )

    # Force all operations to run on the CPU
    device = torch.device("cpu")
    print("Using CPU for all operations.")

    # Clean GPU memory before loading pipelines
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Initialize base and refiner pipelines separately
    base = DiffusionPipeline.from_pretrained(base_model_id)
    base.to(device)

    try:
        refiner = DiffusionPipeline.from_pretrained(refiner_model_id)
        if refiner:
            refiner.to(device)
    except torch.OutOfMemoryError:
        print("Out of GPU memory. Attempting to load refiner on CPU.")
        try:
            refiner = DiffusionPipeline.from_pretrained(refiner_model_id)
            if refiner:
                refiner.to("cpu")
        except Exception as e:
            print(f"Failed to initialize refiner pipeline on CPU: {e}")
        refiner = None
    except Exception as e:
        print(f"Failed to initialize refiner pipeline: {e}")
        refiner = None

    # Check if refiner is initialized before proceeding
    if not refiner:
        print("Warning: Refiner pipeline is not initialized. Proceeding without it.")

    # Set PyTorch CUDA memory configuration to reduce fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Try to move the refiner pipeline to the GPU
    # Ensure refiner is not None before moving to device
    if refiner is not None:
        try:
            refiner.to(device)
        except torch.OutOfMemoryError:
            print("Out of GPU memory. Attempting to load refiner on CPU.")
            refiner.to("cpu")
    else:
        print("Warning: Refiner pipeline is not initialized. Proceeding without it.")

# Initialize the diffusion noise schedule
n_steps = 1000
beta_start = 0.0001
beta_end = 0.02
init_diffusion(n_steps=n_steps, beta_start=beta_start, beta_end=beta_end, device_override=device)

num_epochs = 3
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gen-enabled', action='store_true')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--grad-accum-steps', type=int, default=4)
args, unknown = parser.parse_known_args()
args = args

# Enable mixed precision training to reduce memory usage
scaler = GradScaler(init_scale=2.0)  # Updated to use recommended syntax for 'torch.amp.GradScaler'

writer = SummaryWriter()
epoch_train_loss = []
epoch_val_loss = []

if __name__ == '__main__':
    # Wrap the main execution logic inside this guard

    # Correct the unpacking of the tuple returned by create_train_loader
    train_loader, val_loader, test_loader = create_train_loader(
        data_dir='C:/Users/presc/AppData/Local/Temp/_tmp_flyers',  # Updated to use the temporary directory
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
        splits=(0.8, 0.1, 0.1),
        seed=42,
        num_workers=0  # Set to 0 to avoid multiprocessing issues on Windows
    )

    # Ensure model is initialized before use
    model = SDXL(
        img_channels=IMG_CH,
        img_size=IMG_SIZE,
        down_channels=(32, 64, 128, 256),
        t_embed_dim=128,
        c_embed_dim=128,
        num_classes=10
    ).to(device)

    # Define the optimizer and loss function after model initialization
    optimizer = Adam(model.parameters(), lr=0.001)  # Define the optimizer
    criterion = nn.MSELoss()  # Define the loss function for diffusion (MSE between predicted and added noise)

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            data, labels, metadata = batch
            data, labels = data.to(device), labels.to(device).long()

            # Sample random timesteps
            t = torch.randint(0, n_steps, (data.size(0),), device=device)

            # Add noise to the data
            noisy_data, added_noise = add_noise(data, t)

            # Conditioning input from labels
            c = labels

            # Debugging logs
            print(f"Data shape: {data.shape}, Labels: {labels}, t: {t}, c: {c}")

            validate_model_parameters(model)
            enable_cpu_offload(model)

            with autocast(device_type='cuda'):
                # Forward pass
                outputs = model(noisy_data, t, c)
                loss = criterion(outputs, added_noise)

            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

            # Scale the loss and backpropagate
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

    if writer:
        writer.close()

    print("Before gen check")
    print(f"args.gen_enabled: {args.gen_enabled}")
    print(f"base is not None: {base is not None}")
    if args.gen_enabled and base is not None:
        print(f"gen_enabled: {args.gen_enabled}, base is not None: {base is not None}")
        print("Starting image generation...")
        # Use the trained model in the pipeline
        # base.unet = model  # Temporarily comment out to use original model
        try:
            prompt = "A flyer that matches the brand for a mimosa and mingle event on November 8th from 10 to noon"
            num_inference_steps = 50
            generated_image = generate_image(base, prompt, num_inference_steps=num_inference_steps)
            save_image(generated_image, "generated_image.png")
            print("Image saved at: generated_image.png")  # Log the save path

            # Display the saved image
            try:
                import os
                from PIL import Image

                img = Image.open("generated_image.png")
                temp_path = "generated_image_temp.png"
                img.save(temp_path)
                open_image_devcontainer(temp_path)
            except Exception as e:
                print(f"Error displaying the image: {e}")
        
            log_pipeline_run(prompt, num_inference_steps, 0)
        except Exception as e:
            print(f"Error during image generation: {e}")

    input_data = torch.randn((args.batch_size, IMG_CH, IMG_SIZE, IMG_SIZE), device=device)
    benchmark_metrics(model, input_data, device=str(device))

    # Data visualization
    plt.plot(epoch_train_loss, label='Train Loss')
    plt.show()
    plt.plot(epoch_val_loss, label='Val Loss')  
    plt.show()
