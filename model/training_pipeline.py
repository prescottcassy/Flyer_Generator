# Custom imports for environment setup, datasets, diffusion utils, and model
# To run, enter this in the terminal:
# 'function RunTraining { & "$PSScriptRoot\venv\Scripts\python.exe" "$PSScriptRoot\model\training_pipeline.py" @args }'
# or 'cd /path/to/BrandBeast./venv/bin/python ./model/training_pipeline.py --gen-enabled' in bash terminal
from setup_env import set_random_seed, device, SEED
set_random_seed(SEED)
from datasets import (
    make_datasets_and_loaders, 
    IMG_SIZE, 
    IMG_CH,
    create_train_loader
)

import torch, torch.nn as nn, sys, os, matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from dotenv import load_dotenv
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from huggingface_hub import InferenceClient
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from PIL import Image

from utils import (
    validate_model_parameters,
    enable_cpu_offload,
    generate_image,
    save_image,
    benchmark_metrics,
    log_pipeline_run,
)

from sdxl_diffusion import (
    init_diffusion,
    add_noise,
)

from model.NueralNetwork import SDXL

load_dotenv("model/secrets.env")

# Ensure HF_TOKEN is loaded and provide a fallback error message
api_key = os.getenv('HF_TOKEN')
if not api_key:
    raise RuntimeError("HF_TOKEN is not set. Please check your secrets.env file.")

client = InferenceClient(
    provider="auto",
    api_key=api_key,
)

# Define refiner_model_id globally
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

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
args = parser.parse_args()

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

            # Display the saved image using os.startfile as a fallback
            try:
                import os
                from PIL import Image

                img = Image.open("generated_image.png")
                img.save("generated_image_temp.png")
                os.startfile("generated_image_temp.png")
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
