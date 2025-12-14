from setup_env import (
    set_random_seed, 
    device, 
    SEED
)
set_random_seed(SEED)
from datasets import (
    make_datasets_and_loaders, 
    IMG_SIZE, 
    IMG_CH,
    create_train_loader
)

import torch, torch.nn as nn, sys, os, matplotlib.pyplot as plt, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

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
    open_image_devcontainer,
    benchmark_metrics,
    log_pipeline_run,
)

from sdxl_diffusion import (
    init_diffusion,
    add_noise,
)

from model.NueralNetwork import SDXL

# Ensure SERVICEACCOUNT is loaded and provide a fallback error message
api_key = os.getenv('SERVICEACCOUNT')
if not api_key:
    raise RuntimeError("SERVICEACCOUNT is not set. Please check your secrets.env file.")

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

def train_model(train_loader, val_loader, model, optimizer, criterion, num_epochs, device, scaler, writer, args):
    """
    Main training loop for the SDXL diffusion model.
    
    Args:
        train_loader: PyTorch DataLoader for training data
        val_loader: PyTorch DataLoader for validation data
        model: SDXL model instance
        optimizer: PyTorch optimizer
        criterion: Loss function
        num_epochs: Number of training epochs
        device: Device to train on (cuda/cpu)
        scaler: GradScaler for mixed precision training
        writer: TensorBoard SummaryWriter
        args: Parsed command line arguments
    """
    print(f"Starting training on device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.grad_accum_steps}")
    
    global epoch_train_loss, epoch_val_loss
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                data, labels, metadata = batch
                data, labels = data.to(device), labels.to(device).long()

                # Sample random timesteps
                t = torch.randint(0, n_steps, (data.size(0),), device=device)

                # Add noise to the data using the diffusion process
                noisy_data, added_noise = add_noise(data, t)

                # Conditioning input from labels
                c = labels

                # Validate model parameters before forward pass
                validate_model_parameters(model)
                enable_cpu_offload(model)

                # Forward pass with mixed precision
                device_type = 'cuda' if device.type == 'cuda' else 'cpu'
                with autocast(device_type=device_type):
                    # Predict noise from noisy image
                    predicted_noise = model(noisy_data, t, c)
                    loss = criterion(predicted_noise, added_noise)

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                train_loss += loss.item()
                num_batches += 1

                # Gradient accumulation
                if (batch_idx + 1) % args.grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Log loss to TensorBoard
                    global_step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar('Loss/train_step', loss.item(), global_step)

                if (batch_idx + 1) % 10 == 0:
                    print(f"Batch {batch_idx + 1}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"Error in training batch {batch_idx}: {str(e)}")
                continue

        # Calculate epoch average loss
        avg_train_loss = train_loss / max(num_batches, 1)
        epoch_train_loss.append(avg_train_loss)
        
        print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    data, labels, metadata = batch
                    data, labels = data.to(device), labels.to(device).long()

                    t = torch.randint(0, n_steps, (data.size(0),), device=device)
                    noisy_data, added_noise = add_noise(data, t)
                    c = labels

                    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
                    with autocast(device_type=device_type):
                        predicted_noise = model(noisy_data, t, c)
                        loss = criterion(predicted_noise, added_noise)

                    val_loss += loss.item()
                    num_val_batches += 1

                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue

        avg_val_loss = val_loss / max(num_val_batches, 1)
        epoch_val_loss.append(avg_val_loss)
        
        print(f"Epoch {epoch + 1} - Average Validation Loss: {avg_val_loss:.4f}")
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        
        # Save model checkpoint
        checkpoint_path = f"model_checkpoint_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved: {checkpoint_path}")


def generate_images_from_model(base, refiner, prompts, num_inference_steps=50):
    """
    Generate images using the SDXL diffusion pipeline.
    
    Args:
        base: Base SDXL pipeline
        refiner: Refiner SDXL pipeline (optional)
        prompts: List of text prompts for image generation
        num_inference_steps: Number of inference steps
    """
    print(f"\nGenerating images with {num_inference_steps} inference steps...")
    
    if base is None:
        print("Error: Base pipeline is not initialized. Cannot generate images.")
        return

    generated_images = []
    
    for idx, prompt in enumerate(prompts):
        try:
            print(f"Generating image {idx + 1}/{len(prompts)}: {prompt[:50]}...")
            
            generated_image = generate_image(
                base, 
                prompt, 
                num_inference_steps=num_inference_steps
            )
            
            image_name = f"generated_image_{idx + 1}.png"
            save_image(generated_image, image_name)
            print(f"Image saved: {image_name}")
            
            generated_images.append((image_name, prompt))
            
            # Log to pipeline run
            log_pipeline_run(prompt, num_inference_steps, idx)
            
        except Exception as e:
            print(f"Error generating image for prompt '{prompt}': {str(e)}")
            continue
    
    return generated_images


if __name__ == '__main__':
    # Wrap the main execution logic inside this guard
    
    print("=" * 60)
    print("SDXL Diffusion Model Training Pipeline")
    print("=" * 60)
    
    try:
        # Resolve data directory path based on OS
        import platform
        if platform.system() == "Windows":
            data_dir = 'C:/Users/presc/AppData/Local/Temp/_tmp_flyers'
        else:
            # For Linux/macOS, use the data directory in the project
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', '_tmp_flyers')
        
        print(f"Loading data from: {data_dir}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_train_loader(
            data_dir=data_dir,
            image_size=IMG_SIZE,
            batch_size=args.batch_size,
            splits=(0.8, 0.1, 0.1),
            seed=42,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )

        # Initialize the SDXL model
        print("\nInitializing SDXL model...")
        model = SDXL(
            img_channels=IMG_CH,
            img_size=IMG_SIZE,
            down_channels=(32, 64, 128, 256),
            t_embed_dim=128,
            c_embed_dim=128,
            num_classes=10
        ).to(device)
        
        print(f"Model initialized. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Initialize optimizer and loss function
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        print(f"Optimizer: Adam (lr=0.001)")
        print(f"Loss Function: MSELoss")

        # Train the model
        print("\n" + "=" * 60)
        print("Starting Training Loop")
        print("=" * 60)
        
        train_model(
            train_loader, 
            val_loader, 
            model, 
            optimizer, 
            criterion, 
            num_epochs, 
            device, 
            scaler, 
            writer, 
            args
        )

        print("\n" + "=" * 60)
        print("Training Complete")
        print("=" * 60)

        # Close TensorBoard writer
        if writer:
            writer.close()
            print("TensorBoard logs saved.")

        # Image generation phase
        print("\nBefore generation check:")
        print(f"  Generation enabled: {args.gen_enabled}")
        print(f"  Base pipeline available: {base is not None}")
        
        if args.gen_enabled and base is not None:
            print("\n" + "=" * 60)
            print("Starting Image Generation")
            print("=" * 60)
            
            # Define sample prompts for generation
            generation_prompts = [
                "A flyer that matches the brand for a mimosa and mingle event on November 8th from 10 to noon",
                "A professional business flyer with modern design elements",
                "A colorful event flyer for a summer party"
            ]
            
            generated_images = generate_images_from_model(
                base, 
                refiner, 
                generation_prompts, 
                num_inference_steps=50
            )
            
            print(f"\nGenerated {len(generated_images)} images successfully")

        # Benchmark the model
        print("\n" + "=" * 60)
        print("Running Model Benchmarks")
        print("=" * 60)
        
        input_data = torch.randn((args.batch_size, IMG_CH, IMG_SIZE, IMG_SIZE), device=device)
        benchmark_metrics(model, input_data, device=str(device))

        # Visualize training metrics
        print("\n" + "=" * 60)
        print("Generating Training Visualizations")
        print("=" * 60)
        
        if epoch_train_loss:
            plt.figure(figsize=(10, 5))
            plt.plot(epoch_train_loss, label='Training Loss', marker='o')
            plt.plot(epoch_val_loss, label='Validation Loss', marker='s')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Over Epochs')
            plt.legend()
            plt.grid(True)
            plt.savefig('training_loss.png')
            print("Loss plot saved: training_loss.png")
            plt.show()

        print("\n" + "=" * 60)
        print("Pipeline Execution Completed Successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
