import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import matplotlib.pyplot as plt
from PIL import Image

# Local imports
from .setup_env import (
    set_random_seed,
    device,
    SEED,
)

set_random_seed(SEED)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .utils import (
    validate_model_parameters,
    enable_cpu_offload,
    save_image,
    open_image_devcontainer,
    benchmark_metrics,
    log_pipeline_run,
    create_train_loader
)

from .diffusion_process import (
    init_diffusion,
    add_noise,
)

from .model import SDXL

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
IMG_SIZE = 512
IMG_CH = 4
n_steps = 1000
beta_start = 0.0001
beta_end = 0.02
num_epochs = 3

# Pipeline placeholder
base = None

# -------------------------------------------------------------------
# Pipeline loader
# -------------------------------------------------------------------
def load_pipeline():
    """
    Load your custom SDXL model and initialize diffusion schedule.
    """
    global base
    base = SDXL(
        img_channels=IMG_CH,
        img_size=IMG_SIZE,
        down_channels=(32, 64, 128, 256),
        t_embed_dim=128,
        c_embed_dim=128,
        num_classes=10
    ).to(device)

    init_diffusion(n_steps=n_steps, beta_start=beta_start, beta_end=beta_end, device_override=device)
    print("Custom SDXL pipeline loaded successfully.")

# -------------------------------------------------------------------
# Image generation using your model
# -------------------------------------------------------------------
def generate_image(pipeline, prompt, num_inference_steps=50):
    """
    Run inference using your custom SDXL model.
    """
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized.")

    with torch.no_grad():
        # Example: forward pass with noise schedule
        noise = torch.randn((1, pipeline.img_channels, IMG_SIZE, IMG_SIZE), device=device)
        t = torch.randint(0, num_inference_steps, (1,), device=device)
        c = torch.zeros((1,), device=device)  # dummy class label

        outputs = pipeline(noise, t, c)

    # Convert tensor to PIL image
    img = (outputs.squeeze().cpu().clamp(0, 1) * 255).byte()
    pil_img = Image.fromarray(img.numpy())
    return pil_img

# -------------------------------------------------------------------
# Training loop (optional, for local training or DeepInfra acceleration)
# -------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen-enabled', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    args, unknown = parser.parse_known_args()
    scaler = GradScaler()

    # Data loaders
    train_loader, val_loader, test_loader = create_train_loader(
        data_dir='C:/Users/presc/AppData/Local/Temp/_tmp_flyers',
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
        splits=(0.8, 0.1, 0.1),
        seed=42,
        num_workers=0
    )

    # Model setup
    model = SDXL(
        img_channels=IMG_CH,
        img_size=IMG_SIZE,
        down_channels=(32, 64, 128, 256),
        t_embed_dim=128,
        c_embed_dim=128,
        num_classes=10
    ).to(device)

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    writer = SummaryWriter()

    epoch_train_loss = []
    epoch_val_loss = []

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            data, labels, metadata = batch
            data, labels = data.to(device), labels.to(device).long()
            t = torch.randint(0, n_steps, (data.size(0),), device=device)
            noisy_data, added_noise = add_noise(data, t)
            c = labels

            validate_model_parameters(model)
            enable_cpu_offload(model)

            with autocast(device_type='cuda'):
                outputs = model(noisy_data, t, c)
                loss = criterion(outputs, added_noise)

            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

            scaler.scale(loss).backward()
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

    if writer:
        writer.close()

    # Inference check
    print("Before gen check")
    load_pipeline()
    if args.gen_enabled and base is not None:
        try:
            prompt = "A flyer that matches the brand for a mimosa and mingle event on November 8th from 10 to noon"
            num_inference_steps = 50
            generated = generate_image(base, prompt, num_inference_steps=num_inference_steps)
            save_image(generated, "generated_image.png")
            print("Image saved at: generated_image.png")
            log_pipeline_run(prompt, num_inference_steps, 0)
        except Exception as e:
            print(f"Error during image generation: {e}")

    # Benchmark
    input_data = torch.randn((args.batch_size, IMG_CH, IMG_SIZE, IMG_SIZE), device=device)
    benchmark_metrics(model, input_data, device=str(device))
