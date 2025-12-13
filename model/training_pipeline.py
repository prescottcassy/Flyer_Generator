import os, sys, torch, torch.nn as nn, matplotlib.pyplot as plt, json, requests
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from PIL import Image

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
from diffusers import DiffusionPipeline

# Constants
IMG_SIZE = 512
IMG_CH = 4
n_steps = 1000
beta_start = 0.0001
beta_end = 0.02
num_epochs = 3

# DeepInfra setup
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
BASE_URL = "https://api.deepinfra.com/v1/inference"
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

# Pipeline placeholders
base = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion", use_auth_token=os.getenv("DEEPINFRA_API_KEY"))
refiner = None

def load_pipeline():
    """
    Instead of loading HuggingFace locally, store model names for DeepInfra.
    """
    global base, refiner
    if not DEEPINFRA_API_KEY:
        raise RuntimeError("DEEPINFRA_API_KEY is not set in Railway variables.")

    base = {"model_name": base_model_id}
    refiner = {"model_name": refiner_model_id}
    print("Using DeepInfra for inference. Pipelines initialized as model name references.")

def generate_image(pipeline, prompt, num_inference_steps=50):
    """
    Proxy image generation to DeepInfra API.
    """
    model_name = pipeline["model_name"]
    response = requests.post(
        f"{BASE_URL}/{model_name}",
        headers={"Authorization": f"Bearer {DEEPINFRA_API_KEY}"},
        json={"prompt": prompt, "num_inference_steps": num_inference_steps},
    )
    response.raise_for_status()
    return response.json()

# Initialize diffusion schedule
if __name__ == '__main__':
    init_diffusion(n_steps=n_steps, beta_start=beta_start, beta_end=beta_end, device_override=device)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gen-enabled', action='store_true')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--grad-accum-steps', type=int, default=4)
args, unknown = parser.parse_known_args()
scaler = GradScaler()

epoch_train_loss = []
epoch_val_loss = []

if __name__ == '__main__':
    train_loader, val_loader, test_loader = create_train_loader(
        data_dir='C:/Users/presc/AppData/Local/Temp/_tmp_flyers',
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
        splits=(0.8, 0.1, 0.1),
        seed=42,
        num_workers=0
    )

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

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            data, labels, metadata = batch
            data, labels = data.to(device), labels.to(device).long()
            t = torch.randint(0, n_steps, (data.size(0),), device=device)
            noisy_data, added_noise = add_noise(data, t)
            c = labels

            print(f"Data shape: {data.shape}, Labels: {labels}, t: {t}, c: {c}")

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

    input_data = torch.randn((args.batch_size, IMG_CH, IMG_SIZE, IMG_SIZE), device=device)
    benchmark_metrics(model, input_data, device=str(device))

    plt.plot(epoch_train_loss, label='Train Loss')
    plt.show()
    plt.plot(epoch_val_loss, label='Val Loss')
    plt.show()
