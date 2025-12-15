"""
Modal GPU Function for SDXL Flyer Generation
Deploy with: modal deploy modal_gpu.py
"""

import modal
import base64
from io import BytesIO

app = modal.App("flyer-generator")

# Build container image with all dependencies
image = modal.Image.debian_slim().pip_install(
    "diffusers==0.30.3",
    "torch==2.5.1",
    "transformers==4.46.3",
    "accelerate==1.1.1",
    "safetensors==0.4.5",
    "pillow==10.4.0"
)

@app.function(
    gpu="A10G",  # NVIDIA A10G GPU (~$1.10/hour, only when running)
    image=image,
    timeout=300,  # 5 minute timeout
    container_idle_timeout=120,  # Keep warm for 2 minutes between requests
    memory=16384  # 16GB RAM
)
def generate_flyer(prompt: str, num_inference_steps: int = 50):
    """
    Generate a flyer image using Stable Diffusion XL
    
    Args:
        prompt: Text description of the flyer to generate
        num_inference_steps: Number of denoising steps (higher = better quality, slower)
    
    Returns:
        Base64-encoded PNG image string
    """
    from diffusers import StableDiffusionXLPipeline
    import torch
    
    print(f"Loading SDXL model...")
    
    # Load SDXL pipeline (automatically cached after first run)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,  # Use half precision for speed
        use_safetensors=True,
        variant="fp16"
    )
    pipe.to("cuda")
    
    print(f"Generating image with prompt: {prompt[:50]}...")
    
    # Generate image
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=7.5,  # How closely to follow the prompt
        height=1024,
        width=1024
    )
    
    image = result.images[0]
    
    print("Image generated successfully, encoding to base64...")
    
    # Convert PIL Image to base64 string
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return img_base64


@app.local_entrypoint()
def test():
    """Test the function locally"""
    print("Testing Modal GPU function...")
    result = generate_flyer.remote(
        "A vibrant music festival flyer with colorful lights and crowd",
        num_inference_steps=30
    )
    print(f"Success! Generated image (base64 length: {len(result)})")