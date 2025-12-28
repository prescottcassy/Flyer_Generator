"""
Gradio App for Flyer Generation with SDXL
Uses Gradio Zero GPU for serverless GPU compute
"""
import gradio as gr
from gradio.themes import Soft
import spaces
from model.model import SDXLModel
from model.utils import create_output_dir
from datetime import datetime
import os
from PIL import Image


# Global model instance (loaded on-demand)
model = None


def get_model():
    """Lazy load the model."""
    global model
    if model is None:
        model = SDXLModel(use_refiner=True)
    return model


@spaces.GPU(duration=120)  # Allocate GPU for 120 seconds
def generate_flyer(
    prompt: str,
    negative_prompt: str = "blurry, bad quality, distorted, ugly",
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    use_refiner: bool = True
):
    """
    Generate a flyer using SDXL.
    
    This function runs on a GPU provided by Gradio Spaces.
    """
    if not prompt:
        return None, "Please enter a prompt!"
    
    # Get or create model
    sdxl_model = get_model()
    
    # Use random seed if -1
    actual_seed = None if seed == -1 else seed
    
    # Generate image
    try:
        image = sdxl_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=actual_seed,
            num_images=1
        )
        
        # Save the generated image
        if image is not None:
            output_dir = create_output_dir("outputs")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"flyer_{timestamp}.png"
            if isinstance(image, Image.Image):
                image.save(output_path)
            elif isinstance(image, list) and len(image) > 0:
                if isinstance(image[0], Image.Image):
                    image[0].save(output_path)
            else:
                # Convert numpy array or tensor to PIL Image
                import numpy as np
                import torch
                if isinstance(image, torch.Tensor):
                    image = image.cpu()
                    if hasattr(image, 'numpy'):
                        image = image.numpy()
                if isinstance(image, np.ndarray):
                    image = Image.fromarray((image * 255).astype(np.uint8)) if image.max() <= 1 else Image.fromarray(image.astype(np.uint8))
                    image.save(output_path)
                
        info = f"✅ Generated successfully!\nSeed: {actual_seed if actual_seed else 'random'}\nSteps: {num_steps}"
        
        return image, info
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# Create Gradio interface
def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Image Generator", theme=Soft()) as demo:
        gr.Markdown(
            """
            # 🎨 Image Generator
            Generate professional images using Stable Diffusion XL.
            Powered by Gradio Zero GPU for fast, serverless inference.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe your image: 'A professional business flyer for a coffee shop...'",
                    lines=3
                )
                
                with gr.Row():
                    width = gr.Slider(
                        minimum=512,
                        maximum=1536,
                        value=1024,
                        step=128,
                        label="Width"
                    )
                    height = gr.Slider(
                        minimum=512,
                        maximum=1536,
                        value=1024,
                        step=128,
                        label="Height"
                    )
                
                with gr.Row():
                    num_steps = gr.Slider(
                        minimum=20,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Inference Steps"
                    )
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale"
                    )
                
                seed = gr.Number(
                    label="Seed (-1 for random)",
                    value=-1,
                    precision=0
                )
                
                use_refiner = gr.Checkbox(
                    label="Use Refiner (better quality, slower)",
                    value=True
                )
                
                generate_btn = gr.Button("🚀 Generate Image", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Generated Image",
                    type="pil"
                )
        
        # Connect the generate button
        generate_btn.click(
            fn=generate_flyer,
            inputs=[prompt, num_steps, guidance_scale, width, height, seed, use_refiner],
            outputs=[output_image]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
