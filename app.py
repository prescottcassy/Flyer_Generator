"""
Gradio App for Flyer Generation with SDXL
Uses Gradio Zero GPU for serverless GPU compute
"""
import gradio as gr
from gradio.themes import Soft
import spaces
from model.model import SDXLModel
from model.utils import save_image, create_output_dir
from datetime import datetime
import os


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
        images = sdxl_model.generate(
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
        output_dir = create_output_dir("outputs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"flyer_{timestamp}.png"
        save_image(images[0], output_path)
        
        info = f"✅ Generated successfully!\nSeed: {actual_seed if actual_seed else 'random'}\nSteps: {num_steps}"
        
        return images, info
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# Create Gradio interface
def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Flyer Generator - SDXL", theme=Soft()) as demo:
        gr.Markdown(
            """
            # 🎨 Flyer Generator with SDXL
            Generate professional flyers using Stable Diffusion XL.
            Powered by Gradio Zero GPU for fast, serverless inference.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe your flyer: 'A professional business flyer for a coffee shop...'",
                    lines=3
                )
                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="blurry, bad quality, distorted, ugly, text errors, watermark",
                    lines=2
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
                
                generate_btn = gr.Button("🚀 Generate Flyer", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Generated Flyer",
                    type="pil"
                )
                output_info = gr.Textbox(
                    label="Generation Info",
                    lines=3
                )
        
        # Examples
        gr.Examples(
            examples=[
                [
                    "Professional coffee shop flyer, modern design, warm colors, coffee beans, minimalist, clean layout",
                    "blurry, bad quality, distorted, ugly",
                    50,
                    7.5,
                    1024,
                    1024,
                    42,
                    True
                ],
                [
                    "Tech conference flyer, futuristic design, blue and purple gradient, circuit patterns, 'TECH 2025' text, professional",
                    "blurry, bad quality, text errors",
                    50,
                    7.5,
                    1024,
                    1024,
                    123,
                    True
                ],
                [
                    "Summer music festival poster, vibrant colors, tropical theme, palm trees, sunset, energetic design",
                    "blurry, bad quality, distorted",
                    50,
                    7.5,
                    1024,
                    1024,
                    456,
                    True
                ]
            ],
            inputs=[prompt, negative_prompt, num_steps, guidance_scale, width, height, seed, use_refiner],
            outputs=[output_image, output_info],
            fn=generate_flyer,
            cache_examples=False
        )
        
        # Connect the generate button
        generate_btn.click(
            fn=generate_flyer,
            inputs=[prompt, negative_prompt, num_steps, guidance_scale, width, height, seed, use_refiner],
            outputs=[output_image, output_info]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
