"""
SDXL Model for Flyer Generation
Uses Stable Diffusion XL from Hugging Face diffusers library
"""
import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from typing import Optional, List, Union
import os


class SDXLModel:
    """
    Stable Diffusion XL model wrapper for flyer generation.
    
    This class provides a simple interface to the SDXL pipeline from diffusers.
    It supports both base and refiner models for high-quality image generation.
    """
    
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        refiner_id: Optional[str] = "stabilityai/stable-diffusion-xl-refiner-1.0",
        use_refiner: bool = True,
        device: Optional[str] = None,
        torch_dtype = torch.float16
    ):
        """
        Initialize the SDXL model.
        
        Args:
            model_id: HuggingFace model ID for the base SDXL model
            refiner_id: HuggingFace model ID for the refiner model (optional)
            use_refiner: Whether to use the refiner for enhanced quality
            device: Device to run the model on (cuda/cpu). Auto-detected if None.
            torch_dtype: Data type for model weights (float16 recommended for GPU)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype if self.device == "cuda" else torch.float32
        self.use_refiner = use_refiner and refiner_id is not None
        
        print(f"Initializing SDXL on {self.device}...")
        
        # Load VAE for better quality
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=self.torch_dtype
        )
        
        # Load base pipeline
        self.base_pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
            variant="fp16" if self.device == "cuda" else None
        )
        self.base_pipeline.to(self.device)
        
        # Enable memory optimizations
        if self.device == "cuda":
            self.base_pipeline.enable_model_cpu_offload()
            self.base_pipeline.enable_vae_slicing()
            self.base_pipeline.enable_vae_tiling()
        
        # Load refiner pipeline if enabled
        self.refiner_pipeline = None
        if self.use_refiner:
            print(f"Loading refiner model...")
            self.refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                refiner_id,
                vae=vae,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
                variant="fp16" if self.device == "cuda" else None
            )
            self.refiner_pipeline.to(self.device)
            
            if self.device == "cuda":
                self.refiner_pipeline.enable_model_cpu_offload()
                self.refiner_pipeline.enable_vae_slicing()
                self.refiner_pipeline.enable_vae_tiling()
        
        print("SDXL model initialized successfully!")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        num_images: int = 1,
        high_noise_frac: float = 0.8
    ):
        """
        Generate images using SDXL.
        
        Args:
            prompt: Text description of the desired image
            negative_prompt: Things to avoid in the generated image
            num_inference_steps: Number of denoising steps (higher = better quality, slower)
            guidance_scale: How closely to follow the prompt (7-12 recommended)
            width: Image width (must be multiple of 8, recommended: 1024)
            height: Image height (must be multiple of 8, recommended: 1024)
            seed: Random seed for reproducibility
            num_images: Number of images to generate
            high_noise_frac: Fraction of steps for base model (rest for refiner)
        
        Returns:
            List of generated PIL images
        """
        # Set random seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate with base model
        if self.use_refiner and self.refiner_pipeline is not None:
            # Use base model for high noise fraction, then refiner
            print(f"Generating with base model (steps: {int(num_inference_steps * high_noise_frac)})...")
            latents = self.base_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
                num_images_per_prompt=num_images,
                output_type="latent",
                denoising_end=high_noise_frac
            )[0]
            
            # Decode latents to image for refiner input
            print(f"Decoding latents to image...")
            image = self.base_pipeline.vae.decode(latents / self.base_pipeline.vae.config.scaling_factor, return_dict=False)[0]
            image = (image / 2 + 0.5).clamp(0, 1)
            
            # Refine the output
            print(f"Refining with refiner model (steps: {int(num_inference_steps * (1 - high_noise_frac))})...")
            output = self.refiner_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image=image,
                generator=generator,
                denoising_start=high_noise_frac
            )
            images = output[0] if isinstance(output, tuple) else output.images
        else:
            # Use only base model
            print(f"Generating with base model (steps: {num_inference_steps})...")
            output = self.base_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
                num_images_per_prompt=num_images
            )
            images = output[0] if isinstance(output, tuple) else output.images
        
        print("Generation complete!")
        return images
    
    def clear_memory(self):
        """Clear GPU memory."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            print("GPU memory cleared")


def load_model(use_refiner: bool = True) -> SDXLModel:
    """
    Convenience function to load the SDXL model with default settings.
    
    Args:
        use_refiner: Whether to load and use the refiner model
    
    Returns:
        Initialized SDXLModel instance
    """
    return SDXLModel(use_refiner=use_refiner)
