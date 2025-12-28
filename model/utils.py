"""
Utility functions for the Flyer Generator
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Union


def save_image(image: Union[Image.Image, np.ndarray], output_path: Union[str, Path], create_dirs: bool = True) -> str:
    """
    Save an image to disk. Accepts PIL Images or NumPy arrays.
    
    Args:
        image: PIL Image or NumPy array to save
        output_path: Path where to save the image
        create_dirs: Whether to create parent directories if they don't exist
    
    Returns:
        Absolute path to the saved image
    """
    output_path = Path(output_path)
    
    # Convert NumPy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    if create_dirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    
    return str(output_path.absolute())


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load an image from disk.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        PIL Image object
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    return Image.open(image_path)


def create_output_dir(base_dir: str = "outputs") -> Path:
    """
    Create an output directory for generated images.
    
    Args:
        base_dir: Base directory name
    
    Returns:
        Path to the created directory
    """
    output_dir = Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_image_size(image: Image.Image) -> tuple:
    """
    Get the width and height of an image.
    
    Args:
        image: PIL Image
    
    Returns:
        Tuple of (width, height)
    """
    return image.size


def resize_image(
    image: Image.Image,
    width: Optional[int] = None,
    height: Optional[int] = None,
    maintain_aspect: bool = True
) -> Image.Image:
    """
    Resize an image to specified dimensions.
    
    Args:
        image: PIL Image to resize
        width: Target width (if None, calculated from height)
        height: Target height (if None, calculated from width)
        maintain_aspect: Whether to maintain aspect ratio
    
    Returns:
        Resized PIL Image
    """
    if width is None and height is None:
        return image
    
    original_width, original_height = image.size
    
    if maintain_aspect:
        aspect_ratio = original_width / original_height
        
        if width is None and height is not None:
            width = int(height * aspect_ratio)
        elif height is None and width is not None:
            height = int(width / aspect_ratio)
        else:
            width = original_width
            height = original_height
    else:
        width = width or original_width
        height = height or original_height
    
    # Ensure dimensions are multiples of 8 for SDXL
    width = (int(width) // 8) * 8
    height = (int(height) // 8) * 8
    
    return image.resize((width, height), Image.Resampling.LANCZOS)
