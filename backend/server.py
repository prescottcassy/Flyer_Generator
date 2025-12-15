"""
FastAPI Backend for Flyer Generator
Now uses Modal for GPU inference - lightweight proxy server
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import modal
import base64
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Flyer Generator API",
    description="AI-powered flyer generation using SDXL via Modal GPU",
    version="2.0.0"
)

# CORS configuration - allow all origins for now (can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 50
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A vibrant summer music festival flyer with bold text and colorful background",
                "num_inference_steps": 50
            }
        }

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Flyer Generator API",
        "version": "2.0.0",
        "gpu_backend": "Modal (Serverless)",
        "endpoints": {
            "health": "/health",
            "generate": "/generate"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Test Modal connection
        f = modal.Function.lookup("flyer-generator", "generate_flyer")
        
        return {
            "status": "healthy",
            "backend": "Railway",
            "gpu_provider": "Modal",
            "model": "Stable Diffusion XL",
            "modal_connected": True
        }
    except Exception as e:
        logger.error(f"Modal connection failed: {e}")
        return {
            "status": "degraded",
            "backend": "Railway",
            "gpu_provider": "Modal",
            "modal_connected": False,
            "error": str(e)
        }

@app.post("/generate")
async def generate_image(request: GenerateRequest):
    """
    Generate a flyer image from a text prompt
    
    Args:
        request: GenerateRequest with prompt and optional num_inference_steps
    
    Returns:
        PNG image bytes
    """
    logger.info(f"Received generation request: {request.prompt[:50]}...")
    
    try:
        # Validate inputs
        if not request.prompt or len(request.prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        if request.num_inference_steps < 1 or request.num_inference_steps > 150:
            raise HTTPException(
                status_code=400, 
                detail="num_inference_steps must be between 1 and 150"
            )
        
        # Lookup Modal function
        logger.info("Looking up Modal GPU function...")
        f = modal.Function.lookup("flyer-generator", "generate_flyer")
        
        # Call Modal GPU function (this is where the magic happens!)
        logger.info("Calling Modal GPU for image generation...")
        img_base64 = f.remote(request.prompt, request.num_inference_steps)
        
        # Decode base64 to bytes
        logger.info("Decoding image and sending response...")
        img_bytes = base64.b64decode(img_base64)
        
        return Response(
            content=img_bytes,
            media_type="image/png",
            headers={
                "Content-Disposition": "inline; filename=generated_flyer.png"
            }
        )
        
    except modal.exception.NotFoundError:
        logger.error("Modal function not found - did you deploy modal_gpu.py?")
        raise HTTPException(
            status_code=503,
            detail="GPU service not available. Please deploy Modal function first."
        )
    
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {str(e)}"
        )

@app.get("/info")
def get_info():
    """Get information about the service"""
    return {
        "model": "Stable Diffusion XL",
        "version": "1.0",
        "gpu": "NVIDIA A10G (via Modal)",
        "max_inference_steps": 150,
        "default_inference_steps": 50,
        "image_size": "1024x1024",
        "average_generation_time": "2-4 seconds"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)