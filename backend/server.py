import io
import base64
import logging
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# FastAPI setup
# -------------------------------------------------------------------
app = FastAPI(title="Flyer Generator API")

# CORS configuration - MUST be added before other middleware
# Allow requests from GitHub Pages and localhost
cors_origins = [
    "https://prescottcassy.github.io",
    "https://prescottcassy.github.io/Flyer_Generator",
    "https://prescottcassy.github.io/Flyer_Generator/",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost",
    "*",  # Allow all origins as fallback (less secure but ensures CORS works)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# -------------------------------------------------------------------
# Request schema
# -------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 50

# -------------------------------------------------------------------
# Global state
# -------------------------------------------------------------------
base = None
refiner = None
pipeline_loaded = False

# -------------------------------------------------------------------
# Lazy load functions
# -------------------------------------------------------------------
def lazy_load_pipeline():
    """Lazy load pipeline on first use."""
    global base, refiner, pipeline_loaded
    if pipeline_loaded:
        return base, refiner
    
    try:
        logger.info("Loading SDXL pipeline...")
        from model.training_pipeline import load_pipeline
        base, refiner = load_pipeline()
        pipeline_loaded = True
        logger.info("✓ SDXL pipeline loaded successfully")
        return base, refiner
    except ImportError as e:
        logger.error(f"Import error loading pipeline: {str(e)}")
        logger.warning("Using fallback: pipeline functions not available")
        return None, None
    except Exception as e:
        logger.error(f"Error loading pipeline: {str(e)}")
        return None, None

# -------------------------------------------------------------------
# Routes - Health check
# -------------------------------------------------------------------
@app.get("/")
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Flyer Generator API",
        "pipeline_loaded": pipeline_loaded
    }

# -------------------------------------------------------------------
# Routes - Generate endpoint
# -------------------------------------------------------------------
@app.post("/generate")
async def generate(req: GenerateRequest):
    """Generate an image from a text prompt."""
    try:
        # Lazy load pipeline
        base_pipe, refiner_pipe = lazy_load_pipeline()
        
        if base_pipe is None:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Pipeline not available. Please check server logs.",
                    "error": "Pipeline initialization failed"
                }
            )
        
        logger.info(f"Generating image for prompt: {req.prompt[:50]}...")
        from model.training_pipeline import generate_image
        
        img = generate_image(base_pipe, req.prompt, req.num_inference_steps)
        
        # Convert image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        logger.info("✓ Image generated successfully")
        return {
            "image": img_base64,
            "prompt": req.prompt,
            "num_inference_steps": req.num_inference_steps
        }
    except Exception as e:
        logger.exception(f"Error during generation: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Generation failed: {str(e)}"}
        )

# -------------------------------------------------------------------
# Error handlers
# -------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# -------------------------------------------------------------------
# Startup/Shutdown
# -------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("=" * 60)
    logger.info("Starting Flyer Generator API")
    logger.info("=" * 60)
    # Attempt to load pipeline on startup
    lazy_load_pipeline()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Flyer Generator API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
