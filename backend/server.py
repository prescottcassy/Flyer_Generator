import io
import base64
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Import your own pipeline loader and generator
from model.training_pipeline import load_pipeline, generate_image, get_generator_args

# -------------------------------------------------------------------
# FastAPI setup
# -------------------------------------------------------------------
app = FastAPI()

# Allow CORS for your frontend (GitHub Pages domain)
# Fix CORS to allow proper origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://prescottcassy.github.io",
        "https://prescottcassy.github.io/Flyer_Generator/",
        "http://localhost:3000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
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

# -------------------------------------------------------------------
# Startup event
# -------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Load your custom SDXL pipeline on startup."""
    global base, refiner
    try:
        base, refiner = load_pipeline()
        logging.info("Custom SDXL pipeline initialized.")
    except Exception as e:
        logging.error(f"Failed to load pipeline: {str(e)}")

# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_loaded": base is not None}

# -------------------------------------------------------------------
# Generate route
# -------------------------------------------------------------------
@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        if base is None:
            return {"detail": "Pipeline not loaded. Please check server logs."}
        
        img = generate_image(base, req.prompt, req.num_inference_steps)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
        return {"image": b64}
    except Exception as e:
        logging.exception("Error during generation")
        return {"detail": str(e)}
