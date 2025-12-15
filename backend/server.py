import io
import base64
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Import your own pipeline loader and generator
from model import training_pipeline
from model.training_pipeline import load_pipeline, generate_image

# -------------------------------------------------------------------
# FastAPI setup
# -------------------------------------------------------------------
app = FastAPI()

# Allow CORS for your frontend (GitHub Pages domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://prescottcassy.github.io/Flyer_Generator/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Request schema
# -------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 50

# -------------------------------------------------------------------
# Startup event
# -------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    # Load your custom SDXL pipeline
    load_pipeline()
    logging.info("Custom SDXL pipeline initialized.")

# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# -------------------------------------------------------------------
# Generate route
# -------------------------------------------------------------------
@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        img = generate_image(training_pipeline.base, req.prompt, req.num_inference_steps)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
        return {"image": b64}
    except Exception as e:
        logging.exception("Error during generation")
        return {"detail": str(e)}
