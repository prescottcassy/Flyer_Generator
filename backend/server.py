from fastapi import FastAPI, HTTPException, UploadFile, File
from starlette.middleware.cors import CORSMiddleware
from model.utils import generate_image
from pydantic import BaseModel
import logging
import os
import traceback

logger = logging.getLogger(__name__)

app = FastAPI()

# Development-friendly CORS: allow all origins to simplify local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 50


@app.on_event("startup")
def load_pipeline():
    """Attempt to import the training pipeline on startup using DeepInfra API key."""
    api_key = os.getenv("DEEPINFRA_API_KEY")
    if not api_key:
        logger.info("DEEPINFRA_API_KEY not set; skipping training pipeline import on startup.")
        return

    try:
        import importlib
        training_pipeline = importlib.import_module("model.training_pipeline")
        training_pipeline.load_pipeline()
        if not hasattr(training_pipeline, "base") or training_pipeline.base is None:
            logger.warning("`training_pipeline` imported but `base` pipeline not loaded.")
        else:
            logger.info("Model pipeline loaded successfully with DeepInfra API key.")
    except Exception as e:
        logger.exception("Error loading training_pipeline on startup: %s", e)

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend is running"}
    
@app.get("/health")
def health():
    """Return basic status and whether the model pipeline appears ready."""
    try:
        import importlib
        training_pipeline = importlib.import_module("model.training_pipeline")
        pipeline = getattr(training_pipeline, "base", None)
        ready = pipeline is not None
    except Exception:
        ready = False
    return {"status": "ok", "pipeline_ready": ready}


@app.post("/generate")
def generate(req: GenRequest):
    try:
        import io, base64

        api_key = os.getenv("DEEPINFRA_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="DEEPINFRA_API_KEY not set; cannot generate images.")

        import importlib
        training_pipeline = importlib.import_module("model.training_pipeline")
        if training_pipeline.base is None:
            training_pipeline.load_pipeline()
        if training_pipeline.base is None:
            raise RuntimeError("Model pipeline not initialized or unavailable.")
        else:
            img = generate_image(training_pipeline.base, req.prompt, req.num_inference_steps)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
        return {"image": b64}
    except Exception as e:
        logger.exception("Error during generation: %s", e)
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb})

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # handle file upload
    contents = await file.read()
    return {"filename": file.filename, "size": len(contents)}

@app.get("/list")
def list_images():
    # return list of stored images
    return {"images": [...]}
