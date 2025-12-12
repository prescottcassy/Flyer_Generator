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
    """Attempt to import the training pipeline on startup. 
    """
    # Allow users to set SERVICEACCOUNT by pointing to a local JSON file path
    # (SERVICEACCOUNT_PATH). This keeps private keys off the repo while letting
    # the server load the secret into the environment for libraries that
    # expect the JSON as a single env var.
    import os
    sa_path = os.getenv("SERVICEACCOUNT_PATH")
    if sa_path and not os.getenv("SERVICEACCOUNT"):
        try:
            with open(sa_path, "r", encoding="utf-8") as fh:
                os.environ["SERVICEACCOUNT"] = fh.read()
            logger.info("Loaded SERVICEACCOUNT from SERVICEACCOUNT_PATH: %s", sa_path)
        except Exception as e:
            logger.exception("Failed to load SERVICEACCOUNT from path %s: %s", sa_path, e)

    # If SERVICEACCOUNT is still not set, try to load from SERVICEACCOUNT.env file
    if not os.getenv("SERVICEACCOUNT"):
        try:
            with open("SERVICEACCOUNT.env", "r", encoding="utf-8") as fh:
                os.environ["SERVICEACCOUNT"] = fh.read()
            logger.info("Loaded SERVICEACCOUNT from SERVICEACCOUNT.env")
        except FileNotFoundError:
            logger.info("SERVICEACCOUNT.env not found; skipping pipeline load")
        except Exception as e:
            logger.exception("Failed to load SERVICEACCOUNT from SERVICEACCOUNT.env: %s", e)

    # Only attempt to import the training pipeline at startup if SERVICEACCOUNT
    # is present. This avoids startup failures when developers run the server
    # locally for frontend work without secrets or heavy ML packages installed.
    if not os.getenv("SERVICEACCOUNT"):
        logger.info("SERVICEACCOUNT not set; skipping training pipeline import on startup.")
        return

    try:
        import importlib

        training_pipeline = importlib.import_module("model.training_pipeline")
        training_pipeline.load_pipeline()
        if not hasattr(training_pipeline, "base") or training_pipeline.base is None:
            logger.warning("`training_pipeline` imported but `base` pipeline not loaded.")
        else:
            logger.info("Model pipeline loaded successfully.")
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

        # Try to load SERVICEACCOUNT from SERVICEACCOUNT_PATH at request time
        # This helps when the server was started without secrets but the
        # developer later sets a local path env var before calling /generate.
        sa_path = os.getenv("SERVICEACCOUNT_PATH")
        if sa_path and not os.getenv("SERVICEACCOUNT"):
            try:
                with open(sa_path, "r", encoding="utf-8") as fh:
                    os.environ["SERVICEACCOUNT"] = fh.read()
                logger.info("Loaded SERVICEACCOUNT from SERVICEACCOUNT_PATH during /generate: %s", sa_path)
            except Exception:
                logger.exception("Failed to load SERVICEACCOUNT from SERVICEACCOUNT_PATH during /generate: %s", sa_path)

        if not os.getenv("SERVICEACCOUNT"):
            raise HTTPException(status_code=500, detail="SERVICEACCOUNT not set; cannot generate images without secrets.")
        else:
            import importlib

            training_pipeline = importlib.import_module("model.training_pipeline")
            if training_pipeline.base is None:
                training_pipeline.load_pipeline()
            if training_pipeline.base is None:
                raise RuntimeError("Model pipeline not initialized or unavailable.")
            else:
                # `generate_image` expects (pipeline, prompt, num_inference_steps, ...)
                img = generate_image(training_pipeline.base, req.prompt, req.num_inference_steps)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
        return {"image": b64}
    except Exception as e:
        # Log full exception to server console
        logger.exception("Error during generation: %s", e)

        # If DEV_DEBUG is set, include full traceback in the HTTP response
        if os.getenv("DEV_DEBUG", "0").lower() in ("1", "true", "yes"):
            tb = traceback.format_exc()
            raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb})

        # Otherwise return a short error message
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
