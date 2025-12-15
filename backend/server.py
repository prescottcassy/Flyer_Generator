import logging
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

# CORS configuration - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
# Routes
# -------------------------------------------------------------------

@app.get("/")
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Flyer Generator API"
    }

@app.post("/generate")
async def generate(req: GenerateRequest):
    """Generate an image from a text prompt."""
    try:
        logger.info(f"Generate request: {req.prompt[:50]}...")
        return {
            "image": "",
            "prompt": req.prompt,
            "num_inference_steps": req.num_inference_steps,
            "status": "placeholder",
            "message": "Image generation will be implemented here"
        }
    except Exception as e:
        logger.exception(f"Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error: {str(e)}"}
        )

# -------------------------------------------------------------------
# Startup/Shutdown
# -------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("=" * 60)
    logger.info("Flyer Generator API starting...")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Flyer Generator API shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
