from fastapi import FastAPI, HTTPException, UploadFile, File
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
import httpx
from dotenv import load_dotenv

load_dotenv()
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

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_BASE = "https://api.deepinfra.com/v1/inference"

class GenRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 50
    width: int = 1024
    height: int = 1024
    guidance_scale: float = 7.5

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend is running"}
    
@app.get("/health")
async def health():
    """Check if DeepInfra API is accessible."""
    if not DEEPINFRA_API_KEY:
        return {"status": "error", "message": "DEEPINFRA_API_KEY not set"}
    return {"status": "ok", "provider": "deepinfra", "api_key_set": bool(DEEPINFRA_API_KEY)}

@app.post("/generate")
async def generate(req: GenRequest):
    """Generate image using DeepInfra's SDXL API."""
    if not DEEPINFRA_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="DEEPINFRA_API_KEY not set. Add it to your .env file."
        )
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{DEEPINFRA_BASE}/stabilityai/sdxl",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
                },
                json={
                    "prompt": req.prompt,
                    "num_inference_steps": req.num_inference_steps,
                    "width": req.width,
                    "height": req.height,
                    "guidance_scale": req.guidance_scale,
                    "negative_prompt": "blurry, low quality, distorted, text, watermark",
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Return base64 image
            return {
                "image": data["images"][0],
                "seed": data.get("seed"),
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"DeepInfra API error: {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"DeepInfra API error: {e.response.text}"
            )
        except Exception as e:
            logger.exception("Error during generation")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Handle file upload (optional - for future features)."""
    contents = await file.read()
    return {"filename": file.filename, "size": len(contents)}

@app.get("/list")
def list_images():
    """Return list of stored images (optional - for future features)."""
    return {"images": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
