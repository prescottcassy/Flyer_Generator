import os
import subprocess
import json
import numpy as np
import clip
import torch
import pytesseract
from pytesseract import TesseractNotFoundError
import shutil
from PIL import Image
from collections import Counter
from torchvision import transforms
from datetime import datetime
from typing import cast
import sys


# CLIP model-based tagging
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
def _configure_tesseract():
    """Configure pytesseract to point to a tesseract binary and tessdata directory.

    Order: TESSERACT_CMD env var -> common install paths -> conda/python prefix -> PATH.
    This avoids hardcoding machine-specific paths in the file.
    """
    # 1) explicit env var override
    cmd = os.getenv("TESSERACT_CMD")
    if cmd and os.path.exists(cmd):
        pytesseract.pytesseract.tesseract_cmd = cmd
    else:
        # 2) common Windows install locations
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        # 3) check current python/conda prefix
        try:
            prefix = os.path.dirname(sys.executable)
            candidates.append(os.path.join(prefix, "Library", "bin", "tesseract.exe"))
            candidates.append(os.path.join(prefix, "bin", "tesseract.exe"))
        except Exception:
            pass

        for c in candidates:
            if c and os.path.exists(c):
                pytesseract.pytesseract.tesseract_cmd = c
                break
        else:
            # 4) fallback to PATH
            path_exe = shutil.which("tesseract")
            if path_exe:
                pytesseract.pytesseract.tesseract_cmd = path_exe

    # Prepare tessdata candidates (we'll search for eng.traineddata when calling OCR)
    global tessdata_candidates
    tessdata_candidates = [
        os.path.join(os.path.dirname(sys.executable), 'Library', 'share', 'tessdata'),
        os.path.join(os.path.dirname(sys.executable), 'share', 'tessdata'),
        r'C:\Program Files\Tesseract-OCR\tessdata'
    ]


# configure on import
_configure_tesseract()

# If we located a tessdata folder during configuration, set TESSDATA_PREFIX
# for the current process so tesseract can find language data. Otherwise
# warn users to install/configure Tesseract.
_found_tess = None
try:
    for td in tessdata_candidates:
        if td and os.path.isdir(td) and os.path.exists(os.path.join(td, 'eng.traineddata')):
            _found_tess = td
            break
except Exception:
    _found_tess = None

if _found_tess:
    os.environ.setdefault("TESSDATA_PREFIX", _found_tess)
else:
    t_cmd = getattr(pytesseract.pytesseract, 'tesseract_cmd', None)
    if not ((t_cmd and os.path.exists(t_cmd)) or shutil.which("tesseract")):
        print("Warning: Tesseract executable or tessdata not found.\n"
              "Install Tesseract-OCR (for Windows try the UB Mannheim build) and ensure\n"
              "the folder containing 'eng.traineddata' is available.\n"
              "You can also set the environment variables TESSDATA_PREFIX or TESSERACT_CMD\n"
              "to point to the tessdata folder or tesseract.exe respectively.")


def _find_tessdata_dir():
    """Return a tessdata directory containing eng.traineddata if found, else None."""
    # check the candidates first
    for td in tessdata_candidates:
        if os.path.isdir(td) and os.path.exists(os.path.join(td, 'eng.traineddata')):
            return td
    # fallback: search the conda/python prefix for eng.traineddata (limited walk)
    try:
        prefix = os.path.dirname(sys.executable)
        for root, dirs, files in os.walk(prefix):
            if 'eng.traineddata' in files:
                return root
    except Exception:
        pass
    return None
# Define tags

candidate_tags = [
    "holiday", "office closed", "community event", "real estate", "sales offer",
    "gathering", "promotion", "education flyer", "event",
    "activity", "food and drink", "celebration", "national holiday"
]

def generate_clip_tags(image_path, top_k=3):
    pil = Image.open(image_path)
    tensor = cast(torch.Tensor, preprocess(pil))
    image = tensor.unsqueeze(0).to(device)
    text = clip.tokenize(candidate_tags).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image = image_features @ text_features.T
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    top_indices = probs.argsort()[-top_k:][::-1]
    return [candidate_tags[i] for i in top_indices]

# Extracts metadata
def extract_tags(image_path):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    file_size = os.path.getsize(image_path)
    dominant_color = get_dominant_color(img)
    try:
        tessdata_dir = _find_tessdata_dir()
        if tessdata_dir:
            # Don't pass the tessdata path via the 'config' string because pytesseract
            # will split it on spaces and break Windows paths like "C:\Program Files".
            # Instead, set the TESSDATA_PREFIX env var for the current process so
            # tesseract finds the language data directory reliably.
            os.environ["TESSDATA_PREFIX"] = tessdata_dir
            text = pytesseract.image_to_string(img)
        else:
            text = pytesseract.image_to_string(img)
    except (TesseractNotFoundError, pytesseract.pytesseract.TesseractError) as e:
        # Print the error for debugging but continue; extracted_text will be empty.
        print("Error occurred while performing OCR:", e)
        text = ""

    return {
        "filename": os.path.basename(image_path),
        "dimensions": f"{width}x{height}",
        "file_size_bytes": file_size,
        "semantic_tags": generate_clip_tags(image_path),
        "dominant_color": dominant_color,
        "extracted_text": text.strip(),
        "timestamp": datetime.now().isoformat()
    }

# Extract dominant color and saves as a JSON file
def get_dominant_color(img):
    img = img.resize((50, 50))
    pixels = np.array(img).reshape(-1, 3)
    most_common = Counter(map(tuple, pixels)).most_common(1)[0][0]
    return f"rgb{most_common}"


def tag_folder(folder_path, output_path="metadata.json"):
    metadata = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            tags = extract_tags(image_path)
            metadata.append(tags)

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {output_path}")


def trigger_node_upload(local_metadata_path):
    # Resolve the project-level scripts path reliably relative to this file.
    # Using __file__ ensures we find the script regardless of the current working directory.
    node_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'uploadMetadata.js'))
    # Only attempt to run node if it's available on PATH
    node_path = shutil.which("node")
    if not node_path:
        print("Node.js not found on PATH — skipping metadata upload. Install Node.js or add it to PATH to enable upload.")
        return
    if not os.path.exists(node_script):
        print(f"Node upload script not found at {node_script} — skipping metadata upload.")
        return
    try:
        subprocess.run([node_path, node_script, local_metadata_path], check=True)
        print("Metadata upload triggered via Node.js")
    except subprocess.CalledProcessError as e:
        print("Upload failed:", e)


if __name__ == "__main__":
    tag_folder("c:/Users/presc/OneDrive/Desktop/AI Resources and References/Datasets/Flyers", output_path="metadata.json")
    trigger_node_upload("metadata.json")  # Ensure this function is defined elsewhere