# Flyer Metadata Tagging Pipeline

This module extracts metadata from flyer images, generates semantic tags using OpenAI's CLIP model, and triggers a Node.js script to upload the metadata to Firebase Storage.

---

## Folder Purpose

The `data-prep/` folder contains Python scripts for preprocessing flyer images, including:

- Extracting image metadata (dimensions, file size, dominant color, OCR text)
- Generating semantic tags using CLIP
- Saving metadata to JSON
- Triggering a Node.js uploader to send metadata to Firebase Storage

---

## 🛠 Setup Instructions

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> Note: `pytesseract` requires [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system.

### 2. Install CLIP model

CLIP is installed via GitHub:

```bash
pip install git+https://github.com/openai/CLIP.git
```

---

## File Overview

| File | Purpose |
|------|---------|
| `tag_images.py` | Main script for tagging images and triggering upload |
| `requirements.txt` | Python dependencies |
| `README.md` | Documentation for this pipeline |

---

## How It Works

### 1. Metadata Extraction

Each image is processed to extract:

- Filename
- Dimensions
- File size
- Dominant color
- OCR text (via Tesseract)
- Timestamp

### 2. Semantic Tagging with CLIP

Images are passed through CLIP to infer high-level tags from a predefined list:

```python
candidate_tags = [
  "holiday sale", "community event", "real estate flyer", "discount offer",
  "family gathering", "healthcare promotion", "education flyer", "music event",
  "sports activity", "food and drink", "back to school", "grand opening"
]
```

Top 3 tags are selected based on CLIP similarity scores.

### 3. Metadata Output

All metadata is saved to a single JSON file:

```json
[
  {
    "filename": "flyer1.jpg",
    "dimensions": "1080x720",
    "semantic_tags": ["holiday sale", "discount offer", "food and drink"],
    ...
  },
  ...
]
```

### 4. Firebase Upload Trigger

After metadata is saved, the script calls a Node.js uploader:

```python
subprocess.run(["node", "../scripts/uploadMetadata.js", "metadata.json"])
```

This keeps Firebase secrets and SDK logic centralized in Node.

---

## Running the Pipeline

Update the image folder path in `tag_images.py`:

```python
tag_folder("path/to/flyer_images", output_path="metadata.json")
```

Then run:

```bash
python tag_images.py
```

---

## Security Notes

- Firebase Admin SDK is initialized in `scripts/firebase-init.js`
- Python does not handle secrets directly
- Metadata upload is delegated to Node.js for compliance and modularity

---

## Maintainer

Cassy Cormier  
AI Developer  
Houston City College  · BrandBeast  

---
