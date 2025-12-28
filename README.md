# AI Flyer Generator

## Capstone Project - Associates of Science in AI

**Live Demo**: [https://prescottcassy.github.io/Flyer_Generator/](https://prescottcassy.github.io/Flyer_Generator/)

---

## Project Overview

The **AI Flyer Generator** is a full-stack web application that leverages generative AI to create custom promotional flyers from natural language prompts. This capstone project demonstrates proficiency in machine learning, API design, cloud deployment, and modern web development.

### Key Capabilities

- **Text-to-Image Generation**: Generate professional flyers using Stable Diffusion XL (SDXL)
- **Interactive Web UI**: Intuitive React interface with real-time generation feedback
- **REST API Backend**: Scalable FastAPI server with lazy-loaded ML pipeline
- **Cloud Deployment**: Containerized deployment on Railway with GitHub Pages frontend

---

## Technical Architecture

### System Design

- **Frontend**: React 19 + Vite on GitHub Pages
- **Backend**: FastAPI + Uvicorn on Railway (Docker container)
- **ML Model**: Stable Diffusion XL (SDXL) from Hugging Face Diffusers
- **Communication**: REST API with JSON payloads

### Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | React 19 + Vite | Fast, modern SPA with lazy-loaded routes |
| **Backend** | FastAPI + Uvicorn | High-performance async REST API |
| **ML Model** | Stable Diffusion XL | State-of-the-art text-to-image generation |
| **ML Framework** | PyTorch + Diffusers | Efficient model loading and inference |
| **Deployment** | Docker + Railway | Containerized backend with auto-scaling |
| **Hosting** | GitHub Pages | Static frontend deployment |

---

## Features & Implementation

### 1. **Generative AI Pipeline**

- **Model**: Stable Diffusion XL (1.0) from Hugging Face
- **Inference**: Text prompts → Latent diffusion → PNG images
- **Optimization**: Model CPU offloading, VAE slicing, VAE tiling
- **Refiner Support**: Optional SDXL refiner for enhanced quality
- **Gradio Integration**: Web UI with Gradio Zero GPU support
- **Hardware**: CPU-optimized inference with fallback error handling

### 2. **REST API Endpoints**

#### `GET /health`

Returns server and pipeline status.

#### `POST /generate`

Generate an image from a text prompt.

**Request**:

```json
{
  "prompt": "A vibrant flyer for a summer music festival",
  "num_inference_steps": 50
}
```

### 3. **Frontend Features**

- **Responsive Design**: Mobile, tablet, and desktop layouts
- **Error Handling**: User-friendly error messages with retry options
- **Loading States**: Visual feedback during image generation (8-15s typical)
- **Image Export**: Download generated images as PNG files

---

## Development & Deployment

### Local Development

1. **Clone & Setup Backend**:

   ```bash
   git clone https://github.com/prescottcassy/Flyer_Generator.git
   cd Flyer_Generator
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python -m uvicorn backend.server:app --reload --port 8000
   ```

2. **Setup Frontend**:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

### Production Deployment

- **Backend**: Railway (auto-deploys on git push)
- **Frontend**: GitHub Pages (auto-deploys on push to main)

---

## Learning Outcomes

### Technical Skills Demonstrated

| Skill | Evidence |
|-------|----------|
| **Machine Learning** | Stable Diffusion XL inference; CPU optimization; memory constraints |
| **Backend API Design** | Stateless REST API; error handling; lazy loading; CORS |
| **Frontend Development** | React hooks; responsive CSS; state management; API integration |
| **Cloud Deployment** | Docker; Railway; GitHub Pages; CI/CD automation |
| **Full-Stack Integration** | End-to-end data flow; async communication; error propagation |
| **Software Engineering** | Git; modular code; documentation; debugging |

### Performance Metrics

- **Image Generation**: 8-15 seconds (CPU)
- **API Response**: <100ms (excluding inference)
- **Frontend Size**: ~230KB (gzipped: 74KB)
- **Deploy Time**: <5 minutes (Railway auto-redeploy)

---

## Project Structure

```bash
Flyer_Generator/
├── backend/server.py           # FastAPI application
├── model/                      # ML utilities and training
├── frontend/src/               # React components & styling
├── requirements.txt            # Python dependencies
├── Procfile                    # Railway start command
├── Dockerfile                  # Container image
└── README.md
```

---

## Security & Best Practices

- **Secrets Management**: Environment variables (never in code)
- **CORS**: Frontend-only origin access
- **Error Handling**: Graceful degradation
- **Lazy Loading**: Optimize startup time
- **Lazy Model Loading**: Pipeline loads on first request

---

## Capstone Significance

This project demonstrates:

1. **AI/ML Integration**: Production-grade Stable Diffusion inference
2. **Full-Stack Development**: Backend → Frontend → Deployment
3. **Cloud Architecture**: Containerized, auto-scaling infrastructure
4. **Production Readiness**: Error handling, monitoring, optimization
5. **Real-World Problem**: Practical utility with scalable solution

---

## Author

**Cassy Cormier**  
Associates of Science in Artificial Intelligence from Houston City College
[GitHub](https://github.com/prescottcassy) | [Live Demo](https://prescottcassy.github.io/Flyer_Generator/)

---

**Status**: ✅ Live on GitHub Pages & Railway  
**Last Updated**: December 2025</content>
