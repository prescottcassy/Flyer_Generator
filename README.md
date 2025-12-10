# Flyer Generator

A web application for generating custom flyers using AI-powered image generation with Stable Diffusion XL.

## Features

- **AI-Powered Generation**: Create unique flyers from text prompts using Stable Diffusion XL
- **Web Interface**: Modern React frontend with responsive design
- **REST API**: FastAPI backend for reliable image generation
- **Real-time Feedback**: Loading states and error handling
- **Download Support**: Save generated images directly

## Tech Stack

- **Backend**: Python 3.13, FastAPI, PyTorch, Diffusers
- **Frontend**: React 19, Vite, React Router
- **AI Model**: Stable Diffusion XL (SDXL) via Hugging Face
- **Deployment**: Local development with Uvicorn

## Prerequisites

- Python 3.13+
- Node.js 18+
- Google Cloud service account (for model access)
- Git

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/prescottcassy/Flyer_Generator.git
   cd Flyer_Generator
   ```

2. **Set up Python environment**:

   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # Windows
   # source .venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

3. **Set up frontend**:

   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Configure secrets**:
   - Create `SERVICEACCOUNT.env` with your Google Cloud service account JSON
   - Or set `SERVICEACCOUNT_PATH` environment variable to the JSON file path

## Usage

1. **Start the backend**:

   ```bash
   python -m uvicorn backend.server:app --port 8000
   ```

2. **Start the frontend** (in a new terminal):

   ```bash
   cd frontend
   npm run dev
   ```

3. **Open your browser** to `http://localhost:5173`

4. **Generate a flyer**:
   - Enter a descriptive prompt (e.g., "A vibrant concert poster for a rock band")
   - Adjust inference steps (higher = better quality, slower)
   - Click "Generate"
   - Download the result when ready

## API Endpoints

### GET /health

Check server status and pipeline readiness.

**Response**:

```json
{
  "status": "ok",
  "pipeline_ready": true
}
```

### POST /generate

Generate a flyer image.

**Request Body**:

```json
{
  "prompt": "A beautiful sunset over mountains",
  "num_inference_steps": 50
}
```

**Response**:

```json
{
  "image": "base64-encoded-png-data"
}
```

## Project Structure

```bash
Flyer_Generator/
├── backend/
│   └── server.py          # FastAPI server
├── frontend/
│   ├── src/
│   │   ├── App.jsx        # Main React app
│   │   ├── api.js         # API client
│   │   ├── pages/         # Route components
│   │   └── App.css        # Styles
│   └── package.json
├── model/
│   ├── training_pipeline.py  # ML pipeline loading
│   ├── utils.py           # Image generation utilities
│   └── sdxl_diffusion.py  # SDXL model wrapper
├── data/                  # Training data
├── scripts/               # Utility scripts
├── requirements.txt       # Python dependencies
└── README.md
```

## Development

### Running Tests

```bash
# Backend tests (if implemented)
pytest

# Frontend tests (if implemented)
cd frontend
npm test
```

### Building for Production

```bash
# Build frontend
cd frontend
npm run build

# Backend is ready for deployment with Uvicorn
```

## Configuration

- **SERVICEACCOUNT**: Google Cloud service account JSON (required for model access)
- **SERVICEACCOUNT_PATH**: Path to service account JSON file
- **DEV_DEBUG**: Set to "1" for detailed error tracebacks
- **VITE_API_BASE_URL**: Frontend API base URL (defaults to localhost:8000)

## Troubleshooting

- **Pipeline not loading**: Ensure SERVICEACCOUNT is properly configured
- **Import errors**: Check Python environment and requirements.txt
- **CORS errors**: Backend allows all origins for development
- **Slow generation**: Reduce inference steps or use GPU if available

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

See LICENSE file for details.

## Acknowledgments

- Stable Diffusion XL by Stability AI
- Hugging Face for model hosting
- FastAPI and React communities</content>
