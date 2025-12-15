#!/bin/bash
# Railway startup script for Flyer Generator backend

echo "Installing dependencies..."
pip install -r requirements.txt
pip install -r backend/requirements.txt

echo "Starting Uvicorn server on port ${PORT:-8000}..."
uvicorn backend.server:app --host 0.0.0.0 --port ${PORT:-8000}
