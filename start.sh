#!/bin/bash
# Railway startup script for Flyer Generator backend

echo "Installing dependencies..."
pip install -r requirements.txt
pip install -r backend/requirements.txt

echo "Starting Uvicorn server..."
uvicorn backend.server:app --host 0.0.0.0 --port 8000
