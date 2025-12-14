FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose port (Railway will set $PORT)
EXPOSE 8000

# Run the app
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
