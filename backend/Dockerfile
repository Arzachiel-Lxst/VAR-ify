FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/uploads data/results

# Expose port (Railway uses dynamic PORT)
EXPOSE 8000

# Environment variable for port
ENV PORT=8000

# Run the API server with dynamic port
CMD uvicorn api_server:app --host 0.0.0.0 --port $PORT
