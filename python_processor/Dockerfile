FROM python:3.11-slim

# Install necessary system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip

# Copy the application code
COPY . .

# Expose the Flask port
EXPOSE 5000

# Health check for container
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl --fail http://localhost:5000/health || exit 1

# Start the API
CMD ["python", "api.py"]
