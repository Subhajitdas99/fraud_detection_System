# 1. Use official lightweight Python image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application code
COPY ./fraud_detection_System /app

# 6. Expose port for FastAPI
EXPOSE 8000

# 7. Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

