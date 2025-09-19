
# -------------------------
# 1. Use official lightweight Python image
# -------------------------
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# -------------------------
# 2. Install system dependencies
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# 3. Install Python dependencies
# -------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# 4. Copy application code
# -------------------------
COPY . .

# -------------------------
# 5. Expose port for FastAPI
# -------------------------
EXPOSE 8000

# -------------------------
# 6. Run FastAPI with Uvicorn
# -------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
