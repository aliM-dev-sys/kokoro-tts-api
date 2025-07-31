FROM python:3.10-slim

WORKDIR /app

# Install only necessary system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends espeak-ng ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only necessary files first to leverage Docker cache
COPY main.py .

EXPOSE 3000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]
