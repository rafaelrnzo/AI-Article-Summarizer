# Gunakan base image Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies system (biar crawl4ai & bs4 lancar)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    chromium-driver \
    chromium \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files termasuk .env
COPY . .

# Expose port
EXPOSE 8000

# Jalankan app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]
