FROM python:3.11-slim

# Pillow/opencv deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api

# Render erwartet PORT
ENV PORT=8000
CMD ["uvicorn", "api.scan:app", "--host", "0.0.0.0", "--port", "8000"]
