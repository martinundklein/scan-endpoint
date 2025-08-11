FROM python:3.11-slim

# System deps for OpenCV & reportlab
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    gcc \ 
    libgl1 \ 
    libglib2.0-0 \ 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api

ENV PORT=8000
CMD ["uvicorn", "api.scan:app", "--host", "0.0.0.0", "--port", "8000"]
