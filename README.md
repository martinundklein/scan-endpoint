# Scan Endpoint (Docker)
Simple FastAPI service that deskews/crops a receipt photo and returns a B/W PDF.

## How to run locally
```
docker build -t scan-endpoint .
docker run -p 8000:8000 scan-endpoint
```
POST http://localhost:8000/scan with JSON:
{"file_url":"https://example.com/your-image.jpg"}
