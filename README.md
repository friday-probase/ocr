# AI OCR Service

Advanced OCR microservice with AI capabilities for document processing.

## Features

- **Multi-Engine OCR**: Tesseract, PaddleOCR, and TrOCR with consensus algorithm
- **AI Document Classification**: Vision transformers for document type detection
- **Smart Consensus**: Weighted voting between engines for maximum accuracy
- **Structured Data Extraction**: Payslip, ID, invoice, and other document types
- **Async Processing**: Job queue integration for batch processing
- **Confidence Scoring**: Automatic human verification queuing for low confidence

## Quick Start

### 1. Build and Run Docker Container

```bash
cd /home/friday/Documents/projects/work/ocr_ai_service
docker build -t ocr-ai-service .
docker run -p 8000:8000 ocr-ai-service
```

### 2. Install Dependencies Locally

```bash
pip install -r requirements.txt
python main.py
```

### 3. Use from Elixir Client

```elixir
# Simple text extraction
{:ok, result} = App.AI.OcrClient.extract_text("/path/to/image.jpg")

# Payslip-specific extraction
{:ok, payslip} = App.AI.EnhancedOcr.extract_payslip_data("payslip.jpg")

# Document processing with classification
{:ok, doc_result} = App.AI.EnhancedOcr.process_document("document.pdf")

# Async processing
{:ok, job} = App.AI.OcrClient.extract_text_async("large_image.png")
```

## API Endpoints

### Classify Document
```
POST /classify
Content-Type: multipart/form-data
file: <image>

Response:
{
  "type": "payslip",
  "confidence": 0.92,
  "description": "Employee salary payment document"
}
```

### Extract Text
```
POST /extract
Content-Type: multipart/form-data
file: <image>
document_types: ["payslip"]
consensus_threshold: 0.85
enable_ai_enhancement: true

Response:
{
  "document_type": "payslip",
  "confidence": 0.89,
  "extracted_text": "EMPLOYEE NAME: JOHN DOE\nBASIC SALARY: $5,000.00...",
  "structured_data": {
    "employee_name": "JOHN DOE",
    "basic_salary": "5000.00",
    "net_pay": "4200.00"
  },
  "processing_time": 2.34,
  "engines_used": ["tesseract", "paddle_ocr", "trocr"],
  "requires_human_verification": false
}
```

### Batch Process
```
POST /batch-process
Content-Type: multipart/form-data
files: [<image1>, <image2>, <image3>]

Response:
{
  "results": [
    {"filename": "img1.jpg", "document_type": "payslip", ...},
    {"filename": "img2.jpg", "document_type": "id_document", ...}
  ]
}
```

### Health Check
```
GET /health

Response:
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Configuration

### Elixir Application (config/dev.exs)

```elixir
config :app, :ocr_service_url, "http://localhost:8000"
config :app, :ocr_timeout, 30_000
config :app, :http_client, HTTPoison
```

### Environment Variables (OCR Service)

```bash
# Service configuration
OCR_SERVICE_HOST=0.0.0.0
OCR_SERVICE_PORT=8000
OCR_SERVICE_WORKERS=2

# Model paths (optional)
MODEL_CACHE_DIR=/app/models
TEMP_DIR=/app/temp

# Database (for job queuing)
DATABASE_URL=postgresql://user:pass@localhost/ocr_service

# Redis (for caching)
REDIS_URL=redis://localhost:6379
```

## Supported Document Types

1. **Payslips** - Employee salary documents with structured financial data
2. **ID Documents** - National ID, passports, driver's licenses
3. **Invoices** - Commercial invoices with amount and payment details
4. **Bank Statements** - Account statements with transaction data
5. **Contracts** - Legal agreements with signature fields
6. **Receipts** - Proof of payment documents
7. **Certificates** - Award and completion certificates
8. **Application Forms** - Various form types with structured fields

## OCR Engines

### Tesseract
- **Strength**: Fast, reliable baseline OCR
- **Use case**: General text extraction
- **Confidence**: Moderate (70-85%)

### PaddleOCR
- **Strength**: Good with various languages and layouts
- **Use case**: Complex document layouts
- **Confidence**: Good (75-90%)

### TrOCR (AI-powered)
- **Strength**: Transformer-based, handles handwriting
- **Use case**: Challenging text, handwritten documents
- **Confidence**: Excellent (80-95%)

## Consensus Algorithm

The service uses a weighted consensus approach:

1. **Run multiple OCR engines** on the same document
2. **Compare results** using text similarity algorithms
3. **Apply engine weights** based on historical performance
4. **Select consensus text** with highest agreement
5. **Calculate confidence** based on agreement quality
6. **Queue for verification** if confidence below threshold

## Error Handling

### Network Issues
- Automatic fallback to local OCR in Elixir client
- Retry mechanisms with exponential backoff
- Health check integration

### Low Confidence
- Automatic queuing for human verification
- Confidence threshold configuration
- Multiple processing strategies

### Service Unavailable
- Graceful degradation to local processing
- Service health monitoring
- Automatic recovery detection

## Performance

### Processing Times
- **Classification**: 0.5-2 seconds
- **Single OCR Engine**: 1-3 seconds
- **Multi-Engine Consensus**: 2-5 seconds
- **Batch Processing**: Parallel processing available

### Throughput
- **Single Instance**: ~100 documents/minute
- **Scaled Deployment**: Linear scaling with instances
- **Async Queue**: Unlimited with sufficient workers

## Security

### Input Validation
- File type verification (images only)
- Size limits (max 50MB per file)
- Malware scanning (optional)

### Data Privacy
- Temporary file cleanup
- No persistent storage of images
- Optional encryption for sensitive documents

### Access Control
- API key authentication (optional)
- Rate limiting
- Request logging

## Monitoring

### Metrics
- Processing time per document
- Confidence score distribution
- Engine performance comparison
- Error rates by type

### Logging
- Structured JSON logging
- Request/response logging
- Error stack traces
- Performance metrics

### Health Checks
- Service availability
- Model loading status
- Resource utilization
- Queue depth monitoring

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Adding New Document Types
1. Update `DocumentClassifier` with new type
2. Add extraction patterns in `OCREngine`
3. Update response models if needed
4. Add tests for new type

### Adding New OCR Engines
1. Implement engine in `OCREngine`
2. Add to consensus processor
3. Update weights configuration
4. Test against existing engines

## Deployment

### Docker Production
```bash
# Production build
docker build -t ocr-ai-service:prod .
docker run -d \
  -p 8000:8000 \
  -e OCR_SERVICE_WORKERS=4 \
  -e MODEL_CACHE_DIR=/app/models \
  ocr-ai-service:prod
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ocr-ai-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ocr-ai-service
  template:
    metadata:
      labels:
        app: ocr-ai-service
    spec:
      containers:
      - name: ocr-service
        image: ocr-ai-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: OCR_SERVICE_WORKERS
          value: "2"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

## Troubleshooting

### Common Issues

1. **Service Not Starting**
   - Check Docker logs: `docker logs ocr-ai-service`
   - Verify dependencies installed
   - Check port availability

2. **OCR Accuracy Low**
   - Try different consensus threshold
   - Enable AI enhancement
   - Check image quality

3. **Performance Slow**
   - Increase worker count
   - Use smaller images
   - Enable GPU acceleration

4. **Memory Issues**
   - Reduce worker count
   - Limit concurrent processing
   - Monitor model caching

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py
```

This AI OCR service provides enterprise-grade document processing with advanced AI capabilities while maintaining compatibility with your existing Elixir/Phoenix application.