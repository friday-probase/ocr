# AI Coding Agent Instructions for OCR AI Service

## Architecture Overview

This is a FastAPI-based OCR microservice that processes document images using multiple OCR engines with consensus algorithms. The service integrates with Elixir/Phoenix applications and supports structured data extraction from various document types.

### Core Components

- **main.py**: FastAPI application with REST endpoints (`/classify`, `/extract`, `/batch-process`, `/health`)
- **services/ocr_engine.py**: Multi-engine OCR processing (Tesseract, PaddleOCR, TrOCR)
- **services/document_classifier.py**: AI-powered document type classification
- **services/consensus_processor.py**: Weighted consensus algorithm for combining OCR results
- **models/response_models.py**: Pydantic models for API responses and data structures

### Data Flow

1. Image uploaded via multipart/form-data
2. Document classified using vision transformers + keyword analysis
3. Multiple OCR engines process image in parallel
4. Consensus processor combines results using weighted voting
5. Structured data extracted based on document type (payslip, ID, invoice, etc.)
6. Response includes confidence scores and human verification flags

## Key Patterns & Conventions

### OCR Engine Integration

- **Async processing**: All OCR operations use `async/await` for parallel execution
- **Engine weights**: TrOCR (0.4), PaddleOCR (0.35), Tesseract (0.25) - higher weights for AI-powered engines
- **Preprocessing**: Document-type-specific image preprocessing (contrast enhancement for payslips, sharpening for IDs)
- **Error handling**: Graceful fallback when engines fail - return empty results with 0 confidence

### Consensus Algorithm

- **Weighted similarity**: Use `difflib.SequenceMatcher` for text comparison
- **Structured data voting**: Fields with multiple values resolved by engine weight
- **Confidence calculation**: 70% similarity + 30% individual engine confidence
- **Threshold-based verification**: Low confidence results flagged for human review

### Document Processing

- **Type-specific extraction**: Regex patterns tailored to document formats (NRC format: `\d{6}\/\d{2}\/\d{1}`)
- **Robust payslip parsing**: Handles OCR errors with text cleaning, multi-pattern matching, and context-aware extraction
- **Fallback strategies**: AI classification falls back to keyword matching
- **Confidence boosting**: Substantial text (>200 chars) gets +0.1 confidence bonus

### API Design

- **Pydantic models**: All responses use typed models with `Optional` fields
- **Multipart uploads**: Image files handled via `UploadFile`
- **Background tasks**: Support for async processing (not fully implemented)
- **CORS enabled**: Configured for cross-origin requests from Elixir frontend

## Development Workflow

### Local Development
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run development server
python main.py
```

### Docker Development
```bash
# Build and run
docker build -t ocr-ai-service .
docker run -p 8000:8000 ocr-ai-service

# With environment variables
docker run -e OCR_SERVICE_WORKERS=4 -p 8000:8000 ocr-ai-service
```

### Testing OCR Engines
- Test individual engines in `services/ocr_engine.py`
- Use consensus processor for result comparison
- Check confidence scores against known documents

## Integration Points

### Elixir Client Integration
- Service designed for `App.AI.OcrClient` Elixir module
- Endpoints match Elixir client expectations
- Error handling compatible with Elixir patterns

### External Dependencies
- **Tesseract**: System-installed OCR engine
- **PaddleOCR**: Python package for advanced OCR
- **TrOCR**: HuggingFace transformers model
- **Vision Transformers**: For document classification

### Configuration
- Environment variables: `OCR_SERVICE_HOST`, `OCR_SERVICE_PORT`, `MODEL_CACHE_DIR`
- Consensus threshold: Default 0.8, configurable per request
- AI enhancement: Optional TrOCR processing

## Common Tasks

### Adding New Document Types
1. Add keywords to `DocumentClassifier.document_types`
2. Implement extraction logic in `OCREngine._extract_{type}_data()` or `main.extract_structured_data()`
3. Add Pydantic fields to `StructuredData` model
4. Update response examples in README

### Improving Document Extraction
- Use text cleaning to handle OCR errors (remove non-ASCII, normalize whitespace)
- Implement multiple regex patterns for robustness
- Look for currency symbols (£, $) near amount keywords
- Validate extracted data doesn't match header/footer text

### Adding New OCR Engines
1. Implement engine class method in `OCREngine`
2. Add to `extract_with_multiple_engines()` call
3. Set weight in `ConsensusProcessor.engine_weights`
4. Handle preprocessing in `_preprocess_for_{engine}()`

### Improving Accuracy
- Adjust engine weights based on performance testing
- Fine-tune preprocessing for specific document types
- Add document-type-specific regex patterns
- Implement custom confidence scoring

## File Organization

- **services/**: Business logic separated by concern
- **models/**: Data models and response schemas
- **utils/**: Shared utilities (currently empty)
- **main.py**: Thin API layer delegating to services

## Performance Considerations

- **Parallel processing**: Engines run concurrently with asyncio
- **GPU support**: TrOCR uses CUDA if available
- **Memory management**: Images processed in memory, temporary files cleaned up
- **Worker scaling**: Gunicorn workers configured in Dockerfile

## Error Handling

- **Engine failures**: Continue with remaining engines
- **Invalid images**: HTTP 400 with descriptive messages
- **Processing timeouts**: Async processing with configurable limits
- **Fallback responses**: Always return valid JSON, even on errors