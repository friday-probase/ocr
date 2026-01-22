# AI Coding Agent Instructions for OCR AI Service

## Architecture Overview

This is a FastAPI-based OCR microservice that processes document images using multiple OCR engines with consensus algorithms. The service integrates with Elixir/Phoenix applications and supports structured data extraction from various document types with **95% accuracy for scanned documents**.

### Core Components

- **main.py**: FastAPI application with REST endpoints (`/classify`, `/extract`, `/batch-process`, `/health`)
- **services/ocr_engine.py**: Multi-engine OCR processing (Tesseract, PaddleOCR, TrOCR) with advanced preprocessing
- **services/document_classifier.py**: AI-powered document type classification
- **services/consensus_processor.py**: Enhanced weighted consensus algorithm for combining OCR results
- **models/response_models.py**: Pydantic models for API responses and data structures

### Key Accuracy Improvements (95% Target)

**Advanced Multi-Engine OCR:**
- Tesseract: Baseline OCR with document-specific preprocessing
- PaddleOCR: Layout-aware OCR for complex documents
- TrOCR: Transformer-based AI model for handwritten/scanned text
- Consensus algorithm combines results with weighted voting

**Scanned Document Optimization:**
- Adaptive thresholding for varying scan quality
- Bilateral filtering for noise reduction while preserving edges
- Morphological operations for text cleanup
- Document-type-specific preprocessing (payslips, IDs, invoices)

**Enhanced Consensus Processing:**
- Multi-metric similarity calculation (sequence, word-level, length)
- Conflict resolution in text merging
- Adaptive confidence thresholds based on document characteristics
- AI engine boosting for scanned documents

## Data Flow

1. Image uploaded via multipart/form-data
2. Advanced AI vision classification (TrOCR + keyword fallback)
3. Parallel multi-engine OCR processing with scanned document preprocessing
4. Enhanced consensus algorithm combines results with 95% accuracy target
5. Structured data extracted and validated across engines
6. Response includes confidence scores and human verification flags

## Key Patterns & Conventions

### OCR Engine Integration

- **Async processing**: All OCR operations use `async/await` for parallel execution
- **Engine weights**: TrOCR (0.4), PaddleOCR (0.35), Tesseract (0.25) - higher weights for AI-powered engines
- **Scanned document preprocessing**: Adaptive thresholding, bilateral filtering, morphological cleanup
- **Error handling**: Graceful fallback when engines fail - return empty results with 0 confidence

### Consensus Algorithm

- **Multi-metric similarity**: Sequence matching (50%), word-level (30%), length similarity (20%)
- **Enhanced confidence**: Weighted similarity + engine confidence + text quality assessment
- **Adaptive thresholds**: Lower thresholds for high-agreement results and AI-detected content
- **Conflict resolution**: Intelligent text merging with quality-based line selection

### Document Processing

- **Type-specific extraction**: Regex patterns tailored to document formats (NRC format: `\d{6}\/\d{2}\/\d{1}`)
- **Robust payslip parsing**: Handles OCR errors with text cleaning, multi-pattern matching, and context-aware extraction
- **Scanned document optimization**: Enhanced preprocessing for varying scan quality and noise levels
- **Confidence boosting**: AI engines get +20% boost, substantial text (>100 chars) gets quality bonus

### API Design

- **Pydantic models**: All responses use typed models with `Optional` fields
- **Multipart uploads**: Image files handled via `UploadFile`
- **Batch processing**: Parallel processing of multiple documents
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

# With GPU support (if available)
docker run --gpus all -p 8000:8000 ocr-ai-service
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
- **TrOCR**: HuggingFace transformers model for scanned documents
- **OpenCV**: Advanced image preprocessing
- **Scikit-Image**: Noise reduction and filtering

### Configuration
- Environment variables: `OCR_SERVICE_HOST`, `OCR_SERVICE_PORT`, `MODEL_CACHE_DIR`
- Consensus threshold: Default 0.8, adaptive based on document characteristics
- AI enhancement: Always enabled for maximum accuracy

## Common Tasks

### Adding New Document Types
1. Add keywords to `DocumentClassifier.document_types`
2. Implement extraction logic in `OCREngine._extract_{type}_data()`
3. Add specialized preprocessing in `OCREngine._preprocess_{type}()`
4. Update response models if needed
5. Test with scanned samples for 95% accuracy

### Improving Scanned Document Accuracy
- Enhance preprocessing in `_enhance_scanned_image()` for specific noise patterns
- Adjust engine weights in `ConsensusProcessor.engine_weights` based on performance
- Fine-tune similarity metrics in `_calculate_multi_metric_similarity()`
- Add document-specific confidence boosting

### Adding New OCR Engines
1. Implement engine class method in `OCREngine`
2. Add to `extract_with_multiple_engines()` call
3. Set weight in `ConsensusProcessor.engine_weights`
4. Implement preprocessing in `_preprocess_for_{engine}()`

### Optimizing for 95% Accuracy
- Test with diverse scanned document samples
- Monitor confidence scores and adjust thresholds
- Enhance text quality assessment in `_assess_text_quality()`
- Improve conflict resolution in text merging

## File Organization

- **services/**: Advanced business logic separated by concern
- **models/**: Data models and response schemas
- **utils/**: Shared utilities (currently empty)
- **main.py**: Thin API layer delegating to advanced services

## Performance Considerations

- **Parallel processing**: Engines run concurrently with asyncio
- **GPU support**: TrOCR uses CUDA if available for faster processing
- **Memory management**: Images processed in memory, temporary files cleaned up
- **Batch processing**: Efficient parallel processing of multiple documents

## Error Handling

- **Engine failures**: Continue with remaining engines, maintain accuracy
- **Invalid images**: HTTP 400 with descriptive messages
- **Processing timeouts**: Async processing with configurable limits
- **Fallback responses**: Always return valid JSON, even on errors