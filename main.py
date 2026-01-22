from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
from datetime import datetime
import asyncio
import time

# Import advanced services
from services.ocr_engine import OCREngine
from services.document_classifier import DocumentClassifier
from services.consensus_processor import ConsensusProcessor
from models.response_models import OCRResponse, DocumentType, ProcessingRequest, BatchProcessRequest, BatchProcessResponse, PDFResponse, PageResult

app = FastAPI(
    title="AI OCR Service",
    description="Enhanced OCR service for document processing with 95% accuracy",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize advanced services
ocr_engine = OCREngine()
document_classifier = DocumentClassifier()
consensus_processor = ConsensusProcessor()

class ProcessRequest(BaseModel):
    consensus_threshold: float = 0.8
    enable_ai_enhancement: bool = True

class OCRResponse(BaseModel):
    document_type: str
    confidence: float
    extracted_text: str
    structured_data: Optional[Dict[str, Any]] = None
    processing_time: float
    engines_used: List[str]
    requires_human_verification: bool
    metadata: Optional[Dict[str, Any]] = None

class DocumentType(BaseModel):
    type: str
    confidence: float
    description: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "AI OCR Service is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/classify", response_model=DocumentType)
async def classify_document(file: UploadFile = File(...)):
    """Classify document type using advanced AI vision model"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_bytes = await file.read()
        
        # Use advanced document classifier
        result = await document_classifier.classify(image_bytes)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/extract", response_model=OCRResponse)
async def extract_text(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    request: ProcessingRequest = ProcessingRequest()
):
    """Extract text using advanced multi-engine OCR with consensus"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        start_time = time.time()
        
        image_bytes = await file.read()
        
        # Classify document first
        doc_classification = await document_classifier.classify(image_bytes)
        doc_type = doc_classification.type
        
        # Extract with multiple OCR engines
        engine_results = await ocr_engine.extract_with_multiple_engines(
            image_bytes=image_bytes,
            document_type=doc_type,
            enable_ai=request.enable_ai_enhancement
        )
        
        # Process consensus
        consensus_result = await consensus_processor.process_consensus(
            engine_results=engine_results,
            threshold=request.consensus_threshold
        )
        
        processing_time = time.time() - start_time
        
        # Create structured data from consensus
        structured_data = consensus_result.structured_data
        
        response = OCRResponse(
            document_type=doc_type,
            confidence=consensus_result.confidence,
            extracted_text=consensus_result.text.strip(),
            structured_data=structured_data,
            processing_time=processing_time,
            engines_used=consensus_result.engines_used,
            requires_human_verification=consensus_result.confidence < request.consensus_threshold
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@app.post("/extract-pdf", response_model=PDFResponse)
async def extract_pdf_text(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    request: ProcessingRequest = ProcessingRequest()
):
    """Extract text from PDF documents using advanced multi-engine OCR"""
    if not file.content_type == 'application/pdf':
        raise HTTPException(status_code=400, detail="File must be a PDF document")
    
    try:
        start_time = time.time()
        
        pdf_bytes = await file.read()
        
        # Process PDF pages
        page_results = await ocr_engine.process_pdf(
            pdf_bytes=pdf_bytes,
            document_type="unknown",  # Will be determined per page
            enable_ai=request.enable_ai_enhancement
        )
        
        if not page_results:
            raise HTTPException(status_code=500, detail="Failed to process PDF")
        
        # Combine results
        total_pages = len(page_results)
        combined_text = "\n\n".join([f"--- Page {page['page_number']} ---\n{page['extracted_text']}" 
                                    for page in page_results])
        
        # Calculate overall confidence (weighted average)
        total_confidence = sum(page['confidence'] for page in page_results)
        overall_confidence = total_confidence / total_pages if total_pages > 0 else 0
        
        # Collect all engines used
        all_engines = set()
        for page in page_results:
            all_engines.update(page['engines_used'])
        engines_used = list(all_engines)
        
        # Check if any page requires verification
        requires_verification = any(page['requires_human_verification'] for page in page_results)
        
        # Merge structured data from all pages
        merged_structured_data = {}
        for page in page_results:
            if page['structured_data']:
                merged_structured_data.update(page['structured_data'])
        
        # Convert page results to PageResult objects
        pages = [
            PageResult(
                page_number=page['page_number'],
                document_type=page['document_type'],
                confidence=page['confidence'],
                extracted_text=page['extracted_text'],
                structured_data=page['structured_data'],
                processing_time=page['processing_time'],
                engines_used=page['engines_used'],
                requires_human_verification=page['requires_human_verification']
            )
            for page in page_results
        ]
        
        total_processing_time = time.time() - start_time
        
        response = PDFResponse(
            total_pages=total_pages,
            overall_confidence=overall_confidence,
            combined_text=combined_text,
            pages=pages,
            structured_data=merged_structured_data if merged_structured_data else None,
            total_processing_time=total_processing_time,
            engines_used=engines_used,
            requires_human_verification=requires_verification,
            metadata={
                "file_name": file.filename,
                "file_size": len(pdf_bytes),
                "processing_method": "multi-engine OCR with consensus"
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

@app.post("/batch-process", response_model=BatchProcessResponse)
async def batch_process_documents(
    files: List[UploadFile] = File(...),
    processing_options: ProcessingRequest = ProcessingRequest()
):
    """Process multiple documents in batch with advanced OCR"""
    
    results = []
    total_processing_time = 0
    successful = 0
    failed = 0
    
    for file in files:
        try:
            start_time = time.time()
            
            file_bytes = await file.read()
            
            # Handle different file types
            if file.content_type == 'application/pdf':
                # Process PDF
                page_results = await ocr_engine.process_pdf(
                    pdf_bytes=file_bytes,
                    document_type="unknown",
                    enable_ai=processing_options.enable_ai_enhancement
                )
                
                if not page_results:
                    failed += 1
                    continue
                
                # Create combined result for PDF
                total_pages = len(page_results)
                combined_text = "\n\n".join([f"--- Page {page['page_number']} ---\n{page['extracted_text']}" 
                                            for page in page_results])
                
                total_confidence = sum(page['confidence'] for page in page_results)
                overall_confidence = total_confidence / total_pages if total_pages > 0 else 0
                
                all_engines = set()
                for page in page_results:
                    all_engines.update(page['engines_used'])
                
                merged_structured_data = {}
                for page in page_results:
                    if page['structured_data']:
                        merged_structured_data.update(page['structured_data'])
                
                result = OCRResponse(
                    document_type="pdf_document",
                    confidence=overall_confidence,
                    extracted_text=combined_text,
                    structured_data=merged_structured_data if merged_structured_data else None,
                    processing_time=time.time() - start_time,
                    engines_used=list(all_engines),
                    requires_human_verification=overall_confidence < processing_options.consensus_threshold,
                    metadata={
                        "file_name": file.filename,
                        "file_type": "pdf",
                        "total_pages": total_pages
                    }
                )
                
            elif file.content_type.startswith('image/'):
                # Process image (existing logic)
                # Classify document
                doc_classification = await document_classifier.classify(file_bytes)
                doc_type = doc_classification.type
                
                # Extract with multiple OCR engines
                engine_results = await ocr_engine.extract_with_multiple_engines(
                    image_bytes=file_bytes,
                    document_type=doc_type,
                    enable_ai=processing_options.enable_ai_enhancement
                )
                
                # Process consensus
                consensus_result = await consensus_processor.process_consensus(
                    engine_results=engine_results,
                    threshold=processing_options.consensus_threshold
                )
                
                # Create structured data from consensus
                structured_data = consensus_result.structured_data
                
                result = OCRResponse(
                    document_type=doc_type,
                    confidence=consensus_result.confidence,
                    extracted_text=consensus_result.text.strip(),
                    structured_data=structured_data,
                    processing_time=time.time() - start_time,
                    engines_used=consensus_result.engines_used,
                    requires_human_verification=consensus_result.confidence < processing_options.consensus_threshold,
                    metadata={
                        "file_name": file.filename,
                        "file_type": "image"
                    }
                )
            else:
                failed += 1
                continue
            
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            
            results.append(result)
            successful += 1
            
            results.append(result)
            successful += 1
            
        except Exception as e:
            failed += 1
            print(f"Failed to process {file.filename}: {str(e)}")
    
    return BatchProcessResponse(
        results=results,
        total_files=len(files),
        successful=successful,
        failed=failed,
        total_processing_time=total_processing_time
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )