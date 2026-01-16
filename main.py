from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
from datetime import datetime
import pytesseract
from PIL import Image
import io

app = FastAPI(
    title="AI OCR Service",
    description="Enhanced OCR service for document processing",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    """Classify document type using pattern matching"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_bytes = await file.read()
        
        # Extract text for classification
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        
        # Classify based on keywords
        doc_type, confidence = classify_by_keywords(text)
        
        return DocumentType(
            type=doc_type,
            confidence=confidence,
            description=get_type_description(doc_type)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/extract", response_model=OCRResponse)
async def extract_text(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    request: ProcessRequest = ProcessRequest()
):
    """Extract text using enhanced OCR"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        import time
        start_time = time.time()
        
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Extract text
        text = pytesseract.image_to_string(image)
        
        # Classify document
        doc_type, classification_confidence = classify_by_keywords(text)
        
        # Extract structured data based on document type
        structured_data = extract_structured_data(text, doc_type)
        
        # Calculate confidence
        confidence = calculate_overall_confidence(
            text, structured_data, classification_confidence
        )
        
        processing_time = time.time() - start_time
        
        response = OCRResponse(
            document_type=doc_type,
            confidence=confidence,
            extracted_text=text.strip(),
            structured_data=structured_data,
            processing_time=processing_time,
            engines_used=["tesseract_enhanced"],
            requires_human_verification=confidence < request.consensus_threshold
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

def classify_by_keywords(text: str) -> tuple:
    """Classify document based on keywords"""
    text_lower = text.lower()
    
    # Keyword matching
    doc_types = {
        "payslip": ["payslip", "salary", "pay", "employee", "income", "basic", "net"],
        "id_document": ["id", "identification", "national", "passport", "nrc", "name"],
        "invoice": ["invoice", "bill", "amount", "due", "payment", "total"],
        "contract": ["contract", "agreement", "terms", "signature", "party"],
        "bank_statement": ["bank", "statement", "account", "balance", "transaction"],
        "receipt": ["receipt", "proof", "payment", "cash", "amount"],
    }
    
    scores = {}
    for doc_type, keywords in doc_types.items():
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        scores[doc_type] = matches / len(keywords) if keywords else 0
    
    if scores:
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        # Boost confidence if text is substantial
        if len(text) > 200:
            confidence = min(confidence + 0.1, 0.95)
        
        return best_type, confidence
    
    return "unknown", 0.0

def get_type_description(doc_type: str) -> str:
    """Get human-readable description"""
    descriptions = {
        "payslip": "Employee salary payment document",
        "id_document": "Government-issued identification", 
        "invoice": "Commercial invoice for goods/services",
        "contract": "Legal agreement document",
        "bank_statement": "Bank account statement",
        "receipt": "Proof of payment document",
        "unknown": "Unrecognized document type"
    }
    return descriptions.get(doc_type, "Unknown document type")

def extract_structured_data(text: str, doc_type: str) -> Dict[str, Any]:
    """Extract structured data based on document type"""
    import re
    
    if doc_type == "payslip":
        data = {}
        
        # Clean and normalize text for better extraction
        cleaned_text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)  # Remove non-ASCII
        cleaned_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned_text)  # Remove control chars
        
        # Also keep original text for some patterns
        original_text = text
        
        # Extract employee name - look for firstname or name patterns
        name_patterns = [
            r'(?:firstname|first\s+name)[:\s\'"]*([A-Za-z]+)',
            r'Name[:\s]*([A-Za-z]+)',
            r'Employee[:\s]*([A-Za-z]+)'
        ]
        for pattern in name_patterns:
            match = re.search(pattern, cleaned_text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 1 and name.lower() not in ['number', 'department', 'position']:
                    data['employee_name'] = name
                    break
        
        # Extract employee number - handle various formats
        emp_no_patterns = [
            r'Employee\s+Number[:\s>]*([A-Z0-9\-]+)',
            r'Emp\s+No[:\s]*([A-Z0-9\-]+)',
            r'Staff\s+ID[:\s]*([A-Z0-9\-]+)',
            r'Employee\s+ID[:\s]*([A-Z0-9\-]+)'
        ]
        for pattern in emp_no_patterns:
            match = re.search(pattern, cleaned_text, re.IGNORECASE)
            if match:
                data['employee_no'] = match.group(1).strip()
                break
        
        # Extract department
        dept_patterns = [
            r'Department[:\s:]*([A-Za-z\s]+?)(?:\s+(?:Firstname|Position|Payment|$))',
            r'Dept[:\s]*([A-Za-z\s]+)'
        ]
        for pattern in dept_patterns:
            match = re.search(pattern, cleaned_text, re.IGNORECASE)
            if match:
                dept = match.group(1).strip()
                if len(dept) > 2 and dept.lower() not in ['it', 'hr', 'finance']:
                    data['department'] = dept
                    break
        
        # Extract position/role
        position_match = re.search(r'Position[:\s:]*([A-Za-z\s]+?)(?:\s+(?:Leave|Payment|$))', cleaned_text, re.IGNORECASE)
        if position_match:
            data['position'] = position_match.group(1).strip()
        
        # Extract monetary amounts - look for amounts after specific labels
        amount_patterns = {
            'basic_salary': [
                r'Basic\s+Pay[:\s]*([\d,]+\.?\d*)',
                r'Salary[:\s]*([\d,]+\.?\d*)'
            ],
            'gross_pay': [
                r'Gross\s+Pay[:\s]*([\d,]+\.?\d*)'
            ],
            'net_pay': [
                r'Net\s+Pay[:\s]*([\d,]+\.?\d*)',
                r'Net\s+Pay\s*\n+\s*([\d,]+\.?\d*)'  # Amount on next line
            ],
            'total_deductions': [
                r'Total\s+Deductions[:\s]*([\d,]+\.?\d*)'
            ]
        }
        
        for field, patterns in amount_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, original_text, re.IGNORECASE)
                if match:
                    data[field] = match.group(1).strip()
                    break
        
        # If net pay not found but we have gross and deductions, calculate it
        if 'net_pay' not in data and 'gross_pay' in data and 'total_deductions' in data:
            try:
                gross = float(data['gross_pay'].replace(',', ''))
                deductions = float(data['total_deductions'].replace(',', ''))
                net = gross - deductions
                data['net_pay'] = f"{net:.2f}"
            except (ValueError, KeyError):
                pass
        
        return data
    
    elif doc_type == "id_document":
        data = {}
        
        # NRC patterns (Zambia format)
        nrc_match = re.search(r'(\d{6}\/\d{2}\/\d{1})', text)
        if nrc_match:
            data['nrc'] = nrc_match.group(1)
        
        # Name patterns
        name_match = re.search(r'(?:name|surname)[:\s]+([A-Z][A-Z\s]+)', text, re.IGNORECASE)
        if name_match:
            data['full_name'] = name_match.group(1).strip()
        
        return data
    
    elif doc_type == "invoice":
        data = {}
        
        # Invoice number
        inv_match = re.search(r'(?:invoice\s*(?:no|number)?)[:\s]*(\w+-?\w*)', text, re.IGNORECASE)
        if inv_match:
            data['invoice_number'] = inv_match.group(1).strip()
        
        # Amount patterns
        amount_match = re.search(r'(?:total|amount)[:\s$]*([\d,\.]+)', text, re.IGNORECASE)
        if amount_match:
            data['total_amount'] = amount_match.group(1).strip()
        
        return data
    
    return {"raw_text": text}

def calculate_overall_confidence(text: str, structured_data: Dict, classification_confidence: float) -> float:
    """Calculate overall confidence score"""
    
    # Base confidence from classification
    confidence = classification_confidence * 0.4
    
    # Boost based on structured data completeness
    if structured_data and structured_data != {"raw_text": text}:
        filled_fields = sum(1 for value in structured_data.values() if value and str(value).strip())
        total_fields = len(structured_data)
        completeness_score = filled_fields / total_fields if total_fields > 0 else 0
        confidence += completeness_score * 0.4
    
    # Boost based on text length (substantial documents are more reliable)
    if len(text) > 100:
        length_score = min(len(text) / 1000, 0.2)
        confidence += length_score
    
    return min(confidence, 0.95)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )