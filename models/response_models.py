from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class DocumentType(BaseModel):
    type: str
    confidence: float
    description: Optional[str] = None

class StructuredData(BaseModel):
    employee_name: Optional[str] = None
    employee_no: Optional[str] = None
    basic_salary: Optional[str] = None
    net_pay: Optional[str] = None
    pay_month: Optional[str] = None
    department: Optional[str] = None
    gross_pay: Optional[str] = None
    taxable_income: Optional[str] = None
    total_deductions: Optional[str] = None
    position: Optional[str] = None
    # Add more fields based on document type
    
    class Config:
        extra = "allow"

class EngineResult(BaseModel):
    engine_name: str
    text: str
    confidence: float
    processing_time: float
    structured_data: Optional[Dict[str, Any]] = None

class OCRResponse(BaseModel):
    document_type: str
    confidence: float
    extracted_text: str
    structured_data: Optional[StructuredData] = None
    processing_time: float
    engines_used: List[str]
    requires_human_verification: bool
    metadata: Optional[Dict[str, Any]] = None

class ConsensusResult(BaseModel):
    text: str
    confidence: float
    engines_used: List[str]
    structured_data: Dict[str, Any]
    processing_time: float

class ProcessingRequest(BaseModel):
    document_types: Optional[List[str]] = None
    consensus_threshold: float = 0.8
    enable_ai_enhancement: bool = True
    language: str = "eng"

class BatchProcessRequest(BaseModel):
    files: List[str]
    processing_options: ProcessingRequest

class BatchProcessResponse(BaseModel):
    results: List[OCRResponse]
    total_files: int
    successful: int
    failed: int
    total_processing_time: float