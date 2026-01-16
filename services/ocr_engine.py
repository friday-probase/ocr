import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from paddleocr import PaddleOCR
from transformers import pipeline
import torch

from models.response_models import EngineResult

class OCREngine:
    def __init__(self):
        self.paddle_ocr = None
        self.trocr_pipeline = None
        self.tesseract_config = {
            'config': '--oem 3 --psm 6',
            'lang': 'eng'
        }
    
    async def initialize(self):
        """Initialize OCR engines"""
        if self.paddle_ocr is None:
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
        if self.trocr_pipeline is None:
            try:
                self.trocr_pipeline = pipeline(
                    "image-to-text",
                    model="microsoft/trocr-base-handwritten",
                    device=0 if torch.cuda.is_available() else -1
                )
            except Exception as e:
                print(f"TrOCR initialization failed: {e}")
                self.trocr_pipeline = None
    
    async def extract_with_multiple_engines(
        self, 
        image_bytes: bytes, 
        document_type: str = "unknown",
        enable_ai: bool = True
    ) -> List[EngineResult]:
        """Run multiple OCR engines and return results"""
        await self.initialize()
        
        # Convert bytes to different formats
        pil_image = Image.open(io.BytesIO(image_bytes))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Store results
        results = []
        
        # Engine 1: Tesseract (baseline)
        tesseract_result = await self._run_tesseract(cv_image, document_type)
        results.append(tesseract_result)
        
        # Engine 2: PaddleOCR
        paddle_result = await self._run_paddle_ocr(cv_image, document_type)
        results.append(paddle_result)
        
        # Engine 3: TrOCR (AI-powered) - if enabled and available
        if enable_ai and self.trocr_pipeline:
            trocr_result = await self._run_trocr(pil_image, document_type)
            results.append(trocr_result)
        
        return results
    
    async def _run_tesseract(self, image: np.ndarray, doc_type: str) -> EngineResult:
        """Run Tesseract OCR with optimized settings"""
        start_time = time.time()
        
        try:
            # Preprocess based on document type
            processed_image = self._preprocess_for_tesseract(image, doc_type)
            
            # Extract text
            text = pytesseract.image_to_string(
                processed_image,
                **self.tesseract_config
            )
            
            # Get confidence
            data = pytesseract.image_to_data(
                processed_image,
                output_type=pytesseract.Output.DICT,
                config=self.tesseract_config['config']
            )
            
            # Calculate average confidence for non-empty words
            confidences = [conf for conf, word in zip(data['conf'], data['text']) 
                          if conf > 0 and word.strip()]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Extract structured data based on document type
            structured_data = await self._extract_structured_data(text, doc_type)
            
            processing_time = time.time() - start_time
            
            return EngineResult(
                engine_name="tesseract",
                text=text.strip(),
                confidence=avg_confidence / 100.0,  # Convert to 0-1 scale
                processing_time=processing_time,
                structured_data=structured_data
            )
            
        except Exception as e:
            print(f"Tesseract error: {e}")
            return EngineResult(
                engine_name="tesseract",
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                structured_data={}
            )
    
    async def _run_paddle_ocr(self, image: np.ndarray, doc_type: str) -> EngineResult:
        """Run PaddleOCR engine"""
        start_time = time.time()
        
        try:
            result = self.paddle_ocr.ocr(image, cls=True)
            
            # Extract text and confidence
            texts = []
            confidences = []
            
            for line in result:
                for word_info in line:
                    if word_info:
                        texts.append(word_info[0] if isinstance(word_info[0], str) else "")
                        confidences.append(word_info[1][1] if len(word_info) > 1 else 0.0)
            
            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Extract structured data
            structured_data = await self._extract_structured_data(combined_text, doc_type)
            
            processing_time = time.time() - start_time
            
            return EngineResult(
                engine_name="paddle_ocr",
                text=combined_text.strip(),
                confidence=avg_confidence,
                processing_time=processing_time,
                structured_data=structured_data
            )
            
        except Exception as e:
            print(f"PaddleOCR error: {e}")
            return EngineResult(
                engine_name="paddle_ocr",
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                structured_data={}
            )
    
    async def _run_trocr(self, image: Image.Image, doc_type: str) -> EngineResult:
        """Run TrOCR AI model"""
        start_time = time.time()
        
        try:
            # TrOCR works best with smaller images
            if image.size[0] > 800:
                image = image.resize((800, int(image.size[1] * 800 / image.size[0])))
            
            # Extract text
            result = self.trocr_pipeline(image)
            text = result[0]['generated_text'] if result else ""
            
            # TrOCR doesn't provide confidence, so we estimate based on document type
            confidence = self._estimate_trocr_confidence(text, doc_type)
            
            # Extract structured data
            structured_data = await self._extract_structured_data(text, doc_type)
            
            processing_time = time.time() - start_time
            
            return EngineResult(
                engine_name="trocr",
                text=text.strip(),
                confidence=confidence,
                processing_time=processing_time,
                structured_data=structured_data
            )
            
        except Exception as e:
            print(f"TrOCR error: {e}")
            return EngineResult(
                engine_name="trocr",
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                structured_data={}
            )
    
    def _preprocess_for_tesseract(self, image: np.ndarray, doc_type: str) -> np.ndarray:
        """Preprocess image based on document type"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if doc_type == "payslip":
            # Payslips often have tables, enhance contrast
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        elif doc_type == "id_document":
            # IDs might have small text, sharpen
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            gray = cv2.filter2D(gray, -1, kernel)
        else:
            # General preprocessing
            gray = cv2.medianBlur(gray, 3)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        return gray
    
    async def _extract_structured_data(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Extract structured data based on document type"""
        structured_data = {}
        
        if doc_type == "payslip":
            structured_data = await self._extract_payslip_data(text)
        elif doc_type == "id_document":
            structured_data = await self._extract_id_data(text)
        elif doc_type == "invoice":
            structured_data = await self._extract_invoice_data(text)
        else:
            structured_data = {"raw_text": text}
        
        return structured_data
    
    async def _extract_payslip_data(self, text: str) -> Dict[str, Any]:
        """Extract payslip-specific data"""
        import re
        
        data = {}
        lines = text.split('\n')
        text_lower = text.lower()
        
        # Employee name
        name_patterns = [
            r'(?i)(?:name|employee)[:\s]+([A-Z][A-Z\s]+)',
            r'(?i)(?:MR|MRS|MS)\.?\s+([A-Z][A-Z\s]+)',
            r'^([A-Z][A-Z\s]{3,30})(?:\s+payslip|$)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                data['employee_name'] = match.group(1).strip()
                break
        
        # Salary amounts
        salary_patterns = [
            r'(?i)basic\s+salary[:\s$]*([\d,\.]+)',
            r'(?i)net\s+pay[:\s$]*([\d,\.]+)',
            r'(?i)gross\s+pay[:\s$]*([\d,\.]+)',
            r'(?i)taxable\s+income[:\s$]*([\d,\.]+)'
        ]
        
        for pattern in salary_patterns:
            match = re.search(pattern, text)
            if match:
                field_name = pattern.split(r'\s+')[1].replace('?', '')
                data[field_name.lower()] = match.group(1).strip()
        
        # Employee number
        emp_no_match = re.search(r'(?i)employee\s+(?:no|number)[:\s]*(\w+)', text)
        if emp_no_match:
            data['employee_no'] = emp_no_match.group(1).strip()
        
        return data
    
    async def _extract_id_data(self, text: str) -> Dict[str, Any]:
        """Extract ID document data"""
        import re
        
        data = {}
        
        # NRC patterns (Zambia format)
        nrc_match = re.search(r'(\d{6}\/\d{2}\/\d{1})', text)
        if nrc_match:
            data['nrc'] = nrc_match.group(1)
        
        # Name patterns
        name_match = re.search(r'(?i)(?:name|surname)[:\s]+([A-Z][A-Z\s]+)', text)
        if name_match:
            data['full_name'] = name_match.group(1).strip()
        
        return data
    
    async def _extract_invoice_data(self, text: str) -> Dict[str, Any]:
        """Extract invoice data"""
        import re
        
        data = {}
        
        # Invoice number
        inv_match = re.search(r'(?i)invoice\s*(?:no|number)?[:\s]*(\w+-?\w*)', text)
        if inv_match:
            data['invoice_number'] = inv_match.group(1).strip()
        
        # Amount patterns
        amount_match = re.search(r'(?i)(?:total|amount)[:\s$]*([\d,\.]+)', text)
        if amount_match:
            data['total_amount'] = amount_match.group(1).strip()
        
        return data
    
    def _estimate_trocr_confidence(self, text: str, doc_type: str) -> float:
        """Estimate confidence for TrOCR based on text characteristics"""
        if not text:
            return 0.0
        
        # Base confidence
        confidence = 0.7
        
        # Boost if text looks good
        if len(text) > 20:  # Decent length
            confidence += 0.1
        
        if any(char.isupper() for char in text):  # Has structure
            confidence += 0.1
        
        if doc_type == "payslip" and any(keyword in text.lower() 
                                       for keyword in ["salary", "pay", "employee"]):
            confidence += 0.1
        
        return min(confidence, 0.95)