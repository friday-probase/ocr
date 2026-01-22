import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from paddleocr import PaddleOCR
from transformers import pipeline
import torch
from skimage import filters, morphology
from skimage.restoration import denoise_bilateral
from pdf2image import convert_from_bytes
import tempfile
import os
import warnings

from models.response_models import EngineResult

# Set environment variables to prevent PaddlePaddle OneDNN/PirAttribute errors
os.environ['FLAGS_enable_mkldnn'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_dnnl'] = '0'  # Alternative flag name for older versions
os.environ['FLAGS_allocator_strategy'] = 'naive_best_fit'
os.environ['FLAGS_fraction_of_cpu_memory_to_use'] = '0.5'
os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'
os.environ['FLAGS_memory_fraction_of_eager_deletion'] = '1.0'
os.environ['PADDLE_DISABLE_JIT'] = '1'
os.environ['PADDLE_DISABLE_IR_OPTIMIZE'] = '1'
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")
warnings.filterwarnings("ignore", message=".*Using a slow image processor.*")

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
            try:
                import os
                # Ensure environment variables are set before initializing PaddleOCR
                os.environ['FLAGS_enable_mkldnn'] = '0'
                os.environ['FLAGS_use_mkldnn'] = '0'
                os.environ['FLAGS_use_dnnl'] = '0'
                os.environ['FLAGS_allocator_strategy'] = 'naive_best_fit'
                os.environ['FLAGS_fraction_of_cpu_memory_to_use'] = '0.5'
                os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'
                os.environ['FLAGS_memory_fraction_of_eager_deletion'] = '1.0'
                os.environ['PADDLE_DISABLE_JIT'] = '1'
                os.environ['PADDLE_DISABLE_IR_OPTIMIZE'] = '1'
                os.environ['FLAGS_prim_all'] = '0'
                os.environ['FLAGS_enable_pir_api'] = '0'
                os.environ['FLAGS_enable_pir_in_executor'] = '0'
                
                # Initialize PaddleOCR - use environment variables for OneDNN settings
                self.paddle_ocr = PaddleOCR(lang='en')
            except Exception as e:
                print(f"PaddleOCR initialization failed: {e}")
                self.paddle_ocr = None
        
        # Initialize TrOCR with better error handling and options
        if self.trocr_pipeline is None:
            try:
                import warnings
                warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")
                warnings.filterwarnings("ignore", message=".*Using a slow image processor.*")
                
                # Try to use a lighter model first, fall back to heavier if needed
                try:
                    self.trocr_pipeline = pipeline(
                        "image-to-text",
                        model="microsoft/trocr-base-handwritten",
                        device=0 if torch.cuda.is_available() else -1,
                        use_fast=True
                    )
                except Exception:
                    # Fall back to printed model if handwritten fails
                    self.trocr_pipeline = pipeline(
                        "image-to-text",
                        model="microsoft/trocr-base-printed",
                        device=0 if torch.cuda.is_available() else -1,
                        use_fast=True
                    )
            except Exception as e:
                print(f"TrOCR initialization failed: {e}")
                self.trocr_pipeline = None
        
        # Initialize additional AI models for better recognition
        try:
            # Attempt to initialize layoutlm for document understanding
            from transformers import AutoTokenizer, VisionEncoderDecoderModel, AutoFeatureExtractor
            # Use a lightweight model for layout analysis
            self.layout_model = None  # Placeholder for future implementation
        except ImportError:
            self.layout_model = None
    
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
        
        # Engine 1: Tesseract (baseline) - always run (fast and reliable)
        try:
            tesseract_result = await asyncio.wait_for(
                self._run_tesseract(cv_image, document_type),
                timeout=30.0
            )
            results.append(tesseract_result)
        except (asyncio.TimeoutError, Exception) as e:
            print(f"Tesseract error: {e}")
            results.append(EngineResult(
                engine_name="tesseract",
                text="",
                confidence=0.0,
                processing_time=0.0,
                structured_data={}
            ))
        
        # Engine 2: PaddleOCR (optional - can be slow)
        # Disabled by default due to initialization time, can be enabled for production
        if self.paddle_ocr:
            try:
                paddle_result = await asyncio.wait_for(
                    self._run_paddle_ocr(cv_image, document_type),
                    timeout=60.0
                )
                results.append(paddle_result)
            except (asyncio.TimeoutError, Exception) as e: 
                print(f"PaddleOCR error: {e}")
        
        # Engine 3: TrOCR (AI-powered, optional) - only if enabled and available
        # Requires significant model download, disabled by default
        if enable_ai and self.trocr_pipeline:
            try:
                trocr_result = await asyncio.wait_for(
                    self._run_trocr(pil_image, document_type),
                    timeout=45.0
                )
                results.append(trocr_result)
            except (asyncio.TimeoutError, Exception) as e:
                print(f"TrOCR error: {e}")
        
        return results
    
    async def _run_tesseract(self, image: np.ndarray, doc_type: str) -> EngineResult:
        """Run Tesseract OCR with optimized settings"""
        start_time = time.time()
        
        try:
            # Preprocess based on document type
            processed_image = self._preprocess_for_tesseract(image, doc_type)
            
            # For payslips, use PSM 4 (single column) which works better for table structures
            if doc_type == "payslip":
                config = '--oem 3 --psm 4'
            else:
                config = self.tesseract_config['config']
            
            # Extract text with primary config
            text = pytesseract.image_to_string(
                processed_image,
                config=config,
                lang=self.tesseract_config['lang']
            )
            
            # Get confidence
            data = pytesseract.image_to_data(
                processed_image,
                output_type=pytesseract.Output.DICT,
                config=config
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
            result = self.paddle_ocr.ocr(image)
            
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
            # Preprocess image for better TrOCR results
            processed_image = self._preprocess_image_for_ocr_ai(image)
            
            # Extract text
            result = self.trocr_pipeline(processed_image)
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
    
    def _preprocess_image_for_ocr_ai(self, image: Image.Image) -> Image.Image:
        """Preprocess image specifically for AI OCR models like TrOCR"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for optimal AI model performance (TrOCR works well with images around 1024px max dimension)
        max_dimension = 1024
        width, height = image.size
        if max(width, height) > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * max_dimension / width)
            else:
                new_height = max_dimension
                new_width = int(width * max_dimension / height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Ensure minimum size for detail preservation
        min_dimension = 224  # Minimum for most vision models
        if min(width, height) < min_dimension:
            if width < height:
                new_width = min_dimension
                new_height = int(height * min_dimension / width)
            else:
                new_height = min_dimension
                new_width = int(width * min_dimension / height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def _preprocess_for_tesseract(self, image: np.ndarray, doc_type: str) -> np.ndarray:
        """Advanced preprocessing for scanned documents"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply advanced preprocessing for scanned documents
        gray = self._enhance_scanned_image(gray, doc_type)
        
        # Apply document-type specific preprocessing
        if doc_type == "payslip":
            gray = self._preprocess_payslip(gray)
        elif doc_type == "id_document":
            gray = self._preprocess_id_document(gray)
        elif doc_type == "invoice":
            gray = self._preprocess_invoice(gray)
        else:
            gray = self._preprocess_general(gray)
        
        return gray
    
    def _enhance_scanned_image(self, image: np.ndarray, doc_type: str) -> np.ndarray:
        """Enhance scanned image quality"""
        # Convert to float for better precision
        image_float = image.astype(np.float32)
        
        # Denoise using bilateral filter (preserves edges while removing noise)
        image = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image)
        
        # Apply unsharp mask to enhance text
        gaussian_blur = cv2.GaussianBlur(image, (0, 0), 2.0)
        image = cv2.addWeighted(image, 1.5, gaussian_blur, -0.5, 0)
        
        # Thresholding with Otsu's method for better binarization
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Additional noise removal
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel)
        
        return image
    
    def _preprocess_payslip(self, image: np.ndarray) -> np.ndarray:
        """Specialized preprocessing for payslips"""
        # Payslips often have tables and small text
        # Enhance contrast and sharpness
        image = cv2.convertScaleAbs(image, alpha=1.7, beta=15)
        
        # Apply Gaussian blur followed by unsharp masking
        blurred = cv2.GaussianBlur(image, (0, 0), 3.0)
        image = cv2.addWeighted(image, 1.8, blurred, -0.8, 0)
        
        # Enhance horizontal and vertical lines (tables)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine the lines to preserve table structure
        table_lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
        
        # Add the lines back to the original image
        image = cv2.addWeighted(image, 1.0, table_lines, 0.1, 0)
        
        return image
    
    def _preprocess_id_document(self, image: np.ndarray) -> np.ndarray:
        """Specialized preprocessing for ID documents"""
        # IDs often have photos and small text - enhance both
        # Apply CLAHE for better contrast in photos and text
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        
        # Enhance edges for text readability
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)
        
        # Reduce noise while preserving text
        image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        return image
    
    def _preprocess_invoice(self, image: np.ndarray) -> np.ndarray:
        """Specialized preprocessing for invoices"""
        # Invoices often have structured layouts and monetary values
        # Enhance contrast using histogram equalization
        image = cv2.equalizeHist(image)
        
        # Enhance text sharpness
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)
        
        # Preserve table structure
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, rect_kernel)
        
        return image
    
    def _preprocess_general(self, image: np.ndarray) -> np.ndarray:
        """General preprocessing for unknown document types"""
        # Apply adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        
        # Apply bilateral filter for noise reduction while keeping edges sharp
        image = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Slight sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        image = cv2.filter2D(image, -1, kernel)
        
        return image
    
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
        """Extract payslip-specific data with improved patterns for Zambian payslips"""
        import re
        
        data = {}
        lines = text.split('\n')
        text_lower = text.lower()
        
        # ===== ENHANCED NET PAY EXTRACTION FOR ZAMBIAN PAYSLIPS =====
        # This section specifically handles Zambian government e-payslip format
        # where "NET PAY" appears on its own line with the amount below
        
        net_pay_value = self._extract_netpay_zambian_format(text, lines)
        if net_pay_value:
            data['net_pay'] = net_pay_value
        
        # Employee name - Multiple patterns to catch different formats
        name_patterns = [
            r'(?i)(?:name|employee)[:\s]+([A-Z][A-Z\s\.]+?)(?:\s+(?:department|position|emp|id|no\.?|#)|$)',
            r'(?i)(?:mr|mrs|ms|miss|dr|prof)\.?\s+([A-Z][A-Z\s\.]+?)(?:\s+[A-Z]{2,}|$)',  # With titles
            r'(?i)(?:surname|last\s+name)[:\s]*([A-Z][A-Z\s\.]+)',
            r'(?i)(?:first\s+name|given\s+name)[:\s]*([A-Z][A-Z\s\.]+)',
            r'(?i)^([A-Z][A-Z\s\.]{3,30}?)(?:\s+payslip|\s+summary|\s+statement|$)',  # At start of document
            r'(?i)(?:employee|staff)\s+(?:name|details?)\s*[:\-\s]+([A-Z][A-Z\s\.]+?)(?:\s+\d+|$)'  # With employee number nearby
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean up the name (remove trailing numbers, punctuation)
                name = re.sub(r'\s*[,\.\-;:]\s*$', '', name)
                if len(name) >= 3:  # Reasonable name length
                    data['employee_name'] = name
                    break
        
        # Employee number patterns
        emp_patterns = [
            r'(?i)employee\s+(?:no\.?|number|id)[:\s]*(\w[\w\-]*\w)',
            r'(?i)(?:emp\.?\s*no\.?|id)[:\s]*(\w[\w\-]*\w)',
            r'(?i)(?:EMP\d{3,}|EMP[\s\-]?\d{3,}|\b\d{4,8}\b)'  # Standalone patterns
        ]
        
        for pattern in emp_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                emp_no = match.group(1) if len(match.groups()) > 0 else match.group(0)
                data['employee_no'] = emp_no.strip()
                break
        
        # Department/Position
        dept_patterns = [
            r'(?i)(?:department|dept)[:\s]*([A-Z][a-z\s&]+?)(?:\s+(?:manager|supervisor|position|role)|$)',
            r'(?i)(?:position|job|title)[:\s]*([A-Z][a-z\s&]+?)(?:\s+(?:department|manager|start|date)|$)',
            r'(?i)(?:designation|role)[:\s]*([A-Z][a-z\s&]+?)(?:\s+(?:department|manager|employee)|$)'
        ]
        
        for pattern in dept_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                data['department'] = match.group(1).strip()
                break
        
        # Financial amounts - More comprehensive patterns
        financial_patterns = {
            'basic_salary': [
                r'(?i)basic\s+(?:salary|pay)[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)',
                r'(?i)basic[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)',
                r'(?i)(?:annual|monthly)\s+basic[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)'
            ],
            'gross_pay': [
                r'(?i)gross\s+(?:pay|earnings|income)[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)',
                r'(?i)gross[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)',
                r'(?i)total\s+earnings[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)',
                r'(?i)taxable\s+income[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)'
            ],
            'total_deductions': [
                r'(?i)total\s+deductions?[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)',
                r'(?i)deds?\.?[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)',
                r'(?i)total\s+deds?\.?[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)'
            ],
            'tax': [
                r'(?i)(?:income\s+)?tax[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)',
                r'(?i)PAYE[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)',
                r'(?i)P\.?A\.?Y\.?E\.?[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)'
            ],
            'social_security': [
                r'(?i)(?:social\s+security|sss|pension)[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)',
                r'(?i)NAPSA[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)',
                r'(?i)pension\s+(?:fund|life)[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)'
            ],
            'health_insurance': [
                r'(?i)(?:health\s+insurance|medical|mediclaim)[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)',
                r'(?i)medical[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)',
                r'(?i)national\s+health[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)'
            ],
            'allowances': [
                r'(?i)(?:housing|house)\s+allowance[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)',
                r'(?i)transport\s+allowance[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)'
            ],
            'overtime': [
                r'(?i)overtime[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)',
                r'(?i)ot\s+pay[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)'
            ]
        }
        
        # Extract financial fields (skip net_pay if already extracted)
        for field, patterns in financial_patterns.items():
            if field in data:  # Skip if already extracted
                continue
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    amount = match.group(1).strip()
                    if amount and self._is_valid_amount(amount):
                        if field not in data:
                            data[field] = self._normalize_amount(amount)
                        break
        
        # Fallback net pay extraction if not found with Zambian format
        if 'net_pay' not in data:
            net_pay_patterns = [
                r'(?i)net\s+(?:pay|salary|earnings|amount)[\s:]*[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)',
                r'(?i)take\s+home\s+(?:pay|amount)[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)',
                r'(?i)net\s+(?:amount|total)[:\s\$€£¥K]*([0-9,]+\.?[0-9]*)',
                r'(?i)(?:total\s+)?net[:\s\$€£¥K]+([0-9,]+\.?[0-9]*)',
            ]
            for pattern in net_pay_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    amount = match.group(1).strip()
                    if self._is_valid_amount(amount):
                        data['net_pay'] = self._normalize_amount(amount)
                        break
        
        # Pay period/dates
        date_patterns = [
            r'(?i)pay\s+(?:period|month)[:\s]*(\d{1,2}[\.\/]\d{1,2}[\.\/]\d{2,4})',
            r'(?i)pay\s+(?:period|month)[:\s]*([A-Z][a-z]+\s+\d{4})',
            r'(?i)pay\s+date[:\s]*(\w+\s+\d{1,2},?\s+\d{4})',
            r'(?i)month[:\s]*([A-Z][a-z]+\s+\d{4})',
            r'(?i)(\d{1,2}[\.\/]\d{1,2}[\.\/]\d{2,4})'  # Date formats like 31.07.2024
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1) if match.groups() else match.group(0)
                if 'pay_month' not in data:
                    data['pay_month'] = date_str.strip()
                break
        
        return data
    
    def _extract_netpay_zambian_format(self, text: str, lines: List[str]) -> Optional[str]:
        """
        Extract net pay specifically for Zambian government e-payslip format.
        Handles formats where 'NET PAY' is on its own line with the amount below.
        """
        import re
        
        # Method 1: Line-by-line detection for "NET PAY" followed by amount
        for i, line in enumerate(lines):
            line_clean = line.strip().upper()
            
            # Check if this line contains "NET PAY"
            if 'NET PAY' in line_clean or 'NET-PAY' in line_clean or 'NETPAY' in line_clean:
                # Try to extract amount from the same line first
                amount_match = re.search(r'([0-9,]+\.[0-9]{2})', line)
                if amount_match:
                    amount = amount_match.group(1)
                    if self._is_valid_amount(amount):
                        return self._normalize_amount(amount)
                
                # Check the next line for the amount
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    amount_match = re.search(r'([0-9,]+\.[0-9]{2})', next_line)
                    if amount_match:
                        amount = amount_match.group(1)
                        if self._is_valid_amount(amount):
                            return self._normalize_amount(amount)
        
        # Method 2: Pattern matching for multi-line format
        multiline_patterns = [
            r'(?i)NET\s+PAY\s*\n+\s*([0-9,]+\.[0-9]{2})',
            r'(?i)NET\s+PAY\s+[^\d]*([0-9,]+\.[0-9]{2})',
            r'(?i)NETPAY\s*[:\-]?\s*([0-9,]+\.[0-9]{2})',
        ]
        
        for pattern in multiline_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                amount = match.group(1)
                if self._is_valid_amount(amount):
                    return self._normalize_amount(amount)
        
        # Method 3: Table footer detection - look for TOTALS row and NET PAY
        # Zambian payslips often have: TOTALS <payment_amount> <deduction_amount> then NET PAY <amount>
        totals_pattern = r'(?i)TOTALS?\s+[\-\s]*([0-9,]+\.?[0-9]*)\s+[\-\s]*([0-9,]+\.?[0-9]*)\s*[\-\s]*\n\s*NET\s+PAY\s+[\-\s]*([0-9,]+\.?[0-9]*)'
        match = re.search(totals_pattern, text, re.MULTILINE)
        if match:
            amount = match.group(3)
            if self._is_valid_amount(amount):
                return self._normalize_amount(amount)
        
        # Method 4: Find the last occurrence of a reasonable payslip amount near NET PAY
        net_pay_section = re.search(r'(?i)(NET\s+PAY.{0,100})', text, re.DOTALL)
        if net_pay_section:
            section = net_pay_section.group(1)
            amounts = re.findall(r'([0-9,]+\.[0-9]{2})', section)
            if amounts:
                # Return the first amount found in the NET PAY section
                for amount in amounts:
                    if self._is_valid_amount(amount):
                        return self._normalize_amount(amount)
        
        return None
    
    def _is_valid_amount(self, amount: str) -> bool:
        """Check if an amount string is a valid payslip amount"""
        try:
            # Remove commas and convert to float
            value = float(amount.replace(',', ''))
            # Valid payslip amounts are typically between 0 and 10,000,000
            return 0 < value < 10000000
        except (ValueError, AttributeError):
            return False
    
    def _normalize_amount(self, amount: str) -> str:
        """Normalize amount format (keep commas for readability)"""
        # Just clean up any extra whitespace
        return amount.strip()
    
    async def _extract_id_data(self, text: str) -> Dict[str, Any]:
        """Extract ID document data with comprehensive patterns"""
        import re
        
        data = {}
        
        # ID/National ID patterns (various formats)
        id_patterns = [
            r'(\d{6}/\d{2}/\d{1})',  # Zambia format: 123456/78/9
            r'(\d{3}-\d{3}-\d{3}-\d{3})',  # Format with dashes
            r'(\d{6}\d{2}\d{1})',  # Concatenated format
            r'(\d{13})',  # 13-digit national ID
            r'(\d{9,12})',  # 9-12 digit ID numbers
            r'(?i)ID[:\s]*(\w[\w\s-]*\w)',  # ID with text
        ]
        
        for pattern in id_patterns:
            match = re.search(pattern, text)
            if match:
                id_number = match.group(1).strip()
                if 'id_number' not in data:
                    data['id_number'] = id_number
                break
        
        # Full name patterns
        name_patterns = [
            r'(?i)(?:surname|last\s+name)[:\s]*([A-Z][A-Z\s\.]+)',
            r'(?i)(?:given\s+name|first\s+name)[:\s]*([A-Z][A-Z\s\.]+)',
            r'(?i)(?:name|holder)[:\s]*([A-Z][A-Z\s\.]+?)(?:\s+\d{4}|\s+[A-Z]{2,}|\n|$)',
            r'(?i)(?:MR|MRS|MS|MISS|DR)\.?\s+([A-Z][A-Z\s\.]+)',  # With titles
            r'(?i)NAME[:\s]*([A-Z][A-Z\s\.]+)',  # Simple format
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if 'full_name' not in data and len(name) >= 3:
                    data['full_name'] = name
                    break
        
        # Date of birth
        dob_patterns = [
            r'(?i)(?:DOB|Date\s+of\s+Birth)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            r'(?i)(?:birth\s+date|date\s+birth)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})'  # Standalone date
        ]
        
        for pattern in dob_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if 'date_of_birth' not in data:
                    data['date_of_birth'] = match.group(1)
                    break
        
        # Expiry date
        expiry_patterns = [
            r'(?i)(?:expiry|expire|valid\s+until)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            r'(?i)(?:valid\s+thru|thru\s+date)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})'
        ]
        
        for pattern in expiry_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if 'expiry_date' not in data:
                    data['expiry_date'] = match.group(1)
                    break
        
        # Address (if present)
        address_patterns = [
            r'(?i)(?:address|residence)[:\s]*([A-Z0-9][^,\n.]*?)(?:\s+\d{4}|\s+[A-Z]{2,}|\n|$)',
            r'(?i)(?:residing\s+at|living\s+at)[:\s]*([A-Z0-9][^,\n.]*?)(?:\s+\d{4}|\s+[A-Z]{2,}|\n|$)'
        ]
        
        for pattern in address_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                address = match.group(1).strip()
                if 'address' not in data and len(address) >= 10:  # Reasonable address length
                    data['address'] = address
                    break
        
        return data
    
    async def _extract_invoice_data(self, text: str) -> Dict[str, Any]:
        """Extract invoice data with comprehensive patterns"""
        import re
        
        data = {}
        
        # Invoice number (comprehensive patterns)
        invoice_patterns = [
            r'(?i)INVOICE\s*(?:NO\.?|NUMBER)?[:\s]*(\w[\w\-\/]*\w)',
            r'(?i)INV\s*[:\s]*(\w[\w\-\/]*\w)',
            r'(?i)#\s*(\w[\w\-\/]*\w)',  # Invoice number with #
            r'(?i)BILL\s+TO\s*[:\s]*\w[\w\s\-]*?(\d[\d\-]*)',  # From bill to section
            r'(?:INV|INVOICE|REF)\s*\d+',  # Standalone invoice references
        ]
        
        for pattern in invoice_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                inv_number = match.group(1) if len(match.groups()) > 0 else match.group(0)
                if 'invoice_number' not in data:
                    data['invoice_number'] = inv_number.strip()
                    break
        
        # Customer/Vendor info
        customer_patterns = [
            r'(?i)(?:Bill\s+To|To|Customer)[:\s\n]*([A-Z][A-Za-z\s\-\.\&]+?)(?:\n\s*\d|\n\s*[A-Z]|$)',
            r'(?i)(?:From|Vendor|Supplier)[:\s\n]*([A-Z][A-Za-z\s\-\.\&]+?)(?:\n\s*\d|\n\s*[A-Z]|$)',
        ]
        
        for pattern in customer_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                company = match.group(1).strip()
                if 'customer_name' not in data and len(company) >= 5:
                    data['customer_name'] = company
                    break
        
        # Dates
        date_patterns = [
            r'(?i)(?:Invoice\s+Date|Date)[:\s]*(\w+\s+\d{1,2},?\s+\d{4})',
            r'(?i)(?:Due\s+Date|Payment\s+Due)[:\s]*(\w+\s+\d{1,2},?\s+\d{4})',
            r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})'  # Standalone dates
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if 'invoice_date' not in data:
                    data['invoice_date'] = match
                elif 'due_date' not in data:
                    data['due_date'] = match
                    break
        
        # Total amounts (comprehensive)
        amount_patterns = [
            r'(?i)(?:Total|TOTAL|Amount\s+Due|Balance\s+Due)[:\s\$€£¥]*([0-9,]+\.?[0-9]*)',
            r'(?i)(?:Net\s+Total|Grand\s+Total)[:\s\$€£¥]*([0-9,]+\.?[0-9]*)',
            r'(?i)SUBTOTAL[:\s\$€£¥]*([0-9,]+\.?[0-9]*)',
            r'(?i)AMOUNT[:\s\$€£¥]*([0-9,]+\.?[0-9]*)',
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount = match.group(1).strip()
                if amount.replace(',', '').replace('.', '').isdigit():
                    if 'total_amount' not in data:
                        data['total_amount'] = amount
                    break
        
        # Tax information
        tax_patterns = [
            r'(?i)Tax[:\s\$€£¥]*([0-9,]+\.?[0-9]*)',
            r'(?i)VAT[:\s\$€£¥]*([0-9,]+\.?[0-9]*)',
            r'(?i)GST[:\s\$€£¥]*([0-9,]+\.?[0-9]*)',
        ]
        
        for pattern in tax_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                tax_amount = match.group(1).strip()
                if tax_amount.replace(',', '').replace('.', '').isdigit() and 'tax_amount' not in data:
                    data['tax_amount'] = tax_amount
                    break
        
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
    
    async def process_pdf(
        self, 
        pdf_bytes: bytes, 
        document_type: str = "unknown",
        enable_ai: bool = True  # Enable AI by default for better accuracy
    ) -> List[Dict[str, Any]]:
        """Process PDF file by converting pages to images and running OCR on each page"""
        start_time = time.time()
        
        try:
            # Convert PDF pages to images with better quality settings
            images = convert_from_bytes(pdf_bytes, dpi=300, fmt='PNG', thread_count=2)
            
            page_results = []
            
            for page_num, image in enumerate(images, 1):
                # Convert PIL image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                
                # Process the page image with OCR
                page_start_time = time.time()
                
                # Run OCR on this page with dynamic document type detection per page
                engine_results = await self.extract_with_multiple_engines(
                    img_bytes, document_type, enable_ai
                )
                
                # Process consensus for this page
                from services.consensus_processor import ConsensusProcessor
                consensus_processor = ConsensusProcessor()
                consensus_result = await consensus_processor.process_consensus(
                    engine_results, threshold=0.7  # Lower threshold for PDF processing
                )
                
                # Classify document type for this specific page if unknown
                from services.document_classifier import DocumentClassifier
                classifier = DocumentClassifier()
                if document_type == "unknown":
                    # Get text from OCR to classify this page
                    temp_img = Image.open(io.BytesIO(img_bytes))
                    classified_type = await classifier.classify(img_bytes)
                    page_doc_type = classified_type.type
                else:
                    page_doc_type = document_type
                
                page_processing_time = time.time() - page_start_time
                
                page_result = {
                    "page_number": page_num,
                    "document_type": page_doc_type,
                    "confidence": consensus_result.confidence,
                    "extracted_text": consensus_result.text,
                    "structured_data": consensus_result.structured_data,
                    "processing_time": page_processing_time,
                    "engines_used": consensus_result.engines_used,
                    "requires_human_verification": consensus_result.confidence < 0.7
                }
                
                page_results.append(page_result)
            
            return page_results
            
        except Exception as e:
            print(f"PDF processing error: {e}")
            import traceback
            traceback.print_exc()
            return []