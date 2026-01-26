import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
# Conditionally import paddleocr and transformers (only if available)
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

try:
    from transformers import pipeline
    import torch
except ImportError:
    pipeline = None
    torch = None

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
        if self.paddle_ocr is None and PaddleOCR is not None:
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
        elif PaddleOCR is None:
            # PaddleOCR module is not available
            self.paddle_ocr = None
        
        # Initialize TrOCR with better error handling and options
        if self.trocr_pipeline is None and pipeline is not None and torch is not None:
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
        elif pipeline is None or torch is None:
            # Transformers or torch modules are not available
            self.trocr_pipeline = None
        
        # Initialize additional AI models for better recognition (only if transformers is available)
        if pipeline is not None:
            try:
                # Attempt to initialize layoutlm for document understanding
                import transformers
                from transformers import AutoTokenizer, VisionEncoderDecoderModel, AutoFeatureExtractor
                # Use a lightweight model for layout analysis
                self.layout_model = None  # Placeholder for future implementation
            except ImportError:
                self.layout_model = None
        else:
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
        if self.paddle_ocr and PaddleOCR is not None:
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
        if enable_ai and self.trocr_pipeline and pipeline is not None:
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
        
        # Check if PaddleOCR is available
        if self.paddle_ocr is None or PaddleOCR is None:
            print("PaddleOCR is not available")
            return EngineResult(
                engine_name="paddle_ocr",
                text="",
                confidence=0.0,
                processing_time=0.0,
                structured_data={}
            )
        
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
        
        # Check if TrOCR is available
        if self.trocr_pipeline is None or pipeline is None or torch is None:
            print("TrOCR is not available")
            return EngineResult(
                engine_name="trocr",
                text="",
                confidence=0.0,
                processing_time=0.0,
                structured_data={}
            )
        
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
            # For unknown types, try to detect payslip from text content
            # This helps when classification fails but text clearly indicates a payslip
            text_lower = text.lower()
            payslip_keywords = [
                'net pay', 'gross pay', 'basic salary', 'payslip', 'employee no',
                'paye', 'napsa', 'deduction', 'allowance', 'e-payslip',
                'government of the republic of zambia', 'zambia army', 'zambia development',
                'total earnings', 'total deductions', 'basic pay', 'transport allowance',
                'housing allowance', 'lunch allowance', 'emp.', 'employee'
            ]
            
            # Also check for corrupted OCR versions of keywords
            corrupted_keywords = [
                'tote', 'tota', 'tota1', 't0tal',  # "total" OCR errors
                'neh', 'net', 'n3t',  # "net" OCR errors
                'pay', 'p4y', 'p@y',  # "pay" OCR errors
                'emp', '3mp',  # "emp" OCR errors
            ]
            
            # Count keyword matches (exact and corrupted)
            keyword_count = sum(1 for keyword in payslip_keywords if keyword in text_lower)
            corrupted_count = sum(1 for keyword in corrupted_keywords if keyword in text_lower)
            
            # If we find multiple payslip keywords (including corrupted ones), try payslip extraction
            total_keyword_count = keyword_count + (corrupted_count // 2)  # Weight corrupted matches less
            if total_keyword_count >= 1:  # Lower threshold - even 1 keyword suggests payslip
                structured_data = await self._extract_payslip_data(text)
                if not structured_data or 'net_pay' not in structured_data:
                    # If extraction didn't find net_pay, still keep raw_text as fallback
                    structured_data = structured_data or {}
                    structured_data['raw_text'] = text
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
            # Enhanced patterns with flexible amount formats
            net_pay_patterns = [
                r'(?i)net\s+(?:pay|salary|earnings|amount)[\s:]*[:\s\$€£¥K]*([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',
                r'(?i)take\s+home\s+(?:pay|amount)[:\s\$€£¥K]*([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',
                r'(?i)net\s+(?:amount|total)[:\s\$€£¥K]*([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',
                r'(?i)(?:total\s+)?net[:\s\$€£¥K]+([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',
                r'(?i)net\s+([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',  # Simple "NET amount"
                r'(?i)\bnet\b\s+([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',  # Word boundary NET
                r'(?i)payable\s*[:\-]?\s*([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',  # "PAYABLE: amount"
                r'(?i)amount\s+payable\s*[:\-]?\s*([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',
                r'(?i)net\s+pay\s*[:\-]?\s*([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',  # NET PAY: amount
            ]
            for pattern in net_pay_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in matches:
                    amount = match.group(1).strip()
                    # Clean amount for validation
                    cleaned = amount.replace(',', '').replace('.', '', 1) if '.' in amount else amount.replace(',', '')
                    if self._is_valid_amount(cleaned):
                        data['net_pay'] = self._normalize_amount(amount)
                        break
                if 'net_pay' in data:
                    break
        
        # Ultimate fallback: Use positional heuristics to find net pay
        # When OCR quality is poor and patterns don't match, use document structure
        if 'net_pay' not in data:
            net_pay = self._extract_netpay_positional_heuristics(text, lines)
            if net_pay:
                data['net_pay'] = net_pay
        
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
        Enhanced with multiple fallback strategies for different payslip formats.
        """
        import re
        
        # Enhanced amount pattern - handles various formats:
        # - 3,990.00 (with comma, 2 decimals)
        # - 3990.00 (no comma, 2 decimals)
        # - 3,990 (with comma, no decimals)
        # - 8689.21 (no comma, 2 decimals)
        # - 10,222.88 (with comma, 2 decimals)
        amount_patterns = [
            r'([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',  # Flexible: 3,990.00, 3990.00, 10,222.88
            r'([0-9,]+\.[0-9]{1,2})',  # Standard: comma-separated with decimals
            r'([0-9]+\.[0-9]{1,2})',  # No comma: 3990.00
            r'([0-9]{1,3}(?:[,\.][0-9]{3})+)',  # Large numbers: 10,222 or 10222
        ]
        
        def extract_amount_from_text(text_snippet: str) -> Optional[str]:
            """Extract the first valid amount from text snippet"""
            for pattern in amount_patterns:
                matches = re.findall(pattern, text_snippet)
                for match in matches:
                    if not match:
                        continue
                    # Clean the match - handle both comma and period as thousands separator
                    # First, determine if period is decimal or thousands separator
                    if '.' in match and ',' in match:
                        # Format like 3,990.00 - comma is thousands, period is decimal
                        cleaned = match.replace(',', '')
                    elif '.' in match:
                        # Could be 3990.00 or 3.990 (European format)
                        # Check if it looks like decimal (has 1-2 digits after period)
                        parts = match.split('.')
                        if len(parts) == 2 and len(parts[1]) <= 2:
                            # Decimal format: 3990.00
                            cleaned = match.replace(',', '')
                        else:
                            # Thousands separator: 3.990
                            cleaned = match.replace('.', '').replace(',', '')
                    elif ',' in match:
                        # Could be 3,990 (thousands) or 3,99 (decimal in some locales)
                        # For payslips, comma is usually thousands separator
                        cleaned = match.replace(',', '')
                    else:
                        cleaned = match
                    
                    # Try to validate
                    try:
                        test_val = float(cleaned)
                        if self._is_valid_amount(cleaned):
                            # Return in original format if it has proper decimal places
                            if '.' in match and len(match.split('.')[-1]) <= 2:
                                return match
                            elif '.' not in match and len(cleaned) > 2:
                                # Add decimal places
                                return f"{int(test_val)}.{int((test_val - int(test_val)) * 100):02d}"
                            else:
                                return match
                    except (ValueError, AttributeError):
                        continue
            return None
        
        # Method 1: Line-by-line detection for "NET PAY" followed by amount
        for i, line in enumerate(lines):
            line_clean = line.strip().upper()
            line_original = line.strip()
            
            # Check if this line contains "NET PAY" in various forms
            if any(keyword in line_clean for keyword in ['NET PAY', 'NET-PAY', 'NETPAY', 'NET PAY:', 'NET-PAY:']):
                # Try to extract amount from the same line first
                amount = extract_amount_from_text(line_original)
                if amount:
                    return self._normalize_amount(amount)
                
                # Check the next 2 lines for the amount (sometimes it's 2 lines away)
                for offset in [1, 2]:
                    if i + offset < len(lines):
                        next_line = lines[i + offset].strip()
                        amount = extract_amount_from_text(next_line)
                        if amount:
                            return self._normalize_amount(amount)
        
        # Method 2: Handle "NET" without "PAY" (e.g., ZAMBIA ARMY format: "NET 8,689.21")
        for i, line in enumerate(lines):
            line_upper = line.strip().upper()
            line_original = line.strip()
            
            # Check for "NET" followed by amount on same line (common in army payslips)
            net_match = re.search(r'(?i)\bNET\b', line_upper)
            if net_match and 'PAY' not in line_upper:
                # Extract amount from same line after "NET"
                net_pos = net_match.end()
                text_after_net = line_original[net_pos:].strip()
                amount = extract_amount_from_text(text_after_net)
                if amount:
                    return self._normalize_amount(amount)
                
                # Also check if standalone "NET" on its own line
                if re.match(r'^NET\s*$', line_upper):
                    # Check next 2 lines
                    for offset in [1, 2]:
                        if i + offset < len(lines):
                            next_line = lines[i + offset].strip()
                            amount = extract_amount_from_text(next_line)
                            if amount:
                                return self._normalize_amount(amount)
        
        # Method 3: Enhanced pattern matching for multi-line format
        multiline_patterns = [
            r'(?i)NET\s+PAY\s*[:\-]?\s*([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',  # NET PAY: 3,990.00
            r'(?i)NET\s+PAY\s*\n+\s*([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',  # NET PAY\n3,990.00
            r'(?i)NET\s+PAY\s+[^\d]*([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',  # NET PAY ... 3,990.00
            r'(?i)NETPAY\s*[:\-]?\s*([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',  # NETPAY: 3,990.00
            r'(?i)NET\s*\n+\s*([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',  # NET\n3,990.00
            r'(?i)NET\s+([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',  # NET 3,990.00 (same line)
            r'(?i)\bNET\b\s+([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',  # NET 8,689.21
        ]
        
        for pattern in multiline_patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))
            for match in matches:
                amount_str = match.group(1)
                # Clean and validate
                cleaned = amount_str.replace(',', '').replace('.', '', 1) if '.' in amount_str else amount_str.replace(',', '')
                if self._is_valid_amount(cleaned):
                    return self._normalize_amount(amount_str)
        
        # Method 4: Table footer detection - look for TOTALS row and NET PAY
        # Zambian payslips often have: TOTALS <payment_amount> <deduction_amount> then NET PAY <amount>
        totals_patterns = [
            r'(?i)TOTALS?\s+[^\d]*([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)\s+[^\d]*([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)\s*\n\s*NET\s+PAY\s+[^\d]*([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',
            r'(?i)TOTALS?\s+[^\d]*([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)\s*\n\s*NET\s+PAY\s+[^\d]*([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',
        ]
        
        for pattern in totals_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                # Get the last group (net pay amount)
                amount_str = match.group(match.lastindex)
                cleaned = amount_str.replace(',', '').replace('.', '', 1) if '.' in amount_str else amount_str.replace(',', '')
                if self._is_valid_amount(cleaned):
                    return self._normalize_amount(amount_str)
        
        # Method 5: Find amounts near "NET PAY" keywords (expanded search window)
        net_pay_keywords = ['NET PAY', 'NET-PAY', 'NETPAY', 'NET']
        for keyword in net_pay_keywords:
            # Search for keyword and extract context (up to 150 chars after)
            pattern = rf'(?i)({re.escape(keyword)}.{{0,150}})'
            matches = list(re.finditer(pattern, text, re.DOTALL | re.IGNORECASE))
            for match in matches:
                section = match.group(1)
                # Try to find amount in this section
                amount = extract_amount_from_text(section)
                if amount:
                    return self._normalize_amount(amount)
        
        # Method 6: Fallback - Find largest amount in bottom 30% of document (where net pay usually appears)
        # This helps when OCR misses the "NET PAY" label
        if len(lines) > 4:
            bottom_percent = max(5, len(lines) // 3)  # Bottom 33% or at least 5 lines
            bottom_lines = lines[-bottom_percent:]
            bottom_text = '\n'.join(bottom_lines)
            
            # Extract all amounts from bottom section
            all_amounts = []
            for pattern in amount_patterns:
                matches = re.findall(pattern, bottom_text)
                for match in matches:
                    cleaned = match.replace(',', '').replace('.', '', 1) if '.' in match else match.replace(',', '')
                    if self._is_valid_amount(cleaned):
                        try:
                            value = float(cleaned)
                            all_amounts.append((value, match))
                        except:
                            continue
            
            if all_amounts:
                # Sort by value and return the largest (net pay is usually the largest in bottom section)
                all_amounts.sort(key=lambda x: x[0], reverse=True)
                return self._normalize_amount(all_amounts[0][1])
        
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
    
    def _extract_netpay_positional_heuristics(self, text: str, lines: List[str]) -> Optional[str]:
        """
        Use positional heuristics to extract net pay when OCR quality is poor.
        Net pay is typically the largest amount in the bottom section of the document.
        """
        import re
        
        # Enhanced amount patterns to handle poor OCR quality
        amount_patterns = [
            r'([0-9]{1,3}(?:[,\.][0-9]{3})*(?:\.[0-9]{1,2})?)',  # Standard format
            r'([0-9,]+\.[0-9]{1,2})',  # With comma thousands
            r'([0-9]+\.[0-9]{1,2})',  # Without comma
            r'([0-9]{3,}(?:\.[0-9]{1,2})?)',  # Large numbers without separators
        ]
        
        # Strategy 1: Find all monetary amounts in the text using multiple patterns
        all_amounts = []
        for pattern in amount_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match:
                    cleaned = match.replace(',', '').replace('.', '', 1) if '.' in match else match.replace(',', '')
                    if self._is_valid_amount(cleaned):
                        try:
                            value = float(cleaned)
                            all_amounts.append((value, match))
                        except:
                            continue
        
        if not all_amounts:
            return None
        
        # Strategy 2: Look for amounts near "TOTAL" keywords (even with OCR errors)
        # Net pay often appears after "Total Deductions" or "Total Earnings"
        total_keywords = ['total', 'tote', 'tota', 'tota1', 't0tal', 't0tals']  # Common OCR errors
        for keyword in total_keywords:
            # Find positions of total keywords
            pattern = rf'(?i){re.escape(keyword)}'
            for match in re.finditer(pattern, text):
                # Look for amounts in the 200 characters after "total"
                context = text[match.end():match.end()+200]
                context_amounts = []
                for amt_pattern in amount_patterns:
                    amt_matches = re.findall(amt_pattern, context)
                    for amt_match in amt_matches:
                        cleaned = amt_match.replace(',', '').replace('.', '', 1) if '.' in amt_match else amt_match.replace(',', '')
                        if self._is_valid_amount(cleaned):
                            try:
                                value = float(cleaned)
                                context_amounts.append((value, amt_match))
                            except:
                                continue
                
                if context_amounts:
                    # Return the first reasonable amount after "total"
                    context_amounts.sort(key=lambda x: x[0], reverse=True)
                    return self._normalize_amount(context_amounts[0][1])
        
        # Strategy 3: Find the amount that appears in the bottom section
        # Net pay is usually in the bottom 30% of the document
        if len(lines) > 4:
            bottom_percent = max(5, len(lines) // 3)
            bottom_lines = lines[-bottom_percent:]
            bottom_text = '\n'.join(bottom_lines)
            
            bottom_amounts = []
            for pattern in amount_patterns:
                matches = re.findall(pattern, bottom_text)
                for match in matches:
                    cleaned = match.replace(',', '').replace('.', '', 1) if '.' in match else match.replace(',', '')
                    if self._is_valid_amount(cleaned):
                        try:
                            value = float(cleaned)
                            bottom_amounts.append((value, match))
                        except:
                            continue
            
            if bottom_amounts:
                # Return the largest amount in the bottom section
                bottom_amounts.sort(key=lambda x: x[0], reverse=True)
                return self._normalize_amount(bottom_amounts[0][1])
        
        # Strategy 4: Return a reasonable amount from all amounts
        # Filter out extremely large amounts that might be totals or sums
        all_amounts.sort(key=lambda x: x[0], reverse=True)
        
        if not all_amounts:
            return None
        
        # Use median-based filtering to find reasonable net pay
        values = [v for v, a in all_amounts]
        if len(values) > 1:
            median_value = sorted(values)[len(values) // 2]
            
            # Net pay should be within a reasonable range (not an extreme outlier)
            # Typically net pay is between 1,000 and 50,000 for Zambian payslips
            reasonable_amounts = [(v, a) for v, a in all_amounts if 1000 <= v <= 50000]
            if reasonable_amounts:
                # Return the largest reasonable amount
                reasonable_amounts.sort(key=lambda x: x[0], reverse=True)
                return self._normalize_amount(reasonable_amounts[0][1])
            
            # If no amounts in reasonable range, use median-based filtering
            if all_amounts[0][0] < median_value * 10:
                return self._normalize_amount(all_amounts[0][1])
            
            # Try second largest if first is too large
            if len(all_amounts) > 1 and all_amounts[1][0] < median_value * 10:
                return self._normalize_amount(all_amounts[1][1])
        else:
            # Only one amount, return it if valid
            return self._normalize_amount(all_amounts[0][1])
        
        return None
    
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
                try:
                    from services.consensus_processor import ConsensusProcessor
                    consensus_processor = ConsensusProcessor()
                    consensus_result = await consensus_processor.process_consensus(
                        engine_results, threshold=0.7  # Lower threshold for PDF processing
                    )
                except Exception as e:
                    print(f"Consensus processor error: {e}")
                    # Fallback: use the first engine result if consensus fails
                    if engine_results:
                        first_result = engine_results[0]
                        consensus_result = type('obj', (object,), {
                            'text': first_result.text,
                            'confidence': first_result.confidence,
                            'engines_used': [first_result.engine_name],
                            'structured_data': first_result.structured_data or {}
                        })()
                    else:
                        consensus_result = type('obj', (object,), {
                            'text': "",
                            'confidence': 0.0,
                            'engines_used': [],
                            'structured_data': {}
                        })()
                
                # Classify document type for this specific page if unknown
                if document_type == "unknown":
                    try:
                        from services.document_classifier import DocumentClassifier
                        classifier = DocumentClassifier()
                        # Get text from OCR to classify this page
                        temp_img = Image.open(io.BytesIO(img_bytes))
                        classified_type = await classifier.classify(img_bytes)
                        page_doc_type = classified_type.type
                    except Exception as e:
                        print(f"Document classifier error: {e}")
                        # Fallback to unknown if classification fails
                        page_doc_type = "unknown"
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