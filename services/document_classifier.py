# Conditionally import transformers (only if available)
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

import numpy as np
from PIL import Image
import io
from typing import Dict, Any
import asyncio

from models.response_models import DocumentType

class DocumentClassifier:
    def __init__(self):
        self.classifier = None
        self.document_types = {
            "payslip": [
                "payslip", "salary", "pay", "employee", "income", "net pay", "gross pay", 
                "deduction", "tax", "basic salary", "allowance", "benefits", "employer", 
                "employee no", "position", "department", "pay period", "hours worked",
                # Zambian-specific keywords
                "e-payslip", "government of the republic of zambia", "sal/scale",
                "taxable income", "leave accrued", "napsa", "pension life", "bayport",
                "accumulation", "payment amount", "deduction amount", "paye"
            ],
            "id_document": [
                "id", "identification", "national", "passport", "driver's license", "license", 
                "birth certificate", "surname", "given name", "address", "dob", "date of birth", 
                "issue date", "expiry date", "issuing authority", "photo", "signature"
            ],
            "contract": [
                "contract", "agreement", "terms", "signature", "party", "effective date", 
                "termination", "compensation", "obligation", "clause", "article", "signatory", 
                "witness", "consideration", "mutual", "binding", "performance"
            ],
            "invoice": [
                "invoice", "bill", "amount", "due", "payment", "invoice no", "invoice number", 
                "customer", "vendor", "item", "quantity", "rate", "subtotal", "total", 
                "tax", "vat", "gst", "due date", "reference", "po number", "purchase order"
            ],
            "bank_statement": [
                "bank", "statement", "account", "balance", "transaction", "debit", "credit", 
                "date", "description", "opening balance", "closing balance", "period", 
                "branch", "account holder", "acc no", "reference", "cheque", "withdrawal", "deposit"
            ],
            "receipt": [
                "receipt", "proof", "payment", "cash", "received", "paid", "amount", "for", 
                "by", "on", "acknowledged", "sold", "bought", "purchased", "transaction", 
                "ref no", "reference", "thank you", "store", "business", "customer copy"
            ],
            "certificate": [
                "certificate", "award", "achievement", "completion", "certified", "presented", 
                "granted", "degree", "diploma", "qualification", "conferred", "awarded", 
                "participant", "course", "training", "study", "excellence", "merit"
            ],
            "application_form": [
                "application", "form", "apply", "request", "personal details", "contact", 
                "address", "phone", "email", "submit", "required", "fill", "complete", 
                "signature", "date", "attachments", "supporting documents", "fee"
            ],
            "medical_record": [
                "medical", "record", "patient", "doctor", "hospital", "clinic", "treatment", 
                "diagnosis", "prescription", "medication", "appointment", "consultation", 
                "lab", "test", "report", "history", "symptoms", "condition", "medicine"
            ],
            "academic_transcript": [
                "transcript", "academic", "grade", "course", "credit", "gpa", "semester", 
                "year", "institution", "student", "number", "major", "minor", "degree", 
                "graduation", "cumulative", "standing", "credits", "subject", "marks"
            ]
        }
    
    async def initialize(self):
        """Initialize AI model asynchronously"""
        if self.classifier is None:
            # Use image classification model only if transformers is available
            if pipeline is not None:
                try:
                    self.classifier = pipeline(
                        "image-classification",
                        model="google/vit-base-patch16-224"
                    )
                except Exception as e:
                    print(f"AI classifier initialization failed: {e}")
                    self.classifier = None
            else:
                print("Transformers not available, skipping AI classifier initialization")
                self.classifier = None
    
    async def classify(self, image_bytes: bytes) -> DocumentType:
        """Classify document type using AI vision model"""
        await self.initialize()
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Get AI classification
            ai_result = await self._classify_with_ai(image)
            
            # Get keyword-based classification as fallback
            keyword_result = await self._classify_with_keywords(image_bytes)
            
            # Combine results with weighted confidence
            final_type, final_confidence = self._combine_results(
                ai_result, keyword_result
            )
            
            return DocumentType(
                type=final_type,
                confidence=final_confidence,
                description=self._get_type_description(final_type)
            )
            
        except Exception as e:
            print(f"Classification error: {e}")
            # Fallback to keyword classification
            return await self._classify_with_keywords(image_bytes)
    
    async def _classify_with_ai(self, image: Image.Image) -> tuple:
        """Use AI model for classification with document-specific logic"""
        # Check if AI classifier is available
        if self.classifier is None or pipeline is None:
            print("AI classifier not available, using keyword-based classification")
            # Return a fallback that will trigger keyword classification
            return "unknown", 0.0
        
        try:
            # Preprocess image for better classification
            processed_image = self._preprocess_image_for_classification(image)
            
            # Use the Vision Transformer model for classification
            results = self.classifier(processed_image)
            
            # Analyze the top results to determine document type
            doc_type = "unknown"
            confidence = 0.0
            
            # Look at top 3 predictions to make a more informed decision
            top_results = results[:3] if len(results) >= 3 else results
            
            for result in top_results:
                label = result['label'].lower()
                score = result['score']
                
                # Check if this label maps to a known document type
                mapped_type = self._map_label_to_document_type(label)
                if mapped_type != "unknown":
                    doc_type = mapped_type
                    confidence = score
                    break
                elif score > confidence:
                    # If no direct mapping, use the highest confidence as fallback
                    confidence = score
            
            # If we couldn't map to a specific type, use text analysis as backup
            if doc_type == "unknown" and confidence < 0.7:
                # Convert image to text and analyze
                import pytesseract
                import numpy as np
                from PIL import Image
                import io
                
                # Convert PIL image to grayscale using PIL
                if image.mode != 'L':
                    img_gray = image.convert('L')
                else:
                    img_gray = image
                
                # Extract text
                text = pytesseract.image_to_string(img_gray).lower()
                
                # Determine type based on text content
                for doc_type_key, keywords in self.document_types.items():
                    keyword_count = sum(1 for keyword in keywords if keyword in text)
                    if keyword_count > 0:
                        doc_type = doc_type_key
                        confidence = min(0.6, confidence + 0.1 * keyword_count)  # Moderate confidence
                        break
            
            return doc_type, confidence
            
        except Exception as e:
            print(f"AI classification failed: {e}")
            return "unknown", 0.0
    
    def _preprocess_image_for_classification(self, image: Image.Image) -> Image.Image:
        """Preprocess image specifically for AI classification"""
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to the expected input size for ViT model (typically 224x224)
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        return image
    
    async def _classify_with_keywords(self, image_bytes: bytes) -> DocumentType:
        """Classify using OCR + keyword analysis with improved scoring"""
        try:
            import pytesseract
            
            # Convert image bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to grayscale for OCR
            if pil_image.mode != 'L':
                pil_image = pil_image.convert('L')
            
            # Extract text for keyword analysis
            text = pytesseract.image_to_string(pil_image).lower()
            
            # Score each document type based on keyword matches with weighted scoring
            scores = {}
            for doc_type, keywords in self.document_types.items():
                # Count exact matches
                exact_matches = sum(1 for keyword in keywords if keyword in text)
                
                # Count partial matches (for variations of terms)
                partial_matches = 0
                for keyword in keywords:
                    # Check for variations like plural forms or different endings
                    variations = [keyword, keyword + 's', keyword + 'es', keyword + 'ed', keyword + 'ing']
                    for var in variations:
                        if var in text and var != keyword:  # Avoid double counting
                            partial_matches += 0.5  # Lower weight for partial matches
                
                # Calculate weighted score
                total_keywords = len(keywords)
                score = (exact_matches + partial_matches) / total_keywords if total_keywords > 0 else 0
                
                # Boost score if key terms are found
                key_terms = ['payslip', 'invoice', 'contract', 'receipt', 'statement', 'certificate', 'id', 'passport']
                if any(term in text for term in key_terms):
                    score *= 1.2  # Boost if key terms are found
                
                # Penalize if conflicting terms are found
                conflicting_terms = ['advertisement', 'notice', 'notification', 'announcement']
                if any(term in text for term in conflicting_terms):
                    score *= 0.5  # Reduce score if conflicting terms found
                
                scores[doc_type] = min(score, 1.0)  # Cap at 1.0
            
            # Get the best match
            if scores:
                best_type = max(scores, key=scores.get)
                confidence = scores[best_type]
                
                # Adjust confidence based on the gap between first and second best
                sorted_scores = sorted(scores.values(), reverse=True)
                if len(sorted_scores) > 1: 
                    gap = sorted_scores[0] - sorted_scores[1]
                    # Increase confidence if there's a clear winner
                    if gap > 0.1:
                        confidence = min(confidence + 0.1, 0.95)
                
                return DocumentType(
                    type=best_type if confidence > 0.05 else "unknown",  # Lower threshold
                    confidence=min(confidence, 0.95),  # Cap at 95%
                    description=self._get_type_description(best_type)
                )
            else:
                return DocumentType(
                    type="unknown",
                    confidence=0.0,
                    description="Unable to classify document"
                )
            
        except Exception as e:
            print(f"Keyword classification failed: {e}")
            return DocumentType(
                type="unknown",
                confidence=0.0,
                description="Unable to classify document"
            )
    
    def _map_label_to_document_type(self, label: str) -> str:
        """Map AI model labels to document types"""
        mapping = {
            "document": "unknown",
            "text": "unknown",
            "receipt": "receipt",
            "invoice": "invoice",
            "form": "application_form",
            "certificate": "certificate",
            "id": "id_document"
        }
        return mapping.get(label, "unknown")
    
    def _combine_results(self, ai_result: tuple, keyword_result: DocumentType) -> tuple:
        """Combine AI and keyword results"""
        ai_type, ai_confidence = ai_result
        
        # Weight AI higher if confidence is good
        if ai_confidence > 0.7:
            return ai_type, ai_confidence
        
        # Otherwise prefer keyword result
        return keyword_result.type, keyword_result.confidence
    
    def _get_type_description(self, doc_type: str) -> str:
        """Get human-readable description"""
        descriptions = {
            "payslip": "Employee salary payment document",
            "id_document": "Government-issued identification",
            "contract": "Legal agreement document",
            "invoice": "Commercial invoice for goods/services",
            "bank_statement": "Bank account statement",
            "receipt": "Proof of payment document",
            "certificate": "Award or completion certificate",
            "application_form": "Form for applications/requests",
            "unknown": "Unrecognized document type"
        }
        return descriptions.get(doc_type, "Unknown document type")