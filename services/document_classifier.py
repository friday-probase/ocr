from transformers import pipeline
import cv2
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
            "payslip": ["payslip", "salary", "pay", "employee", "income"],
            "id_document": ["id", "identification", "national", "passport"],
            "contract": ["contract", "agreement", "terms", "signature"],
            "invoice": ["invoice", "bill", "amount", "due", "payment"],
            "bank_statement": ["bank", "statement", "account", "balance", "transaction"],
            "receipt": ["receipt", "proof", "payment", "cash"],
            "certificate": ["certificate", "award", "achievement", "completion"],
            "application_form": ["application", "form", "apply", "request"]
        }
    
    async def initialize(self):
        """Initialize AI model asynchronously"""
        if self.classifier is None:
            # Use image classification model
            self.classifier = pipeline(
                "image-classification",
                model="google/vit-base-patch16-224"
            )
    
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
        """Use AI model for classification"""
        try:
            # This would need custom training for document types
            # For now, use generic classification as fallback
            results = self.classifier(image)
            
            # Map generic labels to document types
            top_label = results[0]['label'].lower()
            confidence = results[0]['score']
            
            doc_type = self._map_label_to_document_type(top_label)
            return doc_type, confidence
            
        except Exception as e:
            print(f"AI classification failed: {e}")
            return "unknown", 0.0
    
    async def _classify_with_keywords(self, image_bytes: bytes) -> DocumentType:
        """Classify using OCR + keyword analysis"""
        try:
            import pytesseract
            
            # Convert to numpy array for OpenCV
            nparr = np.frombuffer(image_bytes, np.uint8)
            image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Extract text for keyword analysis
            text = pytesseract.image_to_string(image_cv).lower()
            
            # Score each document type based on keyword matches
            scores = {}
            for doc_type, keywords in self.document_types.items():
                score = sum(1 for keyword in keywords if keyword in text)
                total_keywords = len(keywords)
                scores[doc_type] = score / total_keywords if total_keywords > 0 else 0
            
            # Get the best match
            best_type = max(scores, key=scores.get)
            confidence = scores[best_type]
            
            return DocumentType(
                type=best_type if confidence > 0.1 else "unknown",
                confidence=min(confidence * 2, 0.95),  # Boost confidence but cap at 95%
                description=self._get_type_description(best_type)
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