import difflib
from typing import List, Dict, Any, Optional
from collections import defaultdict
import asyncio
import time

from models.response_models import EngineResult, ConsensusResult

class ConsensusProcessor:
    def __init__(self):
        self.engine_weights = {
            "trocr": 0.4,      # AI-powered, highest weight
            "paddle_ocr": 0.35, # Good accuracy
            "tesseract": 0.25   # Baseline, lower weight
        }
    
    async def process_consensus(
        self, 
        engine_results: List[EngineResult], 
        threshold: float = 0.8
    ) -> ConsensusResult:
        """Process multiple OCR results using consensus algorithm"""
        
        if not engine_results:
            return ConsensusResult(
                text="",
                confidence=0.0,
                engines_used=[],
                structured_data={},
                processing_time=0.0
            )
        
        start_time = time.time()
        
        # Filter valid results
        valid_results = [r for r in engine_results if r.text.strip()]
        
        if not valid_results:
            return ConsensusResult(
                text="",
                confidence=0.0,
                engines_used=[r.engine_name for r in engine_results],
                structured_data={},
                processing_time=time.time() - start_time
            )
        
        # Find consensus text
        consensus_text = await self._find_text_consensus(valid_results)
        
        # Calculate confidence based on agreement
        confidence = self._calculate_confidence(valid_results, consensus_text)
        
        # Merge structured data
        merged_structured_data = await self._merge_structured_data(valid_results)
        
        # Apply confidence threshold
        final_confidence = confidence if confidence >= threshold else confidence
        
        return ConsensusResult(
            text=consensus_text,
            confidence=final_confidence,
            engines_used=[r.engine_name for r in valid_results],
            structured_data=merged_structured_data,
            processing_time=time.time() - start_time
        )
    
    async def _find_text_consensus(self, results: List[EngineResult]) -> str:
        """Find consensus text using multiple strategies"""
        
        if len(results) == 1:
            return results[0].text
        
        # Strategy 1: Exact match
        exact_match = self._check_exact_match(results)
        if exact_match:
            return exact_match
        
        # Strategy 2: Weighted similarity
        best_text = await self._find_best_by_similarity(results)
        if best_text:
            return best_text
        
        # Strategy 3: Merge text parts
        merged_text = await self._merge_text_intelligently(results)
        return merged_text
    
    def _check_exact_match(self, results: List[EngineResult]) -> Optional[str]:
        """Check if any texts are exactly the same"""
        text_counts = defaultdict(int)
        text_weights = defaultdict(float)
        
        for result in results:
            text = result.text.strip()
            text_counts[text] += 1
            text_weights[text] += self.engine_weights.get(result.engine_name, 0.25)
        
        # If any text appears more than once, return it
        for text, count in text_counts.items():
            if count > 1:
                return text
        
        # If no exact matches, return highest weighted text
        if text_weights:
            best_text = max(text_weights.items(), key=lambda x: x[1])[0]
            if text_weights[best_text] > 0.4:  # Reasonable confidence
                return best_text
        
        return None
    
    async def _find_best_by_similarity(self, results: List[EngineResult]) -> Optional[str]:
        """Find best text based on weighted similarity"""
        
        if len(results) < 2:
            return results[0].text if results else ""
        
        # Calculate weighted similarity scores
        similarity_scores = []
        
        for i, result1 in enumerate(results):
            total_similarity = 0
            total_weight = 0
            
            for j, result2 in enumerate(results):
                if i != j:
                    similarity = difflib.SequenceMatcher(None, result1.text, result2.text).ratio()
                    weight = self.engine_weights.get(result2.engine_name, 0.25)
                    total_similarity += similarity * weight
                    total_weight += weight
            
            if total_weight > 0:
                avg_similarity = total_similarity / total_weight
                engine_weight = self.engine_weights.get(result1.engine_name, 0.25)
                final_score = (avg_similarity * 0.7) + (engine_weight * 0.3)
                
                similarity_scores.append((result1.text, final_similarity, result1.confidence))
        
        if similarity_scores:
            # Sort by similarity score, then by engine confidence
            best_text = max(similarity_scores, key=lambda x: (x[1], x[2]))
            if best_text[1] > 0.6:  # Minimum similarity threshold
                return best_text[0]
        
        return None
    
    async def _merge_text_intelligently(self, results: List[EngineResult]) -> str:
        """Intelligently merge text from different engines"""
        
        # Sort results by engine weight and confidence
        sorted_results = sorted(
            results, 
            key=lambda x: (self.engine_weights.get(x.engine_name, 0.25), x.confidence),
            reverse=True
        )
        
        primary_text = sorted_results[0].text
        secondary_texts = sorted_results[1:]
        
        # Use primary text as base
        merged_lines = primary_text.split('\n')
        
        # Enhance with information from other engines
        for secondary in secondary_texts:
            secondary_lines = secondary.text.split('\n')
            merged_lines = await self._merge_lines(merged_lines, secondary_lines)
        
        return '\n'.join(filter(None, merged_lines))
    
    async def _merge_lines(self, primary_lines: List[str], secondary_lines: List[str]) -> List[str]:
        """Merge line-by-line intelligently"""
        merged = primary_lines.copy()
        
        for sec_line in secondary_lines:
            sec_line = sec_line.strip()
            if not sec_line:
                continue
            
            # Check if similar line exists in primary
            found_similar = False
            for i, prim_line in enumerate(merged):
                if difflib.SequenceMatcher(None, prim_line, sec_line).ratio() > 0.8:
                    # Keep the longer/better version
                    if len(sec_line) > len(prim_line):
                        merged[i] = sec_line
                    found_similar = True
                    break
            
            # Add if unique
            if not found_similar:
                merged.append(sec_line)
        
        return merged
    
    def _calculate_confidence(self, results: List[EngineResult], consensus_text: str) -> float:
        """Calculate overall confidence based on agreement"""
        
        if not results:
            return 0.0
        
        # Calculate text similarity for each engine result
        similarities = []
        weights = []
        
        for result in results:
            similarity = difflib.SequenceMatcher(None, result.text, consensus_text).ratio()
            weight = self.engine_weights.get(result.engine_name, 0.25)
            
            similarities.append(similarity)
            weights.append(weight)
        
        # Calculate weighted average similarity
        if weights:
            weighted_similarity = sum(s * w for s, w in zip(similarities, weights)) / sum(weights)
        else:
            weighted_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        # Factor in individual engine confidences
        avg_engine_confidence = sum(r.confidence for r in results) / len(results)
        
        # Combine both metrics
        final_confidence = (weighted_similarity * 0.7) + (avg_engine_confidence * 0.3)
        
        return min(final_confidence, 1.0)
    
    async def _merge_structured_data(self, results: List[EngineResult]) -> Dict[str, Any]:
        """Merge structured data from multiple engines"""
        
        if not results:
            return {}
        
        # Collect all structured data
        all_data = {}
        field_votes = defaultdict(list)
        field_values = defaultdict(list)
        
        for result in results:
            if result.structured_data:
                for field, value in result.structured_data.items():
                    if value:  # Only consider non-empty values
                        field_votes[field].append(result.engine_name)
                        field_values[field].append(value)
        
        # Apply voting for each field
        merged_data = {}
        
        for field, values in field_values.items():
            if not values:
                continue
            
            # If all engines agree, use that value
            if len(values) == 1:
                merged_data[field] = values[0]
            elif len(set(values)) == 1:
                # All same values
                merged_data[field] = values[0]
            else:
                # Different values - apply weighted voting
                value_scores = defaultdict(float)
                
                for i, value in enumerate(values):
                    engine_name = field_votes[field][i]
                    weight = self.engine_weights.get(engine_name, 0.25)
                    value_scores[value] += weight
                
                # Choose value with highest score
                best_value = max(value_scores.items(), key=lambda x: x[1])[0]
                merged_data[field] = best_value
        
        return merged_data
    
    async def get_confidence_breakdown(self, results: List[EngineResult]) -> Dict[str, float]:
        """Get detailed confidence breakdown for debugging"""
        
        breakdown = {}
        
        for result in results:
            breakdown[result.engine_name] = {
                'confidence': result.confidence,
                'weight': self.engine_weights.get(result.engine_name, 0.25),
                'weighted_confidence': result.confidence * self.engine_weights.get(result.engine_name, 0.25)
            }
        
        return breakdown