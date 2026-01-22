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
        """Process multiple OCR results using enhanced consensus algorithm for scanned documents"""
        
        if not engine_results:
            return ConsensusResult(
                text="",
                confidence=0.0,
                engines_used=[],
                structured_data={},
                processing_time=0.0
            )
        
        start_time = time.time()
        
        # Filter valid results and sort by confidence
        valid_results = [r for r in engine_results if r.text.strip()]
        valid_results.sort(key=lambda x: x.confidence, reverse=True)
        
        if not valid_results:
            return ConsensusResult(
                text="",
                confidence=0.0,
                engines_used=[r.engine_name for r in engine_results],
                structured_data={},
                processing_time=time.time() - start_time
            )
        
        # Enhanced consensus text finding
        consensus_text = await self._find_enhanced_consensus(valid_results)
        
        # Calculate confidence with improved algorithm
        confidence = self._calculate_enhanced_confidence(valid_results, consensus_text)
        
        # Merge structured data with voting
        merged_structured_data = await self._merge_structured_data_with_voting(valid_results)
        
        # Apply confidence threshold with adaptive adjustment
        final_confidence = self._apply_adaptive_threshold(confidence, threshold, valid_results)
        
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
    
    async def _find_enhanced_consensus(self, results: List[EngineResult]) -> str:
        """Enhanced consensus finding with better text merging"""
        
        if len(results) == 1:
            return results[0].text
        
        # Try exact match first
        exact_match = self._check_exact_match(results)
        if exact_match:
            return exact_match
        
        # Use weighted similarity with better scoring
        best_text = await self._find_best_by_enhanced_similarity(results)
        if best_text:
            return best_text
        
        # Intelligent text merging with conflict resolution
        merged_text = await self._merge_text_with_conflict_resolution(results)
        return merged_text
    
    async def _find_best_by_enhanced_similarity(self, results: List[EngineResult]) -> Optional[str]:
        """Enhanced similarity calculation for scanned documents"""
        
        if len(results) < 2:
            return results[0].text if results else ""
        
        # Calculate enhanced similarity scores
        similarity_scores = []
        
        for i, result1 in enumerate(results):
            total_similarity = 0
            total_weight = 0
            
            for j, result2 in enumerate(results):
                if i != j:
                    # Use multiple similarity metrics
                    similarity = self._calculate_multi_metric_similarity(result1.text, result2.text)
                    weight = self.engine_weights.get(result2.engine_name, 0.25)
                    total_similarity += similarity * weight
                    total_weight += weight
            
            if total_weight > 0:
                avg_similarity = total_similarity / total_weight
                engine_weight = self.engine_weights.get(result1.engine_name, 0.25)
                # Boost confidence for AI engines on scanned documents
                ai_boost = 0.1 if result1.engine_name in ['trocr'] else 0
                final_score = (avg_similarity * 0.7) + (engine_weight * 0.3) + ai_boost
                
                similarity_scores.append((result1.text, final_score, result1.confidence))
        
        if similarity_scores:
            # Sort by enhanced score, then by engine confidence
            best_text = max(similarity_scores, key=lambda x: (x[1], x[2]))
            if best_text[1] > 0.7:  # Higher threshold for scanned documents
                return best_text[0]
        
        return None
    
    def _calculate_multi_metric_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using multiple metrics"""
        if not text1 or not text2:
            return 0.0
        
        # Sequence matcher (primary)
        seq_similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        
        # Word-level similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        word_similarity = len(words1.intersection(words2)) / len(words1.union(words2)) if words1 or words2 else 0
        
        # Length similarity (scanned docs should be similar length)
        len_similarity = 1 - abs(len(text1) - len(text2)) / max(len(text1), len(text2), 1)
        
        # Weighted combination
        return (seq_similarity * 0.5) + (word_similarity * 0.3) + (len_similarity * 0.2)
    
    async def _merge_text_with_conflict_resolution(self, results: List[EngineResult]) -> str:
        """Intelligent text merging with conflict resolution"""
        
        # Sort results by enhanced confidence
        sorted_results = sorted(
            results, 
            key=lambda x: self._calculate_enhanced_engine_score(x),
            reverse=True
        )
        
        primary_text = sorted_results[0].text
        secondary_texts = sorted_results[1:]
        
        # Use primary text as base and enhance with others
        merged_lines = primary_text.split('\n')
        
        # Enhance with information from other engines
        for secondary in secondary_texts:
            secondary_lines = secondary.text.split('\n')
            merged_lines = await self._merge_lines_with_resolution(merged_lines, secondary_lines, secondary)
        
        return '\n'.join(filter(None, merged_lines))
    
    def _calculate_enhanced_engine_score(self, result: EngineResult) -> float:
        """Calculate enhanced score for engine results"""
        base_score = result.confidence * self.engine_weights.get(result.engine_name, 0.25)
        
        # Boost for AI engines on scanned documents
        if result.engine_name == 'trocr':
            base_score *= 1.2
        elif result.engine_name == 'paddle_ocr':
            base_score *= 1.1
        
        return base_score
    
    async def _merge_lines_with_resolution(self, primary_lines: List[str], secondary_lines: List[str], secondary_result: EngineResult) -> List[str]:
        """Merge lines with conflict resolution"""
        merged = primary_lines.copy()
        
        for sec_line in secondary_lines:
            sec_line = sec_line.strip()
            if not sec_line:
                continue
            
            # Check if similar line exists in primary
            found_similar = False
            for i, prim_line in enumerate(merged):
                similarity = difflib.SequenceMatcher(None, prim_line, sec_line).ratio()
                if similarity > 0.85:  # Very similar
                    found_similar = True
                    break
                elif similarity > 0.6:  # Somewhat similar - choose better version
                    if self._choose_better_line(prim_line, sec_line, secondary_result):
                        merged[i] = sec_line
                    found_similar = True
                    break
            
            # Add if unique and high confidence
            if not found_similar and secondary_result.confidence > 0.7:
                merged.append(sec_line)
        
        return merged
    
    def _choose_better_line(self, line1: str, line2: str, secondary_result: EngineResult) -> bool:
        """Choose which line is better based on various factors"""
        # Prefer longer lines (more complete)
        if abs(len(line1) - len(line2)) > 5:
            return len(line2) > len(line1)
        
        # Prefer lines with more alphanumeric characters
        alpha1 = sum(c.isalnum() for c in line1)
        alpha2 = sum(c.isalnum() for c in line2)
        if abs(alpha1 - alpha2) > 3:
            return alpha2 > alpha1
        
        # Prefer from higher confidence engine
        return secondary_result.confidence > 0.8
    
    def _calculate_enhanced_confidence(self, results: List[EngineResult], consensus_text: str) -> float:
        """Enhanced confidence calculation for scanned documents"""
        
        if not results:
            return 0.0
        
        # Calculate multi-metric similarities
        similarities = []
        weights = []
        
        for result in results:
            similarity = self._calculate_multi_metric_similarity(result.text, consensus_text)
            weight = self.engine_weights.get(result.engine_name, 0.25)
            
            # Boost similarity for AI engines
            if result.engine_name == 'trocr':
                similarity = min(similarity * 1.1, 1.0)
            
            similarities.append(similarity)
            weights.append(weight)
        
        # Weighted average similarity
        if weights:
            weighted_similarity = sum(s * w for s, w in zip(similarities, weights)) / sum(weights)
        else:
            weighted_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        # Factor in individual engine confidences
        avg_engine_confidence = sum(r.confidence for r in results) / len(results)
        
        # Enhanced combination with text quality factors
        text_quality = self._assess_text_quality(consensus_text)
        
        final_confidence = (weighted_similarity * 0.5) + (avg_engine_confidence * 0.3) + (text_quality * 0.2)
        
        return min(final_confidence, 1.0)
    
    def _assess_text_quality(self, text: str) -> float:
        """Assess the quality of extracted text"""
        if not text:
            return 0.0
        
        quality = 0.5  # Base quality
        
        # Length factor
        if len(text) > 100:
            quality += 0.2
        elif len(text) < 20:
            quality -= 0.2
        
        # Alphanumeric ratio
        alpha_ratio = sum(c.isalnum() for c in text) / len(text) if text else 0
        quality += (alpha_ratio - 0.5) * 0.4  # Boost for more alphanumeric content
        
        # Structure factor (presence of common document elements)
        if any(keyword in text.lower() for keyword in ['name', 'date', 'amount', 'number']):
            quality += 0.1
        
        return max(0.0, min(quality, 1.0))
    
    def _apply_adaptive_threshold(self, confidence: float, base_threshold: float, results: List[EngineResult]) -> float:
        """Apply adaptive threshold based on document characteristics"""
        
        # If we have high-confidence AI results, be more lenient
        ai_results = [r for r in results if r.engine_name == 'trocr' and r.confidence > 0.8]
        if ai_results:
            base_threshold = min(base_threshold, 0.85)
        
        # If all engines agree strongly, be more lenient
        if confidence > 0.9:
            base_threshold = min(base_threshold, 0.8)
        
        return confidence if confidence >= base_threshold else confidence
    
    async def _merge_structured_data_with_voting(self, results: List[EngineResult]) -> Dict[str, Any]:
        """Enhanced structured data merging with voting and validation"""
        
        if not results:
            return {}
        
        # Collect all structured data with confidence scores
        field_votes = defaultdict(list)
        field_values = defaultdict(list)
        field_confidences = defaultdict(list)
        
        for result in results:
            if result.structured_data:
                engine_weight = self.engine_weights.get(result.engine_name, 0.25)
                for field, value in result.structured_data.items():
                    if value:  # Only consider non-empty values
                        field_votes[field].append(result.engine_name)
                        field_values[field].append(value)
                        field_confidences[field].append(result.confidence * engine_weight)
        
        # Apply enhanced voting for each field
        merged_data = {}
        
        for field, values in field_values.items():
            if not values:
                continue
            
            # If all engines agree, use that value
            if len(values) == 1:
                merged_data[field] = values[0]
            elif len(set(str(v).lower() for v in values)) == 1:  # Case-insensitive match
                merged_data[field] = values[0]
            else:
                # Enhanced weighted voting
                best_value = self._select_best_value_with_validation(
                    values, 
                    field_confidences[field], 
                    field_votes[field]
                )
                if best_value:
                    merged_data[field] = best_value
        
        return merged_data
    
    def _select_best_value_with_validation(self, values: List[Any], confidences: List[float], engines: List[str]) -> Optional[Any]:
        """Select best value with validation for scanned documents"""
        
        # Score each value
        value_scores = defaultdict(float)
        value_counts = defaultdict(int)
        
        for i, value in enumerate(values):
            confidence = confidences[i]
            engine = engines[i]
            
            # Base score from confidence and engine weight
            score = confidence * self.engine_weights.get(engine, 0.25)
            
            # Boost for AI engines
            if engine == 'trocr':
                score *= 1.2
            
            # Boost for consistency (similar values)
            str_value = str(value).lower().strip()
            for j, other_value in enumerate(values):
                if i != j:
                    other_str = str(other_value).lower().strip()
                    if str_value == other_str:
                        score += 0.1
                    elif difflib.SequenceMatcher(None, str_value, other_str).ratio() > 0.8:
                        score += 0.05
            
            value_scores[value] = score
            value_counts[value] += 1
        
        # Select highest scoring value, but prefer consensus
        max_score = max(value_scores.values())
        best_candidates = [v for v, s in value_scores.items() if s == max_score]
        
        # If tie, prefer the most common value
        if len(best_candidates) > 1:
            best_candidates.sort(key=lambda x: value_counts[x], reverse=True)
        
        return best_candidates[0] if best_candidates else None