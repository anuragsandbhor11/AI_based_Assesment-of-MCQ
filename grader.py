


import csv
import logging
from typing import Dict, List, Tuple, Any
import difflib
import re

logger = logging.getLogger(__name__)

class Grader:
    """Handles grading logic for student responses."""
    
    def __init__(self, answer_key_path: str):
        
        self.answer_key = self._load_answer_key(answer_key_path)
        logger.info(f"Loaded answer key with {len(self.answer_key)} questions")
    
    def _load_answer_key(self, answer_key_path: str) -> Dict[str, Dict[str, Any]]:
        
        answer_key = {}
        try:
            with open(answer_key_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    question_id = str(row['question_id']).strip()
                    answer_key[question_id] = {
                        'correct_answer': row['correct_answer'].strip(),
                        'points': float(row['points'])
                    }
            return answer_key
        except Exception as e:
            logger.error(f"Error loading answer key: {e}")
            raise
    
    def grade_student_responses(self, student_responses: Dict[str, str], 
                              student_id: str = "unknown") -> Dict[str, Any]:
        
        results = {
            'student_id': student_id,
            'answers': {},
            'scores': {},
            'raw_score': 0.0,
            'total_possible': 0.0,
            'percentage': 0.0
        }
        
        for question_id, correct_data in self.answer_key.items():
            student_answer = student_responses.get(question_id, "").strip()
            correct_answer = correct_data['correct_answer']
            points = correct_data['points']
            
            # Calculate score for this question
            score = self._calculate_question_score(student_answer, correct_answer, points)
            
            results['answers'][question_id] = student_answer
            results['scores'][question_id] = score
            results['raw_score'] += score
            results['total_possible'] += points
        
        # Calculate percentage
        if results['total_possible'] > 0:
            results['percentage'] = round((results['raw_score'] / results['total_possible']) * 100, 2)
        
        logger.debug(f"Graded {student_id}: {results['raw_score']}/{results['total_possible']} ({results['percentage']}%)")
        return results
    
    def _calculate_question_score(self, student_answer: str, correct_answer: str, points: float) -> float:
        
        if not student_answer:
            return 0.0
        
        # Normalize answers for comparison
        student_normalized = self._normalize_answer(student_answer)
        correct_normalized = self._normalize_answer(correct_answer)
        
        # Exact match
        if student_normalized == correct_normalized:
            return points
        
        # For multiple choice, be strict
        if self._is_multiple_choice(correct_answer):
            return 0.0
        
        # For text answers, allow partial credit based on similarity
        similarity = self._calculate_similarity(student_normalized, correct_normalized)
        
        # Award partial credit if similarity is high enough
        if similarity >= 0.8:  # 80% similarity threshold
            return points
        elif similarity >= 0.6:  # 60% similarity threshold
            return points * 0.5  # Half credit
        else:
            return 0.0
    
    def _normalize_answer(self, answer: str) -> str:
       
        # Convert to lowercase
        normalized = answer.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove common punctuation
        normalized = re.sub(r'[.,!?;:]', '', normalized)
        
        return normalized
    
    def _is_multiple_choice(self, answer: str) -> bool:
       
        normalized = answer.strip().upper()
        return len(normalized) == 1 and normalized in 'ABCDEFGHIJKLMNOP'
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        
        return difflib.SequenceMatcher(None, text1, text2).ratio()


def calculate_class_statistics(individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    
    if not individual_results:
        return {}
    
    percentages = [result['percentage'] for result in individual_results]
    percentages.sort()
    
    # Basic statistics
    stats = {
        'average_score': round(sum(percentages) / len(percentages), 2),
        'median_score': _calculate_median(percentages),
        'highest_score': max(percentages),
        'lowest_score': min(percentages),
        'standard_deviation': round(_calculate_std_dev(percentages), 2)
    }
    
    # Grade distribution
    stats['grade_distribution'] = _calculate_grade_distribution(percentages)
    
    return stats


def _calculate_median(values: List[float]) -> float:
    """Calculate median of a list of values."""
    n = len(values)
    if n % 2 == 0:
        return round((values[n//2 - 1] + values[n//2]) / 2, 2)
    else:
        return round(values[n//2], 2)


def _calculate_std_dev(values: List[float]) -> float:
    """Calculate standard deviation of a list of values."""
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def _calculate_grade_distribution(percentages: List[float]) -> Dict[str, int]:
    
    distribution = {
        'A (90-100)': 0,
        'B (80-89)': 0,
        'C (70-79)': 0,
        'D (60-69)': 0,
        'F (0-59)': 0
    }
    
    for percentage in percentages:
        if percentage >= 90:
            distribution['A (90-100)'] += 1
        elif percentage >= 80:
            distribution['B (80-89)'] += 1
        elif percentage >= 70:
            distribution['C (70-79)'] += 1
        elif percentage >= 60:
            distribution['D (60-69)'] += 1
        else:
            distribution['F (0-59)'] += 1
    
    return distribution


def compare_answers(student_answer: str, correct_answer: str, case_sensitive: bool = False) -> bool:
    
    if not case_sensitive:
        return student_answer.strip().lower() == correct_answer.strip().lower()
    return student_answer.strip() == correct_answer.strip()