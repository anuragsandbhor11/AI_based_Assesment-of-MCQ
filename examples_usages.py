# src/grader.py
"""
Automated Grading System
Supports multiple choice and text-based questions with fuzzy matching
"""
import json
import os
from typing import Dict, Any, List, Tuple
from difflib import SequenceMatcher
import re

class Grader:
    """
    Main grading class that handles different question types
    """
    
    def __init__(self, answer_key: Dict[str, Dict[str, Any]], fuzzy_threshold: float = 0.8):
        """
        Initialize grader with answer key
        
        Args:
            answer_key: Dictionary with question IDs as keys and answer info as values
            fuzzy_threshold: Similarity threshold for text matching (0.0 to 1.0)
        """
        self.answer_key = answer_key
        self.fuzzy_threshold = fuzzy_threshold
        
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not isinstance(text, str):
            text = str(text)
        # Convert to lowercase, remove extra spaces and punctuation
        text = re.sub(r'[^\w\s]', '', text.lower().strip())
        return ' '.join(text.split())
    
    def fuzzy_match(self, student_answer: str, correct_answer: str) -> float:
        """
        Calculate similarity between two text answers
        
        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        student_norm = self.normalize_text(student_answer)
        correct_norm = self.normalize_text(correct_answer)
        
        # Use SequenceMatcher for fuzzy matching
        similarity = SequenceMatcher(None, student_norm, correct_norm).ratio()
        
        # Also check if one is contained in the other (for partial answers)
        if student_norm in correct_norm or correct_norm in student_norm:
            similarity = max(similarity, 0.7)
            
        return similarity
    
    def grade_single_question(self, question_id: str, student_answer: str) -> Dict[str, Any]:
        """
        Grade a single question
        
        Args:
            question_id: ID of the question
            student_answer: Student's answer
            
        Returns:
            Dictionary with grading results
        """
        if question_id not in self.answer_key:
            return {
                'question_id': question_id,
                'points_earned': 0,
                'points_possible': 0,
                'is_correct': False,
                'feedback': 'Question not found in answer key'
            }
        
        question_info = self.answer_key[question_id]
        correct_answer = question_info['answer']
        points_possible = question_info.get('points', 1)
        question_type = question_info.get('type', 'auto')  # auto-detect if not specified
        
        # Auto-detect question type if not specified
        if question_type == 'auto':
            if len(str(correct_answer)) == 1 and str(correct_answer).upper() in 'ABCDEFGHIJ':
                question_type = 'multiple_choice'
            else:
                question_type = 'text'
        
        # Grade based on question type
        if question_type == 'multiple_choice':
            return self._grade_multiple_choice(question_id, student_answer, correct_answer, points_possible)
        else:
            return self._grade_text_question(question_id, student_answer, correct_answer, points_possible)
    
    def _grade_multiple_choice(self, question_id: str, student_answer: str, correct_answer: str, points_possible: int) -> Dict[str, Any]:
        """Grade multiple choice questions"""
        student_clean = str(student_answer).strip().upper()
        correct_clean = str(correct_answer).strip().upper()
        
        is_correct = student_clean == correct_clean
        points_earned = points_possible if is_correct else 0
        
        return {
            'question_id': question_id,
            'points_earned': points_earned,
            'points_possible': points_possible,
            'is_correct': is_correct,
            'student_answer': student_answer,
            'correct_answer': correct_answer,
            'feedback': 'Correct!' if is_correct else f'Incorrect. Correct answer: {correct_answer}'
        }
    
    def _grade_text_question(self, question_id: str, student_answer: str, correct_answer: str, points_possible: int) -> Dict[str, Any]:
        """Grade text-based questions with fuzzy matching"""
        similarity = self.fuzzy_match(student_answer, correct_answer)
        
        if similarity >= self.fuzzy_threshold:
            # Full credit for close matches
            points_earned = points_possible
            is_correct = True
            feedback = 'Correct!'
        elif similarity >= 0.5:
            # Partial credit for somewhat similar answers
            points_earned = int(points_possible * similarity)
            is_correct = False
            feedback = f'Partially correct (similarity: {similarity:.2f}). Expected: {correct_answer}'
        else:
            # No credit for very different answers
            points_earned = 0
            is_correct = False
            feedback = f'Incorrect. Expected: {correct_answer}'
        
        return {
            'question_id': question_id,
            'points_earned': points_earned,
            'points_possible': points_possible,
            'is_correct': is_correct,
            'similarity': similarity,
            'student_answer': student_answer,
            'correct_answer': correct_answer,
            'feedback': feedback
        }
    
    def grade_answers(self, student_answers: Dict[str, str]) -> Dict[str, Any]:
        """
        Grade all student answers
        
        Args:
            student_answers: Dictionary with question IDs as keys and answers as values
            
        Returns:
            Complete grading results
        """
        results = []
        total_points_earned = 0
        total_points_possible = 0
        correct_count = 0
        
        # Grade each question
        for question_id in self.answer_key.keys():
            student_answer = student_answers.get(question_id, '')
            result = self.grade_single_question(question_id, student_answer)
            results.append(result)
            
            total_points_earned += result['points_earned']
            total_points_possible += result['points_possible']
            if result['is_correct']:
                correct_count += 1
        
        # Calculate percentage
        percentage = (total_points_earned / total_points_possible * 100) if total_points_possible > 0 else 0
        
        # Determine letter grade
        letter_grade = self._calculate_letter_grade(percentage)
        
        return {
            'raw_score': total_points_earned,
            'total_possible': total_points_possible,
            'percentage': round(percentage, 2),
            'letter_grade': letter_grade,
            'correct_count': correct_count,
            'total_questions': len(self.answer_key),
            'question_results': results,
            'timestamp': str(datetime.now())
        }
    
    def _calculate_letter_grade(self, percentage: float) -> str:
        """Convert percentage to letter grade"""
        if percentage >= 97:
            return 'A+'
        elif percentage >= 93:
            return 'A'
        elif percentage >= 90:
            return 'A-'
        elif percentage >= 87:
            return 'B+'
        elif percentage >= 83:
            return 'B'
        elif percentage >= 80:
            return 'B-'
        elif percentage >= 77:
            return 'C+'
        elif percentage >= 73:
            return 'C'
        elif percentage >= 70:
            return 'C-'
        elif percentage >= 67:
            return 'D+'
        elif percentage >= 63:
            return 'D'
        elif percentage >= 60:
            return 'D-'
        else:
            return 'F'

def load_answer_key(filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Load answer key from JSON file
    
    Args:
        filepath: Path to JSON file containing answer key
        
    Returns:
        Answer key dictionary
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Answer key file not found: {filepath}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in answer key file: {filepath}")

def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save grading results to JSON file
    
    Args:
        results: Grading results dictionary
        filepath: Output file path
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {filepath}")
    except Exception as e:
        print(f"Error saving results: {e}")

# examples/example_usage.py
"""
Example usage of the Automated Grading System
"""
import sys
import os
import json
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from grader import Grader, load_answer_key, save_results

def example_basic_grading():
    """Basic grading example"""
    print("=== Basic Grading Example ===")
   
    # Sample answer key
    answer_key = {
        '1': {'answer': 'A', 'points': 2},
        '2': {'answer': 'B', 'points': 2},
        '3': {'answer': 'photosynthesis', 'points': 3},
        '4': {'answer': 'C', 'points': 2},
        '5': {'answer': 'mitochondria', 'points': 3}
    }
   
    # Create grader
    grader = Grader(answer_key)
   
    # Student answers
    student_answers = {
        '1': 'A',           # Correct MC
        '2': 'D',           # Wrong MC  
        '3': 'photosynthesis',  # Correct text
        '4': 'C',           # Correct MC
        '5': 'mitochondrion'    # Similar text (should get partial credit)
    }
   
    # Grade the answers
    result = grader.grade_answers(student_answers)
   
    # Display results
    print(f"Student Score: {result['raw_score']}/{result['total_possible']}")
    print(f"Percentage: {result['percentage']}%")
    print(f"Letter Grade: {result['letter_grade']}")
    print(f"Questions Correct: {result['correct_count']}/{result['total_questions']}")
    print("\nDetailed Results:")
    
    for q_result in result['question_results']:
        print(f"Question {q_result['question_id']}: {q_result['points_earned']}/{q_result['points_possible']} - {q_result['feedback']}")
    
    return result

def example_advanced_grading():
    """Advanced grading with mixed question types and partial credit"""
    print("\n=== Advanced Grading Example ===")
    
    answer_key = {
        '1': {'answer': 'A', 'points': 1, 'type': 'multiple_choice'},
        '2': {'answer': 'The process by which plants convert sunlight into energy', 'points': 5, 'type': 'text'},
        '3': {'answer': 'B', 'points': 1, 'type': 'multiple_choice'},
        '4': {'answer': 'cellular respiration', 'points': 3, 'type': 'text'},
        '5': {'answer': 'D', 'points': 2, 'type': 'multiple_choice'},
        '6': {'answer': 'DNA stores genetic information', 'points': 4, 'type': 'text'}
    }
    
    # Create grader with custom fuzzy threshold
    grader = Grader(answer_key, fuzzy_threshold=0.75)
    
    student_answers = {
        '1': 'A',                                           # Correct
        '2': 'Plants use sunlight to make energy',          # Similar, should get partial credit
        '3': 'C',                                           # Wrong
        '4': 'respiration in cells',                        # Partially correct
        '5': 'D',                                           # Correct
        '6': 'DNA contains genetic info'                    # Very similar, should get full credit
    }
    
    result = grader.grade_answers(student_answers)
    
    print(f"Student Score: {result['raw_score']}/{result['total_possible']}")
    print(f"Percentage: {result['percentage']}%")
    print(f"Letter Grade: {result['letter_grade']}")
    print("\nDetailed Results:")
    
    for q_result in result['question_results']:
        feedback = q_result['feedback']
        if 'similarity' in q_result:
            feedback += f" (Similarity: {q_result['similarity']:.2f})"
        print(f"Question {q_result['question_id']}: {q_result['points_earned']}/{q_result['points_possible']} - {feedback}")
    
    return result

def example_from_json_file():
    """Example using answer key from JSON file"""
    print("\n=== JSON File Example ===")
    
    # Create sample answer key file
    sample_answer_key = {
        "1": {"answer": "A", "points": 2, "type": "multiple_choice"},
        "2": {"answer": "The mitochondria is the powerhouse of the cell", "points": 5, "type": "text"},
        "3": {"answer": "C", "points": 1, "type": "multiple_choice"},
        "4": {"answer": "chloroplast", "points": 3, "type": "text"}
    }
    
    # Save to temporary file
    temp_file = "temp_answer_key.json"
    with open(temp_file, 'w') as f:
        json.dump(sample_answer_key, f, indent=2)
    
    try:
        # Load answer key from file
        answer_key = load_answer_key(temp_file)
        grader = Grader(answer_key)
        
        student_answers = {
            "1": "A",
            "2": "mitochondria is powerhouse of cell",
            "3": "B",
            "4": "chloroplasts"
        }
        
        result = grader.grade_answers(student_answers)
        
        print(f"Student Score: {result['raw_score']}/{result['total_possible']}")
        print(f"Percentage: {result['percentage']}%")
        print(f"Letter Grade: {result['letter_grade']}")
        
        # Save results to file
        results_file = "student_results.json"
        save_results(result, results_file)
        
    finally:
        # Clean up temporary files
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists("student_results.json"):
            print("Results saved to student_results.json")

def example_batch_grading():
    """Example of grading multiple students"""
    print("\n=== Batch Grading Example ===")
    
    answer_key = {
        '1': {'answer': 'B', 'points': 2},
        '2': {'answer': 'evolution', 'points': 4},
        '3': {'answer': 'A', 'points': 1},
        '4': {'answer': 'natural selection', 'points': 3}
    }
    
    grader = Grader(answer_key)
    
    # Multiple students' answers
    students = {
        'Student_001': {
            '1': 'B', '2': 'evolution', '3': 'A', '4': 'natural selection'
        },
        'Student_002': {
            '1': 'A', '2': 'evolutionary theory', '3': 'A', '4': 'selection pressure'
        },
        'Student_003': {
            '1': 'B', '2': 'evolution', '3': 'C', '4': 'survival of fittest'
        }
    }
    
    print("Batch Grading Results:")
    print("-" * 50)
    
    all_results = {}
    for student_id, answers in students.items():
        result = grader.grade_answers(answers)
        all_results[student_id] = result
        
        print(f"{student_id}: {result['raw_score']}/{result['total_possible']} "
              f"({result['percentage']:.1f}%) - {result['letter_grade']}")
    
    # Calculate class statistics
    scores = [r['percentage'] for r in all_results.values()]
    avg_score = sum(scores) / len(scores)
    
    print(f"\nClass Average: {avg_score:.1f}%")
    print(f"Highest Score: {max(scores):.1f}%")
    print(f"Lowest Score: {min(scores):.1f}%")

def main():
    """Run all examples"""
    print("Automated Grading System - Examples")
    print("=" * 50)
    
    try:
        example_basic_grading()
        example_advanced_grading()
        example_from_json_file()
        example_batch_grading()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()