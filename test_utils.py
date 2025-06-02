"""
Utility tests for grading module
"""

import unittest
import tempfile
import os
import csv
import json
from unittest.mock import patch, mock_open
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from grader import load_answer_key, Grader


class TestLoadAnswerKey(unittest.TestCase):
    """Test cases for answer key loading functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_csv_content = """question,answer,points
1,A,2
2,B,2
3,photosynthesis,3
4,C,2
5,mitochondria,3"""
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temp files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    def test_load_valid_answer_key(self):
        """Test loading a valid answer key CSV"""
        # Create temporary CSV file
        csv_path = os.path.join(self.temp_dir, 'answer_key.csv')
        with open(csv_path, 'w', newline='') as f:
            f.write(self.sample_csv_content)
        
        # Load answer key
        answer_key = load_answer_key(csv_path)
        
        # Verify structure
        self.assertEqual(len(answer_key), 5)
        self.assertIn('1', answer_key)
        self.assertEqual(answer_key['1']['answer'], 'A')
        self.assertEqual(answer_key['1']['points'], 2)
        self.assertEqual(answer_key['3']['answer'], 'photosynthesis')
        self.assertEqual(answer_key['3']['points'], 3)
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file"""
        with self.assertRaises(FileNotFoundError):
            load_answer_key('nonexistent_file.csv')
    
    def test_load_malformed_csv(self):
        """Test loading malformed CSV"""
        # Create malformed CSV
        csv_path = os.path.join(self.temp_dir, 'bad_answer_key.csv')
        with open(csv_path, 'w') as f:
            f.write("invalid,csv,format\n1,A,not_a_number")
        
        with self.assertRaises(ValueError):
            load_answer_key(csv_path)
    
    def test_load_empty_csv(self):
        """Test loading empty CSV"""
        csv_path = os.path.join(self.temp_dir, 'empty.csv')
        with open(csv_path, 'w') as f:
            f.write("question,answer,points\n")
        
        answer_key = load_answer_key(csv_path)
        self.assertEqual(len(answer_key), 0)
    
    def test_load_csv_with_missing_columns(self):
        """Test CSV with missing required columns"""
        csv_path = os.path.join(self.temp_dir, 'incomplete.csv')
        with open(csv_path, 'w') as f:
            f.write("question,answer\n1,A\n")  # Missing points column
        
        answer_key = load_answer_key(csv_path)
        # Should default to 1 point
        self.assertEqual(answer_key['1']['points'], 1)
    
    def test_load_csv_with_whitespace(self):
        """Test CSV with extra whitespace"""
        csv_content = """question,answer,points
 1 , A ,2
2, B ,2"""
        
        csv_path = os.path.join(self.temp_dir, 'whitespace.csv')
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        
        answer_key = load_answer_key(csv_path)
        self.assertEqual(answer_key['1']['answer'], 'A')
        self.assertEqual(answer_key['2']['answer'], 'B')


class TestGraderUtilities(unittest.TestCase):
    """Test utility methods of Grader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.answer_key = {
            '1': {'answer': 'A', 'points': 2},
            '2': {'answer': 'photosynthesis', 'points': 3}
        }
        self.grader = Grader(self.answer_key)
    
    def test_text_normalization_edge_cases(self):
        """Test text normalization with edge cases"""
        # Empty string
        self.assertEqual(self.grader._normalize_for_comparison(''), '')
        
        # Only whitespace
        self.assertEqual(self.grader._normalize_for_comparison('   '), '')
        
        # Special characters
        self.assertEqual(
            self.grader._normalize_for_comparison('photo@synthesis!'),
            'photosynthesis'
        )
        
        # Multiple hyphens and spaces
        self.assertEqual(
            self.grader._normalize_for_comparison('photo - syn - thesis'),
            'photo syn thesis'
        )
        
        # Unicode characters
        self.assertEqual(
            self.grader._normalize_for_comparison('café'),
            'café'
        )
    
    def test_similarity_edge_cases(self):
        """Test similarity calculation edge cases"""
        # Both empty
        self.assertEqual(self.grader._calculate_similarity('', ''), 1.0)
        
        # One empty
        self.assertEqual(self.grader._calculate_similarity('test', ''), 0.0)
        self.assertEqual(self.grader._calculate_similarity('', 'test'), 0.0)
        
        # Identical
        self.assertEqual(self.grader._calculate_similarity('test', 'test'), 1.0)
        
        # Completely different
        similarity = self.grader._calculate_similarity('abc', 'xyz')
        self.assertEqual(similarity, 0.0)
        
        # Partial match
        similarity = self.grader._calculate_similarity('testing', 'test')
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)
    
    def test_mc_letter_extraction_edge_cases(self):
        """Test multiple choice letter extraction edge cases"""
        # Empty string
        self.assertEqual(self.grader._extract_mc_letter(''), '')
        
        # Only numbers
        self.assertEqual(self.grader._extract_mc_letter('123'), '')
        
        # Only special characters
        self.assertEqual(self.grader._extract_mc_letter('!@#'), '')
        
        # Multiple letters - should get first
        self.assertEqual(self.grader._extract_mc_letter('A and B'), 'A')
        
        # Letter in middle of word
        self.assertEqual(self.grader._extract_mc_letter('Answer is A'), 'A')
        
        # Lowercase
        self.assertEqual(self.grader._extract_mc_letter('answer: b'), 'B')
    
    def test_mc_detection_edge_cases(self):
        """Test multiple choice detection edge cases"""
        # Empty string
        self.assertFalse(self.grader._is_multiple_choice(''))
        
        # Only whitespace
        self.assertFalse(self.grader._is_multiple_choice('   '))
        
        # Multiple letters
        self.assertFalse(self.grader._is_multiple_choice('AB'))
        
        # Numbers
        self.assertFalse(self.grader._is_multiple_choice('1'))
        
        # Special characters
        self.assertFalse(self.grader._is_multiple_choice('@'))
        
        # Valid single letters
        self.assertTrue(self.grader._is_multiple_choice('A'))
        self.assertTrue(self.grader._is_multiple_choice('z'))
        self.assertTrue(self.grader._is_multiple_choice(' B '))


class TestGraderExport(unittest.TestCase):
    """Test export functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.answer_key = {
            '1': {'answer': 'A', 'points': 2},
            '2': {'answer': 'B', 'points': 2}
        }
        self.grader = Grader(self.answer_key)
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample results
        self.sample_results = [
            {
                'student_id': 'student_001',
                'raw_score': 4,
                'total_possible': 4,
                'percentage': 100.0,
                'correct_count': 2,
                'total_questions': 2,
                'question_scores': {
                    '1': {
                        'student_answer': 'A',
                        'correct_answer': 'A',
                        'points_earned': 2,
                        'max_points': 2,
                        'is_correct': True,
                        'similarity_score': 1.0
                    },
                    '2': {
                        'student_answer': 'B',
                        'correct_answer': 'B',
                        'points_earned': 2,
                        'max_points': 2,
                        'is_correct': True,
                        'similarity_score': 1.0
                    }
                }
            }
        ]
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temp files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    def test_export_json(self):
        """Test JSON export functionality"""
        output_path = os.path.join(self.temp_dir, 'results.json')
        
        self.grader.export_results(self.sample_results, output_path, 'json')
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify content
        with open(output_path, 'r') as f:
            exported_data = json.load(f)
        
        self.assertEqual(len(exported_data), 1)
        self.assertEqual(exported_data[0]['student_id'], 'student_001')
        self.assertEqual(exported_data[0]['percentage'], 100.0)
    
    def test_export_csv(self):
        """Test CSV export functionality"""
        output_path = os.path.join(self.temp_dir, 'results.csv')
        
        self.grader.export_results(self.sample_results, output_path, 'csv')
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify content
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['student_id'], 'student_001')
        self.assertEqual(rows[0]['percentage'], '100.0')
        self.assertEqual(rows[0]['q1_answer'], 'A')
        self.assertEqual(rows[0]['q1_correct'], 'True')
    
    def test_export_empty_results(self):
        """Test exporting empty results"""
        output_path = os.path.join(self.temp_dir, 'empty.json')
        
        self.grader.export_results([], output_path, 'json')
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify content
        with open(output_path, 'r') as f:
            exported_data = json.load(f)
        
        self.assertEqual(exported_data, [])
    
    def test_export_invalid_format(self):
        """Test export with invalid format"""
        output_path = os.path.join(self.temp_dir, 'results.txt')
        
        with self.assertRaises(ValueError):
            self.grader.export_results(self.sample_results, output_path, 'invalid')


class TestGraderStatisticsEdgeCases(unittest.TestCase):
    """Test statistical calculations with edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.answer_key = {'1': {'answer': 'A', 'points': 1}}
        self.grader = Grader(self.answer_key)
    
    def test_grade_distribution_edge_cases(self):
        """Test grade distribution with edge cases"""
        # All same grade
        same_scores = [95.0, 95.0, 95.0]
        distribution = self.grader._calculate_grade_distribution(same_scores)
        self.assertEqual(distribution['A (90-100)'], 3)
        self.assertEqual(sum(distribution.values()), 3)
        
        # Boundary scores
        boundary_scores = [90.0, 80.0, 70.0, 60.0, 59.9]
        distribution = self.grader._calculate_grade_distribution(boundary_scores)
        self.assertEqual(distribution['A (90-100)'], 1)
        self.assertEqual(distribution['B (80-89)'], 1)
        self.assertEqual(distribution['C (70-79)'], 1)
        self.assertEqual(distribution['D (60-69)'], 1)
        self.assertEqual(distribution['F (0-59)'], 1)
        
        # Empty list
        distribution = self.grader._calculate_grade_distribution([])
        self.assertEqual(sum(distribution.values()), 0)
    
    def test_question_analysis_no_attempts(self):
        """Test question analysis with no attempts"""
        # Empty results
        empty_results = []
        stats = self.grader.calculate_statistics(empty_results)
        
        self.assertEqual(stats['total_students'], 0)
        self.assertEqual(stats['average_score'], 0.0)
        self.assertEqual(stats['question_analysis'], {})
    
    def test_question_analysis_single_question(self):
        """Test question analysis with single question"""
        results = [
            {
                'student_id': 'student_001',
                'question_scores': {
                    '1': {
                        'student_answer': 'A',
                        'correct_answer': 'A',
                        'is_correct': True,
                        'similarity_score': 1.0
                    }
                }
            },
            {
                'student_id': 'student_002', 
                'question_scores': {
                    '1': {
                        'student_answer': 'B',
                        'correct_answer': 'A',
                        'is_correct': False,
                        'similarity_score': 0.0
                    }
                }
            }
        ]
        
        stats = self.grader.calculate_statistics(results)
        question_1_stats = stats['question_analysis']['1']
        
        self.assertEqual(question_1_stats['correct_count'], 1)
        self.assertEqual(question_1_stats['total_attempts'], 2)
        self.assertEqual(question_1_stats['accuracy_rate'], 0.5)
        self.assertEqual(question_1_stats['average_similarity'], 0.5)


class TestGraderCoreGrading(unittest.TestCase):
    """Test core grading functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.answer_key = {
            '1': {'answer': 'A', 'points': 2},
            '2': {'answer': 'photosynthesis', 'points': 3},
            '3': {'answer': 'B', 'points': 1}
        }
        self.grader = Grader(self.answer_key)
    
    def test_grade_single_student_perfect_score(self):
        """Test grading a single student with perfect score"""
        student_answers = {
            '1': 'A',
            '2': 'photosynthesis',
            '3': 'B'
        }
        
        result = self.grader.grade_student('student_001', student_answers)
        
        self.assertEqual(result['student_id'], 'student_001')
        self.assertEqual(result['raw_score'], 6)
        self.assertEqual(result['total_possible'], 6)
        self.assertEqual(result['percentage'], 100.0)
        self.assertEqual(result['correct_count'], 3)
        self.assertEqual(result['total_questions'], 3)
        
        # Check individual question scores
        self.assertTrue(result['question_scores']['1']['is_correct'])
        self.assertEqual(result['question_scores']['1']['points_earned'], 2)
        self.assertTrue(result['question_scores']['2']['is_correct'])
        self.assertEqual(result['question_scores']['2']['points_earned'], 3)
    
    def test_grade_student_partial_credit(self):
        """Test grading with partial credit for text answers"""
        student_answers = {
            '1': 'A',
            '2': 'photo synthesis',  # Close but not exact
            '3': 'C'  # Wrong
        }
        
        result = self.grader.grade_student('student_002', student_answers)
        
        # Should get full credit for Q1, partial for Q2, none for Q3
        self.assertEqual(result['question_scores']['1']['points_earned'], 2)
        self.assertGreater(result['question_scores']['2']['points_earned'], 0)
        self.assertLess(result['question_scores']['2']['points_earned'], 3)
        self.assertEqual(result['question_scores']['3']['points_earned'], 0)
    
    def test_grade_student_missing_answers(self):
        """Test grading student with missing answers"""
        student_answers = {
            '1': 'A',
            # Question 2 missing
            '3': 'B'
        }
        
        result = self.grader.grade_student('student_003', student_answers)
        
        # Should have entry for missing question with 0 points
        self.assertIn('2', result['question_scores'])
        self.assertEqual(result['question_scores']['2']['points_earned'], 0)
        self.assertEqual(result['question_scores']['2']['student_answer'], '')
    
    def test_grade_multiple_students(self):
        """Test grading multiple students"""
        students_data = [
            {'student_id': 'student_001', 'answers': {'1': 'A', '2': 'photosynthesis', '3': 'B'}},
            {'student_id': 'student_002', 'answers': {'1': 'B', '2': 'cellular respiration', '3': 'A'}},
            {'student_id': 'student_003', 'answers': {'1': 'A', '2': 'photosynthesis', '3': 'C'}}
        ]
        
        results = self.grader.grade_multiple_students(students_data)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['student_id'], 'student_001')
        self.assertEqual(results[1]['student_id'], 'student_002')
        self.assertEqual(results[2]['student_id'], 'student_003')
        
        # First student should have perfect score
        self.assertEqual(results[0]['percentage'], 100.0)
        
        # Other students should have lower scores
        self.assertLess(results[1]['percentage'], 100.0)
        self.assertLess(results[2]['percentage'], 100.0)
    
    def test_multiple_choice_vs_text_grading(self):
        """Test different grading approaches for MC vs text"""
        mc_answer_key = {'1': {'answer': 'A', 'points': 1}}
        text_answer_key = {'2': {'answer': 'photosynthesis', 'points': 1}}
        
        mc_grader = Grader(mc_answer_key)
        text_grader = Grader(text_answer_key)
        
        # MC should be exact match only
        mc_result = mc_grader.grade_student('test', {'1': 'a'})  # lowercase
        self.assertTrue(mc_result['question_scores']['1']['is_correct'])  # Should normalize to uppercase
        
        mc_result_wrong = mc_grader.grade_student('test', {'1': 'Answer A'})
        self.assertTrue(mc_result_wrong['question_scores']['1']['is_correct'])  # Should extract 'A'
        
        # Text should allow partial credit
        text_result = text_grader.grade_student('test', {'2': 'photo synthesis'})
        self.assertGreater(text_result['question_scores']['2']['points_earned'], 0)
        self.assertLess(text_result['question_scores']['2']['points_earned'], 1)


class TestGraderConfigurationAndSettings(unittest.TestCase):
    """Test grader configuration and settings"""
    
    def test_custom_similarity_threshold(self):
        """Test custom similarity threshold setting"""
        answer_key = {'1': {'answer': 'photosynthesis', 'points': 1}}
        
        # Strict grader
        strict_grader = Grader(answer_key, similarity_threshold=0.9)
        
        # Lenient grader  
        lenient_grader = Grader(answer_key, similarity_threshold=0.6)
        
        student_answer = {'1': 'photo synthesis'}
        
        strict_result = strict_grader.grade_student('test', student_answer)
        lenient_result = lenient_grader.grade_student('test', student_answer)
        
        # Lenient should give more credit for the same answer
        self.assertGreaterEqual(
            lenient_result['question_scores']['1']['points_earned'],
            strict_result['question_scores']['1']['points_earned']
        )
    
    def test_case_sensitivity_handling(self):
        """Test case sensitivity in answers"""
        answer_key = {
            '1': {'answer': 'DNA', 'points': 1},
            '2': {'answer': 'photosynthesis', 'points': 1}
        }
        grader = Grader(answer_key)
        
        # Test various case combinations
        test_cases = [
            {'1': 'dna', '2': 'PHOTOSYNTHESIS'},
            {'1': 'Dna', '2': 'PhotoSynthesis'},
            {'1': 'DNA', '2': 'photosynthesis'}
        ]
        
        for answers in test_cases:
            result = grader.grade_student('test', answers)
            # All should be considered correct (case insensitive)
            self.assertTrue(result['question_scores']['1']['is_correct'])
            self.assertTrue(result['question_scores']['2']['is_correct'])


class TestGraderStatisticsComprehensive(unittest.TestCase):
    """Comprehensive tests for statistics calculation"""
    
    def setUp(self):
        """Set up comprehensive test data"""
        self.answer_key = {
            '1': {'answer': 'A', 'points': 1},
            '2': {'answer': 'B', 'points': 2},
            '3': {'answer': 'photosynthesis', 'points': 3}
        }
        self.grader = Grader(self.answer_key)
        
        # Sample results with variety of scores
        self.sample_results = [
            {
                'student_id': 'student_001',
                'raw_score': 6,
                'total_possible': 6,
                'percentage': 100.0,
                'question_scores': {
                    '1': {'is_correct': True, 'similarity_score': 1.0},
                    '2': {'is_correct': True, 'similarity_score': 1.0},
                    '3': {'is_correct': True, 'similarity_score': 1.0}
                }
            },
            {
                'student_id': 'student_002',
                'raw_score': 4.5,
                'total_possible': 6,
                'percentage': 75.0,
                'question_scores': {
                    '1': {'is_correct': True, 'similarity_score': 1.0},
                    '2': {'is_correct': True, 'similarity_score': 1.0},
                    '3': {'is_correct': False, 'similarity_score': 0.5}
                }
            },
            {
                'student_id': 'student_003',
                'raw_score': 1,
                'total_possible': 6,
                'percentage': 16.67,
                'question_scores': {
                    '1': {'is_correct': True, 'similarity_score': 1.0},
                    '2': {'is_correct': False, 'similarity_score': 0.0},
                    '3': {'is_correct': False, 'similarity_score': 0.0}
                }
            }
        ]
    
    def test_comprehensive_statistics(self):
        """Test comprehensive statistics calculation"""
        stats = self.grader.calculate_statistics(self.sample_results)
        
        # Basic stats
        self.assertEqual(stats['total_students'], 3)
        self.assertAlmostEqual(stats['average_score'], 63.89, places=1)
        self.assertEqual(stats['highest_score'], 100.0)
        self.assertEqual(stats['lowest_score'], 16.67)
        
        # Grade distribution
        expected_distribution = {
            'A (90-100)': 1,
            'B (80-89)': 0, 
            'C (70-79)': 1,
            'D (60-69)': 0,
            'F (0-59)': 1
        }
        self.assertEqual(stats['grade_distribution'], expected_distribution)
        
        # Question analysis
        q1_stats = stats['question_analysis']['1']
        self.assertEqual(q1_stats['correct_count'], 3)
        self.assertEqual(q1_stats['total_attempts'], 3)
        self.assertEqual(q1_stats['accuracy_rate'], 1.0)
        
        q3_stats = stats['question_analysis']['3']
        self.assertEqual(q3_stats['correct_count'], 1)
        self.assertEqual(q3_stats['total_attempts'], 3)
        self.assertAlmostEqual(q3_stats['accuracy_rate'], 0.333, places=2)


class TestGraderErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.answer_key = {'1': {'answer': 'A', 'points': 1}}
        self.grader = Grader(self.answer_key)
    
    def test_invalid_student_data(self):
        """Test handling of invalid student data"""
        # Missing student_id
        with self.assertRaises(KeyError):
            self.grader.grade_multiple_students([{'answers': {'1': 'A'}}])
        
        # Missing answers
        with self.assertRaises(KeyError):
            self.grader.grade_multiple_students([{'student_id': 'test'}])
    
    def test_empty_answer_key(self):
        """Test grader with empty answer key"""
        empty_grader = Grader({})
        
        result = empty_grader.grade_student('test', {'1': 'A'})
        
        self.assertEqual(result['raw_score'], 0)
        self.assertEqual(result['total_possible'], 0)
        self.assertEqual(result['percentage'], 0.0)
        self.assertEqual(result['question_scores'], {})
    
    def test_unicode_handling(self):
        """Test handling of unicode characters"""
        unicode_key = {'1': {'answer': 'café', 'points': 1}}
        unicode_grader = Grader(unicode_key)
        
        # Test exact match
        result = unicode_grader.grade_student('test', {'1': 'café'})
        self.assertTrue(result['question_scores']['1']['is_correct'])
        
        # Test similar but different unicode
        result2 = unicode_grader.grade_student('test', {'1': 'cafe'})
        self.assertGreater(result2['question_scores']['1']['similarity_score'], 0.8)
    
    def test_very_long_answers(self):
        """Test handling of very long answers"""
        long_answer = 'A' * 1000
        long_key = {'1': {'answer': long_answer, 'points': 1}}
        long_grader = Grader(long_key)
        
        # Test exact long match
        result = long_grader.grade_student('test', {'1': long_answer})
        self.assertTrue(result['question_scores']['1']['is_correct'])
        
        # Test partial long match
        partial_answer = 'A' * 999 + 'B'
        result2 = long_grader.grade_student('test', {'1': partial_answer})
        self.assertGreater(result2['question_scores']['1']['similarity_score'], 0.95)


if __name__ == '__main__':
    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add all test cases
    test_classes = [
        TestLoadAnswerKey,
        TestGraderUtilities,
        TestGraderExport,
        TestGraderStatisticsEdgeCases,
        TestGraderCoreGrading,
        TestGraderConfigurationAndSettings,
        TestGraderStatisticsComprehensive,
        TestGraderErrorHandling
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)