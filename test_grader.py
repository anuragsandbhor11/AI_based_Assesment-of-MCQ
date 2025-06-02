"""
Unit tests for grading module
"""

import unittest
import sys
import os
import tempfile
import csv
import json
from unittest.mock import patch, mock_open

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from grader import Grader, load_answer_key


class TestGrader(unittest.TestCase):
    """Test cases for Grader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_answer_key = {
            '1': {'answer': 'A', 'points': 2},
            '2': {'answer': 'B', 'points': 2},
            '3': {'answer': 'photosynthesis', 'points': 3},
            '4': {'answer': 'C', 'points': 2},
            '5': {'answer': 'mitochondria', 'points': 3}
        }
        self.grader = Grader(self.sample_answer_key)
    
    def test_initialization(self):
        """Test grader initialization"""
        self.assertEqual(len(self.grader.answer_key), 5)
        self.assertIn('1', self.grader.answer_key)
        self.assertEqual(self.grader.answer_key['1']['answer'], 'A')
        self.assertEqual(self.grader.answer_key['1']['points'], 2)
    
    def test_exact_match_scoring(self):
        """Test exact answer matching"""
        student_answers = {
            '1': 'A',
            '2': 'B',
            '3': 'photosynthesis',
            '4': 'C',
            '5': 'mitochondria'
        }
        
        result = self.grader.grade_answers(student_answers)
        
        self.assertEqual(result['raw_score'], 12)  # 2+2+3+2+3
        self.assertEqual(result['total_possible'], 12)
        self.assertEqual(result['percentage'], 100.0)
        self.assertEqual(result['correct_count'], 5)
        self.assertTrue(result['question_scores']['1']['is_correct'])
    
    def test_case_insensitive_matching(self):
        """Test case insensitive comparison"""
        student_answers = {
            '1': 'a',  # lowercase
            '2': 'b',  # lowercase
            '3': 'PHOTOSYNTHESIS',  # uppercase
            '4': 'c',  # lowercase
            '5': 'MITOCHONDRIA'  # uppercase
        }
        
        result = self.grader.grade_answers(student_answers)
        
        self.assertEqual(result['raw_score'], 12)
        self.assertEqual(result['percentage'], 100.0)
        self.assertEqual(result['correct_count'], 5)
    
    def test_partial_credit(self):
        """Test partial credit scenarios"""
        student_answers = {
            '1': 'A',
            '2': 'B', 
            '3': 'photo-synthesis',  # Hyphenated version
            '4': 'C',
            '5': 'mitochondria'
        }
        
        result = self.grader.grade_answers(student_answers)
        
        # Should get full credit due to normalization
        self.assertEqual(result['percentage'], 100.0)
    
    def test_wrong_answers(self):
        """Test completely wrong answers"""
        student_answers = {
            '1': 'D',  # Wrong MC
            '2': 'A',  # Wrong MC
            '3': 'respiration',  # Wrong text
            '4': 'B',  # Wrong MC
            '5': 'chloroplast'  # Wrong text
        }
        
        result = self.grader.grade_answers(student_answers)
        
        self.assertEqual(result['raw_score'], 0)
        self.assertEqual(result['percentage'], 0.0)
        self.assertEqual(result['correct_count'], 0)
    
    def test_mixed_answers(self):
        """Test mix of correct and wrong answers"""
        student_answers = {
            '1': 'A',  # Correct
            '2': 'D',  # Wrong
            '3': 'photosynthesis',  # Correct
            '4': 'B',  # Wrong
            '5': 'mitochondria'  # Correct
        }
        
        result = self.grader.grade_answers(student_answers)
        
        self.assertEqual(result['raw_score'], 8)  # 2+0+3+0+3
        self.assertEqual(result['total_possible'], 12)
        self.assertEqual(result['percentage'], 8/12 * 100)
        self.assertEqual(result['correct_count'], 3)
    
    def test_missing_answers(self):
        """Test handling of missing answers"""
        student_answers = {
            '1': 'A',
            '3': 'photosynthesis',
            '5': 'mitochondria'
            # Missing questions 2 and 4
        }
        
        result = self.grader.grade_answers(student_answers)
        
        self.assertEqual(result['raw_score'], 8)  # 2+0+3+0+3
        self.assertEqual(result['correct_count'], 3)
        self.assertFalse(result['question_scores']['2']['is_correct'])
        self.assertFalse(result['question_scores']['4']['is_correct'])
    
    def test_empty_answers(self):
        """Test handling of empty answer strings"""
        student_answers = {
            '1': '',
            '2': '   ',  # Whitespace only
            '3': 'photosynthesis',
            '4': 'C',
            '5': ''
        }
        
        result = self.grader.grade_answers(student_answers)
        
        self.assertEqual(result['raw_score'], 5)  # 0+0+3+2+0
        self.assertEqual(result['correct_count'], 2)
    
    def test_normalize_for_comparison(self):
        """Test answer normalization"""
        # Test case normalization
        self.assertEqual(
            self.grader._normalize_for_comparison('PHOTOSYNTHESIS'),
            'photosynthesis'
        )
        
        # Test whitespace removal
        self.assertEqual(
            self.grader._normalize_for_comparison('  mitochondria  '),
            'mitochondria'
        )
        
        # Test punctuation removal
        self.assertEqual(
            self.grader._normalize_for_comparison('photosynthesis!'),
            'photosynthesis'
        )
        
        # Test hyphen handling
        self.assertEqual(
            self.grader._normalize_for_comparison('photo-synthesis'),
            'photosynthesis'
        )
    
    def test_multiple_choice_detection(self):
        """Test multiple choice answer detection"""
        self.assertTrue(self.grader._is_multiple_choice('A'))
        self.assertTrue(self.grader._is_multiple_choice('B'))
        self.assertTrue(self.grader._is_multiple_choice('a'))
        self.assertFalse(self.grader._is_multiple_choice('AB'))
        self.assertFalse(self.grader._is_multiple_choice('photosynthesis'))
        self.assertFalse(self.grader._is_multiple_choice(''))
    
    def test_score_multiple_choice(self):
        """Test multiple choice scoring"""
        # Correct answer
        result = self.grader._score_multiple_choice('A', 'A', 2)
        self.assertEqual(result['points_earned'], 2)
        self.assertTrue(result['is_correct'])
        
        # Wrong answer
        result = self.grader._score_multiple_choice('B', 'A', 2)
        self.assertEqual(result['points_earned'], 0)
        self.assertFalse(result['is_correct'])
        
        # Extract letter from text
        result = self.grader._score_multiple_choice('Answer is A', 'A', 2)
        self.assertEqual(result['points_earned'], 2)
        self.assertTrue(result['is_correct'])
    
    def test_similarity_calculation(self):
        """Test text similarity calculation"""
        # Identical strings
        self.assertEqual(self.grader._calculate_similarity('test', 'test'), 1.0)
        
        # Empty strings
        self.assertEqual(self.grader._calculate_similarity('', ''), 1.0)
        
        # One empty
        self.assertEqual(self.grader._calculate_similarity('test', ''), 0.0)
        
        # Similar strings
        similarity = self.grader._calculate_similarity('photosynthesis', 'photosynthesis')
        self.assertEqual(similarity, 1.0)
        
        similarity = self.grader._calculate_similarity('photosynthesis', 'photo')
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)


class TestGraderStatistics(unittest.TestCase):
    """Test statistical analysis functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_answer_key = {
            '1': {'answer': 'A', 'points': 2},
            '2': {'answer': 'B', 'points': 2},
            '3': {'answer': 'photosynthesis', 'points': 3}
        }
        self.grader = Grader(self.sample_answer_key)
        
        # Sample results for statistics
        self.sample_results = [
            {
                'student_id': 'student_001',
                'raw_score': 7,
                'total_possible': 7,
                'percentage': 100.0,
                'question_scores': {
                    '1': {'is_correct': True, 'student_answer': 'A'},
                    '2': {'is_correct': True, 'student_answer': 'B'},
                    '3': {'is_correct': True, 'student_answer': 'photosynthesis'}
                }
            },
            {
                'student_id': 'student_002',
                'raw_score': 4,
                'total_possible': 7,
                'percentage': 57.14,
                'question_scores': {
                    '1': {'is_correct': True, 'student_answer': 'A'},
                    '2': {'is_correct': True, 'student_answer': 'B'},
                    '3': {'is_correct': False, 'student_answer': 'respiration'}
                }
            },
            {
                'student_id': 'student_003',
                'raw_score': 5,
                'total_possible': 7,
                'percentage': 71.43,
                'question_scores': {
                    '1': {'is_correct': False, 'student_answer': 'B'},
                    '2': {'is_correct': True, 'student_answer': 'B'},
                    '3': {'is_correct': True, 'student_answer': 'photosynthesis'}
                }
            }
        ]
    
    def test_calculate_statistics(self):
        """Test basic statistics calculation"""
        stats = self.grader.calculate_statistics(self.sample_results)
        
        self.assertEqual(stats['total_students'], 3)
        self.assertAlmostEqual(stats['average_score'], 76.19, places=1)
        self.assertAlmostEqual(stats['median_score'], 71.43, places=1)
        self.assertEqual(stats['highest_score'], 100.0)
        self.assertAlmostEqual(stats['lowest_score'], 57.14, places=1)
        self.assertGreater(stats['standard_deviation'], 0)
    
    def test_grade_distribution(self):
        """Test grade distribution calculation"""
        percentages = [95.0, 85.0, 75.0, 65.0, 55.0]
        distribution = self.grader._calculate_grade_distribution(percentages)
        
        self.assertEqual(distribution['A (90-100)'], 1)
        self.assertEqual(distribution['B (80-89)'], 1)
        self.assertEqual(distribution['C (70-79)'], 1)
        self.assertEqual(distribution['D (60-69)'], 1)
        self.assertEqual(distribution['F (0-59)'], 1)
        
        # Test all perfect scores
        perfect_scores = [100.0, 95.0, 92.0]
        distribution = self.grader._calculate_grade_distribution(perfect_scores)
        self.assertEqual(distribution['A (90-100)'], 3)
        self.assertEqual(sum(distribution.values()), 3)
    
    def test_question_analysis(self):
        """Test question difficulty analysis"""
        analysis = self.grader._analyze_questions(self.sample_results)
        
        # Question 1: 2/3 correct
        self.assertEqual(analysis['1']['correct_count'], 2)
        self.assertEqual(analysis['1']['total_attempts'], 3)
        self.assertAlmostEqual(analysis['1']['success_rate'], 66.67, places=1)
        
        # Question 2: 3/3 correct (easy)
        self.assertEqual(analysis['2']['correct_count'], 3)
        self.assertEqual(analysis['2']['difficulty_level'], 'easy')
        
        # Question 3: 2/3 correct
        self.assertEqual(analysis['3']['correct_count'], 2)
        self.assertIn('respiration', analysis['3']['common_wrong_answers'])
    
    def test_empty_results(self):
        """Test statistics with empty results"""
        stats = self.grader.calculate_statistics([])
        self.assertEqual(stats, {})
    
    def test_single_result(self):
        """Test statistics with single result"""
        single_result = [self.sample_results[0]]
        stats = self.grader.calculate_statistics(single_result)
        
        self.assertEqual(stats['total_students'], 1)
        self.assertEqual(stats['average_score'], 100.0)
        self.assertEqual(stats['median_score'], 100.0)
        self.assertEqual(stats['standard_deviation'], 0.0)


class TestLoadAnswerKey(unittest.TestCase):
    """Test answer key loading functionality"""
    
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
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_valid_csv(self):
        """Test loading valid CSV answer key"""
        csv_path = os.path.join(self.temp_dir, 'answer_key.csv')
        with open(csv_path, 'w', newline='') as f:
            f.write(self.sample_csv_content)
        
        answer_key = load_answer_key(csv_path)
        
        self.assertEqual(len(answer_key), 5)
        self.assertEqual(answer_key['1']['answer'], 'A')
        self.assertEqual(answer_key['1']['points'], 2)
        self.assertEqual(answer_key['3']['answer'], 'photosynthesis')
        self.assertEqual(answer_key['3']['points'], 3)
    
    def test_load_csv_with_whitespace(self):
        """Test loading CSV with extra whitespace"""
        csv_content = """question,answer,points
 1 , A ,2
2, B ,2
 3 ,photosynthesis, 3 """
        
        csv_path = os.path.join(self.temp_dir, 'whitespace.csv')
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        
        answer_key = load_answer_key(csv_path)
        
        self.assertEqual(answer_key['1']['answer'], 'A')
        self.assertEqual(answer_key['2']['answer'], 'B')
        self.assertEqual(answer_key['3']['answer'], 'photosynthesis')
    
    def test_load_csv_missing_points(self):
        """Test loading CSV with missing points column"""
        csv_content = """question,answer
1,A
2,B
3,photosynthesis"""
        
        csv_path = os.path.join(self.temp_dir, 'no_points.csv')
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        
        answer_key = load_answer_key(csv_path)
        
        # Should default to 1 point each
        self.assertEqual(answer_key['1']['points'], 1)
        self.assertEqual(answer_key['2']['points'], 1)
        self.assertEqual(answer_key['3']['points'], 1)
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file"""
        with self.assertRaises(FileNotFoundError):
            load_answer_key('nonexistent.csv')
    
    def test_load_malformed_csv(self):
        """Test loading malformed CSV"""
        csv_content = """question,answer,points
1,A,not_a_number
2,B,2"""
        
        csv_path = os.path.join(self.temp_dir, 'malformed.csv')
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        
        with self.assertRaises(ValueError):
            load_answer_key(csv_path)


class TestGraderAdvanced(unittest.TestCase):
    """Test advanced grader functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.answer_key = {
            '1': {'answer': 'A', 'points': 1},
            '2': {'answer': 'photosynthesis', 'points': 2},
            '3': {'answer': 'DNA', 'points': 1}
        }
        self.grader = Grader(self.answer_key)
    
    def test_text_answer_similarity_scoring(self):
        """Test similarity-based scoring for text answers"""
        # Very similar answer
        result = self.grader._score_text_answer('photo synthesis', 'photosynthesis', 2)
        self.assertGreater(result['points_earned'], 1.5)  # Should get high partial credit
        
        # Moderately similar answer
        result = self.grader._score_text_answer('photosyn', 'photosynthesis', 2)
        self.assertGreater(result['points_earned'], 0)
        self.assertLess(result['points_earned'], 2)
        
        # Completely wrong answer
        result = self.grader._score_text_answer('respiration', 'photosynthesis', 2)
        self.assertEqual(result['points_earned'], 0)
    
    def test_extract_mc_letter(self):
        """Test multiple choice letter extraction"""
        self.assertEqual(self.grader._extract_mc_letter('A'), 'A')
        self.assertEqual(self.grader._extract_mc_letter('a'), 'A')
        self.assertEqual(self.grader._extract_mc_letter('The answer is B'), 'B')
        self.assertEqual(self.grader._extract_mc_letter('c)'), 'C')
        self.assertEqual(self.grader._extract_mc_letter('Answer: D'), 'D')
        self.assertEqual(self.grader._extract_mc_letter('no letter here'), '')
        self.assertEqual(self.grader._extract_mc_letter('123'), '')
    
    def test_custom_similarity_threshold(self):
        """Test custom similarity threshold"""
        strict_grader = Grader(self.answer_key, similarity_threshold=0.9)
        lenient_grader = Grader(self.answer_key, similarity_threshold=0.5)
        
        student_answer = {'2': 'photo synthesis'}
        
        strict_result = strict_grader.grade_answers(student_answer)
        lenient_result = lenient_grader.grade_answers(student_answer)
        
        # Lenient grader should give more credit
        self.assertGreaterEqual(
            lenient_result['question_scores']['2']['points_earned'],
            strict_result['question_scores']['2']['points_earned']
        )
    
    def test_batch_grading(self):
        """Test batch grading functionality"""
        students_data = [
            {
                'student_id': 'student_001',
                'answers': {'1': 'A', '2': 'photosynthesis', '3': 'DNA'}
            },
            {
                'student_id': 'student_002', 
                'answers': {'1': 'B', '2': 'respiration', '3': 'RNA'}
            },
            {
                'student_id': 'student_003',
                'answers': {'1': 'A', '2': 'photo synthesis', '3': 'DNA'}
            }
        ]
        
        results = self.grader.grade_batch(students_data)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['student_id'], 'student_001')
        self.assertEqual(results[1]['student_id'], 'student_002')
        self.assertEqual(results[2]['student_id'], 'student_003')
        
        # First student should have perfect score
        self.assertEqual(results[0]['percentage'], 100.0)
        
        # Second student should have lowest score
        self.assertLess(results[1]['percentage'], results[0]['percentage'])
        self.assertLess(results[1]['percentage'], results[2]['percentage'])


class TestGraderExport(unittest.TestCase):
    """Test export functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.answer_key = {'1': {'answer': 'A', 'points': 1}}
        self.grader = Grader(self.answer_key)
        self.temp_dir = tempfile.mkdtemp()
        
        self.sample_results = [
            {
                'student_id': 'student_001',
                'raw_score': 1,
                'total_possible': 1,
                'percentage': 100.0,
                'correct_count': 1,
                'total_questions': 1,
                'question_scores': {
                    '1': {
                        'student_answer': 'A',
                        'correct_answer': 'A',
                        'points_earned': 1,
                        'max_points': 1,
                        'is_correct': True,
                        'similarity_score': 1.0
                    }
                }
            }
        ]
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_export_to_csv(self):
        """Test exporting results to CSV"""
        output_path = os.path.join(self.temp_dir, 'results.csv')
        
        self.grader.export_results(self.sample_results, output_path, format='csv')
        
        self.assertTrue(os.path.exists(output_path))
        
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['student_id'], 'student_001')
        self.assertEqual(rows[0]['percentage'], '100.0')
    
    def test_export_to_json(self):
        """Test exporting results to JSON"""
        output_path = os.path.join(self.temp_dir, 'results.json')
        
        self.grader.export_results(self.sample_results, output_path, format='json')
        
        self.assertTrue(os.path.exists(output_path))
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['student_id'], 'student_001')
        self.assertEqual(data[0]['percentage'], 100.0)
    
    def test_export_invalid_format(self):
        """Test export with invalid format"""
        output_path = os.path.join(self.temp_dir, 'results.txt')
        
        with self.assertRaises(ValueError):
            self.grader.export_results(self.sample_results, output_path, format='txt')


class TestGraderEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.answer_key = {'1': {'answer': 'test', 'points': 1}}
        self.grader = Grader(self.answer_key)
    
    def test_unicode_answers(self):
        """Test handling of unicode characters"""
        unicode_key = {'1': {'answer': 'café', 'points': 1}}
        unicode_grader = Grader(unicode_key)
        
        result = unicode_grader.grade_answers({'1': 'café'})
        self.assertTrue(result['question_scores']['1']['is_correct'])
        
        result = unicode_grader.grade_answers({'1': 'cafe'})
        self.assertGreater(result['question_scores']['1']['similarity_score'], 0.8)
    
    def test_very_long_answers(self):
        """Test handling of very long answers"""
        long_answer = 'A' * 1000
        long_key = {'1': {'answer': long_answer, 'points': 1}}
        long_grader = Grader(long_key)
        
        result = long_grader.grade_answers({'1': long_answer})
        self.assertTrue(result['question_scores']['1']['is_correct'])
    
    def test_empty_answer_key(self):
        """Test grader with empty answer key"""
        empty_grader = Grader({})
        
        result = empty_grader.grade_answers({'1': 'test'})
        
        self.assertEqual(result['raw_score'], 0)
        self.assertEqual(result['total_possible'], 0)
        self.assertEqual(result['percentage'], 0.0)
    
    def test_none_values_handling(self):
        """Test handling of None values"""
        result = self.grader.grade_answers({'1': None})
        self.assertEqual(result['question_scores']['1']['student_answer'], '')
        self.assertFalse(result['question_scores']['1']['is_correct'])
    
    def test_numeric_question_ids(self):
        """Test handling of numeric vs string question IDs"""
        numeric_key = {1: {'answer': 'test', 'points': 1}}
        numeric_grader = Grader(numeric_key)
        
        # Should work with both string and numeric keys
        result1 = numeric_grader.grade_answers({'1': 'test'})
        result2 = numeric_grader.grade_answers({1: 'test'})
        
        self.assertTrue(result1['question_scores']['1']['is_correct'])
        self.assertTrue(result2['question_scores'][1]['is_correct'])


class TestGraderConfiguration(unittest.TestCase):
    """Test grader configuration options"""
    
    def test_default_configuration(self):
        """Test default grader configuration"""
        answer_key = {'1': {'answer': 'test', 'points': 1}}
        grader = Grader(answer_key)
        
        self.assertEqual(grader.similarity_threshold, 0.8)
        self.assertTrue(grader.case_sensitive is False)
    
    def test_custom_configuration(self):
        """Test custom grader configuration"""
        answer_key = {'1': {'answer': 'test', 'points': 1}}
        grader = Grader(
            answer_key, 
            similarity_threshold=0.9,
            case_sensitive=True
        )
        
        self.assertEqual(grader.similarity_threshold, 0.9)
        self.assertTrue(grader.case_sensitive)
    
    def test_invalid_threshold(self):
        """Test invalid similarity threshold"""
        answer_key = {'1': {'answer': 'test', 'points': 1}}
        
        with self.assertRaises(ValueError):
            Grader(answer_key, similarity_threshold=1.5)  # > 1.0
        
        with self.assertRaises(ValueError):
            Grader(answer_key, similarity_threshold=-0.1)  # < 0.0


if __name__ == '__main__':
    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestGrader,
        TestGraderStatistics,
        TestLoadAnswerKey,
        TestGraderAdvanced,
        TestGraderExport,
        TestGraderEdgeCases,
        TestGraderConfiguration
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)