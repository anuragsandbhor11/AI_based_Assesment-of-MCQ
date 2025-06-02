"""
Unit tests for OCR processing module
"""

import unittest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ocr import OCRProcessor


class TestOCRProcessor(unittest.TestCase):
    """Test cases for OCR processor"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock Tesseract to avoid dependency issues in tests
        with patch('ocr.pytesseract.image_to_string') as mock_ocr, \
             patch('ocr.pytesseract.image_to_data') as mock_data:
            mock_ocr.return_value = "Test OCR"
            mock_data.return_value = {'conf': [95, 90, 88]}
            self.ocr_processor = OCRProcessor()
    
    def test_initialization(self):
        """Test OCR processor initialization"""
        self.assertIsNotNone(self.ocr_processor.tesseract_config)
        self.assertIn('--oem', self.ocr_processor.tesseract_config)
    
    @patch('ocr.pytesseract.image_to_string')
    def test_extract_text_basic(self, mock_ocr):
        """Test basic text extraction"""
        # Mock OCR response
        mock_ocr.return_value = "1. A\n2. B\n3. photosynthesis"
        
        # Create test image
        test_image = np.ones((100, 200), dtype=np.uint8) * 255
        
        # Extract text
        result = self.ocr_processor.extract_text(test_image)
        
        # Verify
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        mock_ocr.assert_called_once()
    
    @patch('ocr.pytesseract.image_to_string')
    def test_extract_text_color_image(self, mock_ocr):
        """Test text extraction from color image"""
        mock_ocr.return_value = "Test text"
        
        # Create color test image
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        
        result = self.ocr_processor.extract_text(test_image)
        
        self.assertIsInstance(result, str)
        mock_ocr.assert_called_once()
    
    @patch('ocr.pytesseract.image_to_string')
    def test_extract_text_failure(self, mock_ocr):
        """Test handling of OCR failure"""
        mock_ocr.side_effect = Exception("OCR failed")
        
        test_image = np.ones((100, 200), dtype=np.uint8) * 255
        
        result = self.ocr_processor.extract_text(test_image)
        
        self.assertEqual(result, "")
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        # Test with messy text
        messy_text = "  1.   A  \n\n  2.    B   \n  3. photosynthesis  "
        cleaned = self.ocr_processor._clean_text(messy_text)
        
        self.assertNotIn('\n\n', cleaned)
        self.assertFalse(cleaned.startswith(' '))
        self.assertFalse(cleaned.endswith(' '))
    
    def test_clean_text_empty(self):
        """Test cleaning empty text"""
        result = self.ocr_processor._clean_text("")
        self.assertEqual(result, "")
        
        result = self.ocr_processor._clean_text(None)
        self.assertEqual(result, "")
    
    def test_parse_answers_structured(self):
        """Test parsing structured answers"""
        text = "1. A\n2. B\n3. photosynthesis\n4. C\n5. mitochondria"
        
        answers = self.ocr_processor.parse_answers(text)
        
        self.assertIsInstance(answers, dict)
        self.assertEqual(answers.get('1'), 'A')
        self.assertEqual(answers.get('2'), 'B')
        self.assertEqual(answers.get('3'), 'photosynthesis')
        self.assertEqual(answers.get('4'), 'C')
        self.assertEqual(answers.get('5'), 'mitochondria')
    
    def test_parse_answers_alternative_format(self):
        """Test parsing answers with different formats"""
        text = "1) A\n2: B\n3. photosynthesis"
        
        answers = self.ocr_processor.parse_answers(text)
        
        self.assertIn('1', answers)
        self.assertIn('2', answers)
        self.assertIn('3', answers)
    
    def test_parse_answers_empty(self):
        """Test parsing empty text"""
        answers = self.ocr_processor.parse_answers("")
        self.assertEqual(answers, {})
        
        answers = self.ocr_processor.parse_answers(None)
        self.assertEqual(answers, {})
    
    def test_parse_answers_unstructured(self):
        """Test fallback parsing for unstructured text"""
        text = "A B photosynthesis C mitochondria"
        
        answers = self.ocr_processor.parse_answers(text)
        
        # Should have some answers from fallback parsing
        self.assertIsInstance(answers, dict)
    
    def test_normalize_answer_multiple_choice(self):
        """Test answer normalization for multiple choice"""
        # Single letters should be uppercase
        self.assertEqual(self.ocr_processor._normalize_answer('a'), 'A')
        self.assertEqual(self.ocr_processor._normalize_answer('B'), 'B')
        self.assertEqual(self.ocr_processor._normalize_answer(' c '), 'C')
    
    def test_normalize_answer_text(self):
        """Test answer normalization for text answers"""
        # Text answers should be lowercase and cleaned
        result = self.ocr_processor._normalize_answer('Photosynthesis!')
        self.assertEqual(result, 'photosynthesis')
        
        result = self.ocr_processor._normalize_answer('  MITOCHONDRIA  ')
        self.assertEqual(result, 'mitochondria')
    
    def test_normalize_answer_empty(self):
        """Test normalizing empty answers"""
        self.assertEqual(self.ocr_processor._normalize_answer(''), '')
        self.assertEqual(self.ocr_processor._normalize_answer(None), '')
    
    def test_preprocess_for_ocr(self):
        """Test OCR-specific preprocessing"""
        # Create small test image
        test_image = np.ones((50, 100), dtype=np.uint8) * 255
        
        processed = self.ocr_processor._preprocess_for_ocr(test_image)
        
        # Should be resized if too small
        self.assertGreaterEqual(processed.shape[0], 300)
        self.assertGreaterEqual(processed.shape[1], 300)
    
    @patch('ocr.pytesseract.image_to_data')
    def test_get_ocr_confidence(self, mock_data):
        """Test OCR confidence calculation"""
        # Mock confidence data
        mock_data.return_value = {
            'conf': ['95', '90', '88', '92', '-1', '85']
        }
        
        test_image = np.ones((100, 200), dtype=np.uint8) * 255
        
        confidence = self.ocr_processor.get_ocr_confidence(test_image)
        
        # Should calculate average of valid confidence scores
        expected = (95 + 90 + 88 + 92 + 85) / 5
        self.assertEqual(confidence, expected)
    
    @patch('ocr.pytesseract.image_to_data')
    def test_get_ocr_confidence_no_data(self, mock_data):
        """Test OCR confidence with no valid data"""
        mock_data.return_value = {'conf': ['-1', '-1', '-1']}
        
        test_image = np.ones((100, 200), dtype=np.uint8) * 255
        
        confidence = self.ocr_processor.get_ocr_confidence(test_image)
        
        self.assertEqual(confidence, 0.0)
    
    @patch('ocr.pytesseract.image_to_string')
    def test_extract_with_regions(self, mock_ocr):
        """Test region-based text extraction"""
        mock_ocr.return_value = "Region text"
        
        test_image = np.ones((200, 300), dtype=np.uint8) * 255
        regions = [(10, 10, 50, 50), (100, 100, 50, 50)]
        
        results = self.ocr_processor.extract_with_regions(test_image, regions)
        
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 2)
        self.assertIn('0', results)
        self.assertIn('1', results)
    
    @patch('ocr.pytesseract.image_to_string')
    def test_extract_with_regions_no_regions(self, mock_ocr):
        """Test region extraction with no regions specified"""
        mock_ocr.return_value = "Full image text"
        
        test_image = np.ones((200, 300), dtype=np.uint8) * 255
        
        results = self.ocr_processor.extract_with_regions(test_image, None)
        
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 1)
        self.assertIn('0', results)
    
    def test_fallback_parsing(self):
        """Test fallback parsing method"""
        text = "A B photosynthesis C mitochondria D glucose"
        
        answers = self.ocr_processor._fallback_parsing(text)
        
        self.assertIsInstance(answers, dict)
        # Should extract some answers
        self.assertGreater(len(answers), 0)


class TestOCRIntegration(unittest.TestCase):
    """Integration tests for OCR functionality"""
    
    @patch('ocr.pytesseract.image_to_string')
    def test_full_pipeline(self, mock_ocr):
        """Test complete OCR pipeline"""
        # Mock realistic OCR output
        mock_ocr.return_value = """
        Name: John Doe
        1. A
        2. B
        3. photosynthesis
        4. C
        5. mitochondria
        """
        
        with patch('ocr.pytesseract.image_to_data') as mock_data:
            mock_data.return_value = {'conf': [95, 90, 88]}
            
            processor = OCRProcessor()
            
            # Create test image
            test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
            
            # Extract and parse
            text = processor.extract_text(test_image)
            answers = processor.parse_answers(text)
            
            # Verify results
            self.assertIsInstance(text, str)
            self.assertIsInstance(answers, dict)
            self.assertIn('1', answers)
            self.assertIn('3', answers)


if __name__ == '__main__':
    unittest.main()