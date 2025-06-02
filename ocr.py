

import cv2
import pytesseract
import re
import logging
from PIL import Image
import numpy as np


class OCRProcessor:
    """Handles OCR text extraction and answer parsing"""
    
    def __init__(self, tesseract_config=None):
        self.logger = logging.getLogger(__name__)
        
        # Default Tesseract configuration for better accuracy
        self.tesseract_config = tesseract_config or r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:;!?()-\n '
        
        # Test Tesseract installation
        self._test_tesseract()
    
    def _test_tesseract(self):
        """Test if Tesseract is properly installed"""
        try:
            # Create a simple test image
            test_img = np.ones((100, 200, 3), dtype=np.uint8) * 255
            cv2.putText(test_img, 'TEST', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Try OCR
            result = pytesseract.image_to_string(test_img)
            self.logger.info("Tesseract OCR is working properly")
            
        except Exception as e:
            self.logger.error(f"Tesseract not properly configured: {str(e)}")
            raise Exception("Tesseract OCR not available. Please install Tesseract and ensure it's in your PATH.")
    
    def extract_text(self, image):
        """
        Extract text from preprocessed image
        
        Args:
            image (numpy.ndarray): Preprocessed image
            
        Returns:
            str: Extracted text or empty string if extraction fails
        """
        try:
            # Ensure image is in correct format
            if len(image.shape) == 3:
                # Convert to grayscale if needed
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply additional preprocessing for OCR
            processed_image = self._preprocess_for_ocr(image)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(processed_image, config=self.tesseract_config)
            
            # Clean the extracted text
            cleaned_text = self._clean_text(text)
            
            self.logger.debug(f"Extracted text length: {len(cleaned_text)} characters")
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"OCR text extraction failed: {str(e)}")
            return ""
    
    def _preprocess_for_ocr(self, image):
        """
        Additional preprocessing specifically for OCR
        """
        try:
            # Resize image if too small (OCR works better on larger images)
            height, width = image.shape
            if height < 300 or width < 300:
                scale_factor = max(300 / height, 300 / width)
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Apply Gaussian blur to reduce noise
            image = cv2.GaussianBlur(image, (1, 1), 0)
            
            # Apply dilation and erosion to make text clearer
            kernel = np.ones((1, 1), np.uint8)
            image = cv2.dilate(image, kernel, iterations=1)
            image = cv2.erode(image, kernel, iterations=1)
            
            return image
            
        except Exception as e:
            self.logger.warning(f"OCR preprocessing failed: {str(e)}")
            return image
    
    def _clean_text(self, text):
        """
        Clean extracted text by removing artifacts and normalizing
        """
        if not text:
            return ""
        
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common OCR artifacts
        text = re.sub(r'[|\\/_~`]', '', text)
        
        # Fix common OCR mistakes
        replacements = {
            '0': 'O',  # Sometimes O is recognized as 0
            '1': 'I',  # Sometimes I is recognized as 1
            '5': 'S',  # Sometimes S is recognized as 5
        }
        
        # Apply replacements only in likely letter contexts
        # This is a simple approach - could be made more sophisticated
        for old, new in replacements.items():
            # Replace only if surrounded by letters (indicating it's likely a letter)
            text = re.sub(f'(?<=[A-Za-z]){old}(?=[A-Za-z])', new, text)
        
        return text
    
    def parse_answers(self, text):
        """
        Parse answers from extracted text
        
        Args:
            text (str): Extracted text from OCR
            
        Returns:
            dict: Dictionary mapping question numbers to answers
        """
        answers = {}
        
        if not text:
            return answers
        
        try:
            # Split text into lines for processing
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # Pattern 1: Question number followed by answer (e.g., "1. A" or "1) photosynthesis")
            pattern1 = r'^(\d+)[\.\)\:]?\s*([A-Za-z][A-Za-z0-9\s]*|[A-D])$'
            
            # Pattern 2: Standalone answers with question numbers nearby
            pattern2 = r'(\d+)[^\w]*([A-D]|[A-Za-z][A-Za-z0-9\s]+)'
            
            # Process line by line
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Try pattern 1
                match = re.match(pattern1, line, re.IGNORECASE)
                if match:
                    q_num = int(match.group(1))
                    answer = match.group(2).strip()
                    answers[str(q_num)] = self._normalize_answer(answer)
                    continue
                
                # Try pattern 2
                matches = re.findall(pattern2, line, re.IGNORECASE)
                for match in matches:
                    q_num = int(match[0])
                    answer = match[1].strip()
                    answers[str(q_num)] = self._normalize_answer(answer)
            
            # Fallback: Look for isolated answers and try to map them
            if not answers:
                answers = self._fallback_parsing(text)
            
            self.logger.debug(f"Parsed {len(answers)} answers")
            return answers
            
        except Exception as e:
            self.logger.error(f"Answer parsing failed: {str(e)}")
            return {}
    
    def _fallback_parsing(self, text):
        """
        Fallback parsing method for when structured parsing fails
        """
        answers = {}
        
        try:
            # Look for single letters (multiple choice answers)
            mc_answers = re.findall(r'\b([A-D])\b', text)
            
            # Look for longer text answers
            words = text.split()
            text_answers = []
            
            for word in words:
                # Skip short words and numbers
                if len(word) > 3 and not word.isdigit() and word.isalpha():
                    text_answers.append(word)
            
            # Assign answers sequentially (this is a best guess)
            question_num = 1
            
            # First assign multiple choice answers
            for answer in mc_answers:
                if question_num <= 10:  # Assume max 10 questions
                    answers[str(question_num)] = answer
                    question_num += 1
            
            # Then assign text answers
            for answer in text_answers:
                if question_num <= 10:
                    answers[str(question_num)] = answer
                    question_num += 1
            
            return answers
            
        except Exception as e:
            self.logger.error(f"Fallback parsing failed: {str(e)}")
            return {}
    
    def _normalize_answer(self, answer):
        """
        Normalize answer format
        """
        if not answer:
            return ""
        
        answer = answer.strip()
        
        # If it's a single letter, make it uppercase (multiple choice)
        if len(answer) == 1 and answer.isalpha():
            return answer.upper()
        
        # For text answers, clean and lowercase for consistent comparison
        answer = re.sub(r'[^\w\s]', '', answer)  # Remove punctuation
        answer = answer.lower().strip()
        
        return answer
    
    def get_ocr_confidence(self, image):
        """
        Get OCR confidence score for the image
        
        Args:
            image (numpy.ndarray): Image to analyze
            
        Returns:
            float: Average confidence score (0-100)
        """
        try:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Get detailed OCR data with confidence scores
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Extract confidence scores for words (filter out -1 values)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                self.logger.debug(f"Average OCR confidence: {avg_confidence:.1f}%")
                return avg_confidence
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Could not calculate OCR confidence: {str(e)}")
            return 0.0
    
    def extract_with_regions(self, image, regions=None):
        """
        Extract text from specific regions of interest
        
        Args:
            image (numpy.ndarray): Input image
            regions (list): List of (x, y, width, height) tuples for regions
            
        Returns:
            dict: Dictionary mapping region index to extracted text
        """
        results = {}
        
        if regions is None:
            # If no regions specified, use the whole image
            return {"0": self.extract_text(image)}
        
        try:
            for i, (x, y, w, h) in enumerate(regions):
                # Extract region
                region = image[y:y+h, x:x+w]
                
                # Extract text from region
                text = self.extract_text(region)
                results[str(i)] = text
                
            return results
            
        except Exception as e:
            self.logger.error(f"Region-based OCR failed: {str(e)}")
            return {}