"""
Image Preprocessing Module
Handles image orientation correction, deskewing, and quality enhancement
"""

import cv2
import numpy as np
import logging
from PIL import Image, ImageEnhance
from scipy import ndimage
import math


class ImagePreprocessor:
    """Handles preprocessing of scanned answer sheets"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_image(self, image_path):
        """
        Complete image preprocessing pipeline
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Processed image or None if processing fails
        """
        try:
            # Load image
            image = self._load_image(image_path)
            if image is None:
                return None
            
            # Detect and correct orientation
            image = self._correct_orientation(image)
            
            # Deskew the image
            image = self._deskew_image(image)
            
            # Enhance image quality
            image = self._enhance_quality(image)
            
            # Convert back to format suitable for OCR
            image = self._prepare_for_ocr(image)
            
            self.logger.debug(f"Successfully preprocessed image: {image_path}")
            return image
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def _load_image(self, image_path):
        """Load image from file"""
        try:
            # Try loading with OpenCV first
            image = cv2.imread(image_path)
            if image is not None:
                return image
            
            # Fallback to PIL
            pil_image = Image.open(image_path)
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {str(e)}")
            return None
    
    def _correct_orientation(self, image):
        """
        Detect and correct image orientation using contour analysis
        """
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return image
            
            # Find the largest contour (likely the paper)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            
            # Determine rotation angle
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            
            # Rotate image if needed
            if abs(angle) > 1:  # Only rotate if angle is significant
                self.logger.debug(f"Correcting orientation by {angle:.1f} degrees")
                image = self._rotate_image(image, angle)
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Orientation correction failed: {str(e)}")
            return image
    
    def _deskew_image(self, image):
        """
        Deskew image using line detection
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 100, 200, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                  minLineLength=100, maxLineGap=10)
            
            if lines is None:
                return image
            
            # Calculate angles of detected lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            if not angles:
                return image
            
            # Calculate median angle for deskewing
            median_angle = np.median(angles)
            
            # Only deskew if angle is significant
            if abs(median_angle) > 0.5:
                self.logger.debug(f"Deskewing by {median_angle:.1f} degrees")
                image = self._rotate_image(image, -median_angle)
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Deskewing failed: {str(e)}")
            return image
    
    def _rotate_image(self, image, angle):
        """Rotate image by given angle"""
        try:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # Get rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calculate new dimensions
            cos_val = abs(rotation_matrix[0, 0])
            sin_val = abs(rotation_matrix[0, 1])
            new_width = int((height * sin_val) + (width * cos_val))
            new_height = int((height * cos_val) + (width * sin_val))
            
            # Adjust rotation matrix for new center
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            # Perform rotation
            rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
            
        except Exception as e:
            self.logger.error(f"Image rotation failed: {str(e)}")
            return image
    
    def _enhance_quality(self, image):
        """
        Enhance image quality for better OCR
        """
        try:
            # Convert to PIL for enhancement
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Enhance contrast
            contrast_enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = contrast_enhancer.enhance(1.2)
            
            # Enhance sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = sharpness_enhancer.enhance(1.1)
            
            # Convert back to OpenCV format
            enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Apply noise reduction
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Quality enhancement failed: {str(e)}")
            return image
    
    def _prepare_for_ocr(self, image):
        """
        Prepare image for OCR processing
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding for better text recognition
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up the image
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"OCR preparation failed: {str(e)}")
            # Return grayscale version as fallback
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def resize_image(self, image, max_width=2000, max_height=2000):
        """
        Resize image if it's too large while maintaining aspect ratio
        """
        try:
            height, width = image.shape[:2]
            
            if width <= max_width and height <= max_height:
                return image
            
            # Calculate scaling factor
            scale_w = max_width / width
            scale_h = max_height / height
            scale = min(scale_w, scale_h)
            
            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_width, new_height), 
                               interpolation=cv2.INTER_AREA)
            
            self.logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            return resized
            
        except Exception as e:
            self.logger.error(f"Image resizing failed: {str(e)}")
            return image