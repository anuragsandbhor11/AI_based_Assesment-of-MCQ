

import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            logging.info(f"Logging to file: {log_file}")
        except Exception as e:
            logging.warning(f"Could not set up file logging: {e}")



def validate_input_directory(directory_path):
    """
    Validate that the input directory exists and contains image files
    
    Args:
        directory_path (str): Path to the directory to validate
        
    Returns:
        bool: True if directory is valid, False otherwise
        
    Raises:
        ValueError: If directory doesn't exist or is empty
    """
    if not directory_path:
        raise ValueError("Directory path cannot be empty")
    
    path = Path(directory_path)
    
    if not path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")
    
    # Check for image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
    image_files = []
    
    for file_path in path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    if not image_files:
        raise ValueError(f"No image files found in directory: {directory_path}")
    
    logging.info(f"Found {len(image_files)} image files in {directory_path}")
    return True

def load_answer_key(csv_path):
    """
    Load answer key from CSV file
    
    Args:
        csv_path (str): Path to the CSV file containing answer key
        
    Returns:
        dict: Dictionary mapping question numbers to correct answers
    """
    import pandas as pd
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Answer key file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        # Handle different possible column names for answers
        if 'answer' in df.columns:
            answer_key = dict(zip(df['question'], df['answer']))
        elif 'correct_answer' in df.columns:
            answer_key = dict(zip(df['question'], df['correct_answer']))
        else:
            raise ValueError("CSV must contain either 'answer' or 'correct_answer' column")
        logging.info(f"Loaded {len(answer_key)} answers from {csv_path}")
        return answer_key
    except Exception as e:
        raise ValueError(f"Error loading answer key: {str(e)}")


def ensure_directory_exists(directory_path):
    """
    Ensure a directory exists, create if it doesn't
    
    Args:
        directory_path (str): Path to the directory
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def get_image_files(directory: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) -> List[str]:
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not os.path.isdir(directory):
        raise ValueError(f"Path is not a directory: {directory}")
    
    image_files = []
    
    try:
        for filename in os.listdir(directory):
            if filename.lower().endswith(extensions):
                full_path = os.path.join(directory, filename)
                if os.path.isfile(full_path):
                    image_files.append(full_path)
        
        # Sort files for consistent processing order
        image_files.sort()
        
        logging.info(f"Found {len(image_files)} image files in {directory}")
        return image_files
        
    except PermissionError as e:
        logging.error(f"Permission denied accessing directory {directory}: {e}")
        raise
    except Exception as e:
        logging.error(f"Error reading directory {directory}: {e}")
        raise


def validate_file_paths(input_dir: str, answer_key: str, output_path: str) -> None:
    
    # Check input directory
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Check answer key file
    if not os.path.exists(answer_key):
        raise FileNotFoundError(f"Answer key file not found: {answer_key}")
    
    if not os.path.isfile(answer_key):
        raise ValueError(f"Answer key path is not a file: {answer_key}")
    
    # Check if we can read the answer key
    try:
        with open(answer_key, 'r') as f:
            pass
    except PermissionError:
        raise PermissionError(f"Cannot read answer key file: {answer_key}")
    
    # Check output directory
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Created output directory: {output_dir}")
        except PermissionError:
            raise PermissionError(f"Cannot create output directory: {output_dir}")
    
    # Check if we can write to output location
    try:
        test_path = os.path.join(output_dir, '.write_test')
        with open(test_path, 'w') as f:
            f.write("test")
        os.remove(test_path)
    except PermissionError:
        raise PermissionError(f"Cannot write to output directory: {output_dir}")


def extract_student_id_from_filename(filename: str) -> str:
    """
    Extract student ID from filename.
    
    Args:
        filename: Image filename
        
    Returns:
        Student ID string
    """
    # Remove extension and directory path
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Try common patterns
    patterns = [
        r'student[_-]?(\d+)',
        r'sheet[_-]?(\d+)',
        r'test[_-]?(\d+)',
        r'exam[_-]?(\d+)',
        r'(\d+)'
    ]
    
    import re
    for pattern in patterns:
        match = re.search(pattern, base_name, re.IGNORECASE)
        if match:
            return f"student_{match.group(1).zfill(3)}"
    
    # If no pattern matches, use filename hash
    hash_obj = hashlib.md5(base_name.encode())
    return f"student_{hash_obj.hexdigest()[:6]}"


def save_json_report(data: Dict[str, Any], output_path: str) -> None:
   
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logging.info(f"Report saved to: {output_path}")
        
    except Exception as e:
        logging.error(f"Failed to save report to {output_path}: {e}")
        raise


def create_metadata(total_images: int, processed_images: int, failed_images: int, 
                   total_questions: int) -> Dict[str, Any]:
    
    return {
        'processed_date': datetime.now(timezone.utc).isoformat(),
        'total_images': total_images,
        'successfully_processed': processed_images,
        'failed_images': failed_images,
        'total_questions': total_questions,
        'processing_success_rate': round((processed_images / total_images * 100), 2) if total_images > 0 else 0
    }


def validate_csv_format(csv_path: str) -> bool:
   
    required_columns = {'question_id', 'correct_answer', 'points'}
    
    try:
        import csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Check if required columns exist
            if not reader.fieldnames:
                raise ValueError("CSV file is empty or has no headers")
            
            csv_columns = set(col.strip().lower() for col in reader.fieldnames)
            required_lower = set(col.lower() for col in required_columns)
            
            if not required_lower.issubset(csv_columns):
                missing = required_lower - csv_columns
                raise ValueError(f"CSV missing required columns: {missing}")
            
            # Check if file has data rows
            row_count = sum(1 for _ in reader)
            if row_count == 0:
                raise ValueError("CSV file has no data rows")
            
            logging.info(f"CSV validation passed: {row_count} questions found")
            return True
            
    except Exception as e:
        logging.error(f"CSV validation failed: {e}")
        raise


def get_file_size_mb(file_path: str) -> float:
    
    try:
        size_bytes = os.path.getsize(file_path)
        return round(size_bytes / (1024 * 1024), 2)
    except OSError:
        return 0.0


def ensure_directory_exists(directory: str) -> None:
    
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        logging.error(f"Could not create directory {directory}: {e}")
        raise


def clean_text(text: str) -> str:
   
    if not text:
        return ""
    
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-printable characters except common ones
    text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def format_duration(seconds: float) -> str:
    
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def log_processing_summary(total_files: int, successful: int, failed: int, 
                          duration: float) -> None:
    
    success_rate = (successful / total_files * 100) if total_files > 0 else 0
    
    logging.info("=" * 50)
    logging.info("PROCESSING SUMMARY")
    logging.info("=" * 50)
    logging.info(f"Total files: {total_files}")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Success rate: {success_rate:.1f}%")
    logging.info(f"Processing time: {format_duration(duration)}")
    logging.info("=" * 50)