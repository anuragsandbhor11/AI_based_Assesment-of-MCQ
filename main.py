#!/usr/bin/env python3
"""
Offline AI-Powered Assessment Pipeline
Entry point script for processing scanned answer sheets
"""

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from src.preprocessor import ImagePreprocessor
from src.ocr import OCRProcessor
from src.grader import Grader
from src.utils import setup_logging, validate_input_directory, load_answer_key


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Offline AI-Powered Assessment Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py --input_dir ./answer_sheets --key answers.csv --output report.json
  python main.py --input_dir ./sheets --key answers.csv --output results.json --verbose --log_file processing.log
        '''
    )
    
    parser.add_argument(
        '--input_dir', 
        required=True,
        help='Path to directory containing answer sheet images'
    )
    parser.add_argument(
        '--key', 
        required=True,
        help='Path to CSV file with correct answers'
    )
    parser.add_argument(
        '--output', 
        required=True,
        help='Path for output JSON report'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable detailed logging output'
    )
    parser.add_argument(
        '--log_file', 
        default='processing.log',
        help='Specify custom log file location'
    )
    
    return parser.parse_args()


def process_images(input_dir, answer_key, verbose=False):
    """
    Main processing pipeline
    
    Args:
        input_dir (str): Directory containing images
        answer_key (dict): Answer key data
        verbose (bool): Enable verbose logging
    
    Returns:
        dict: Processing results
    """
    
    # Initialize processors
    preprocessor = ImagePreprocessor()
    ocr_processor = OCRProcessor()
    grader = Grader(answer_key)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    logging.info(f"Found {len(image_files)} image files to process")
    
    results = {
        'metadata': {
            'processed_date': datetime.now().isoformat(),
            'total_images': len(image_files),
            'successfully_processed': 0,
            'failed_images': 0,
            'total_questions': len(answer_key)
        },
        'individual_results': [],
        'failed_files': []
    }
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        try:
            logging.info(f"Processing {i}/{len(image_files)}: {image_path.name}")
            
            # Step 1: Preprocess image
            if verbose:
                logging.info(f"  - Preprocessing image...")
            processed_image = preprocessor.process_image(str(image_path))
            
            if processed_image is None:
                raise Exception("Image preprocessing failed")
            
            # Step 2: Extract text using OCR
            if verbose:
                logging.info(f"  - Extracting text with OCR...")
            extracted_text = ocr_processor.extract_text(processed_image)
            
            if not extracted_text:
                raise Exception("OCR text extraction failed")
            
            # Step 3: Parse answers from extracted text
            if verbose:
                logging.info(f"  - Parsing answers...")
            parsed_answers = ocr_processor.parse_answers(extracted_text)
            
            # Step 4: Grade the answers
            if verbose:
                logging.info(f"  - Grading answers...")
            score_result = grader.grade_answers(parsed_answers)
            
            # Create student result
            student_id = f"student_{image_path.stem}"
            result = {
                'student_id': student_id,
                'image_file': image_path.name,
                'raw_score': score_result['raw_score'],
                'total_possible': score_result['total_possible'],
                'percentage': score_result['percentage'],
                'answers': parsed_answers
            }
            
            results['individual_results'].append(result)
            results['metadata']['successfully_processed'] += 1
            
            if verbose:
                logging.info(f"  - Score: {score_result['raw_score']}/{score_result['total_possible']} ({score_result['percentage']:.1f}%)")
            
        except Exception as e:
            logging.error(f"Failed to process {image_path.name}: {str(e)}")
            if verbose:
                logging.error(traceback.format_exc())
            
            results['failed_files'].append({
                'filename': image_path.name,
                'error': str(e)
            })
            results['metadata']['failed_images'] += 1
    
    # Calculate class statistics
    if results['individual_results']:
        logging.info("Calculating class statistics...")
        results['class_statistics'] = grader.calculate_statistics(results['individual_results'])
    else:
        logging.warning("No successful results to calculate statistics")
        results['class_statistics'] = {}
    
    return results


def main():
    """Main function"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.verbose, args.log_file)
        
        logging.info("Starting Offline AI-Powered Assessment Pipeline")
        logging.info(f"Input directory: {args.input_dir}")
        logging.info(f"Answer key: {args.key}")
        logging.info(f"Output file: {args.output}")
        
        # Validate inputs
        if not validate_input_directory(args.input_dir):
            logging.error(f"Invalid input directory: {args.input_dir}")
            sys.exit(1)
        
        if not os.path.exists(args.key):
            logging.error(f"Answer key file not found: {args.key}")
            sys.exit(1)
        
        # Load answer key
        logging.info("Loading answer key...")
        answer_key = load_answer_key(args.key)
        if not answer_key:
            logging.error("Failed to load answer key")
            sys.exit(1)
        
        logging.info(f"Loaded {len(answer_key)} questions from answer key")
        
        # Process images
        results = process_images(args.input_dir, answer_key, args.verbose)
        
        # Save results
        logging.info(f"Saving results to {args.output}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total images: {results['metadata']['total_images']}")
        print(f"Successfully processed: {results['metadata']['successfully_processed']}")
        print(f"Failed: {results['metadata']['failed_images']}")
        
        if results['individual_results']:
            stats = results['class_statistics']
            print(f"\nClass Statistics:")
            print(f"Average score: {stats.get('average_score', 0):.1f}%")
            print(f"Median score: {stats.get('median_score', 0):.1f}%")
            print(f"Highest score: {stats.get('highest_score', 0):.1f}%")
            print(f"Lowest score: {stats.get('lowest_score', 0):.1f}%")
        
        print(f"\nResults saved to: {args.output}")
        print(f"Log file: {args.log_file}")
        
        logging.info("Pipeline completed successfully")
        
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()