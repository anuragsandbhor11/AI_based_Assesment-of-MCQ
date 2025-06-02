# Automated Grading System

A comprehensive Python-based automated grading system for educational assessments, supporting both multiple-choice and text-based questions with intelligent similarity matching.

## Features

- **Multi-format Support**: Handles both multiple-choice and free-text answers
- **Intelligent Text Matching**: Uses similarity algorithms for partial credit on text answers
- **Statistical Analysis**: Comprehensive statistics including grade distributions and question difficulty analysis
- **Flexible Export**: Export results to JSON, CSV, or custom formats
- **Robust Testing**: Extensive test suite with high code coverage
- **Easy Integration**: Simple API for embedding in larger educational systems

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/automated-grading-system.git
cd automated-grading-system
```

2. Create a virtual environment:

```bash
python -m venv grading_env
source grading_env/bin/activate  # On Windows: grading_env\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.grader import Grader, load_answer_key

# Load answer key from CSV
answer_key = load_answer_key('sample_data/answer_key.csv')

# Create grader instance
grader = Grader(answer_key)

# Student answers
student_answers = {
    '1': 'A',
    '2': 'B',
    '3': 'photosynthesis',
    '4': 'C',
    '5': 'mitochondria'
}

# Grade the answers
result = grader.grade_answers(student_answers)

# Display results
print(f"Score: {result['raw_score']}/{result['total_possible']} ({result['percentage']}%)")
print(f"Correct: {result['correct_count']}/{result['total_questions']}")
```

### Answer Key Format

Create a CSV file with the following format:

```csv
question,answer,points
1,A,2
2,B,2
3,photosynthesis,3
4,C,2
5,mitochondria,3
```

## Project Structure

```
automated-grading-system/
├── src/
│   └── grader.py           # Main grading logic
├── tests/
│   ├── test_grader.py      # Main grader tests
│   └── test_utils.py       # Utility tests
├── sample_data/
│   ├── images/             # Sample answer sheets
│   ├── answer_key.csv      # Sample answer key
│   └── sample_report.json  # Expected output
├── requirements.txt
├── README.md
└── setup.py
```

## API Reference

### Grader Class

#### `__init__(answer_key: Dict[str, Dict[str, Any]])`

Initialize the grader with an answer key.

#### `grade_answers(student_answers: Dict[str, str]) -> Dict[str, Any]`

Grade a set of student answers and return detailed results.

**Parameters:**

- `student_answers`: Dictionary mapping question numbers to student responses

**Returns:**

- Dictionary containing:
  - `raw_score`: Total points earned
  - `total_possible`: Maximum possible points
  - `percentage`: Score as percentage
  - `correct_count`: Number of correct answers
  - `question_scores`: Detailed per-question results

#### `calculate_statistics(results: List[Dict]) -> Dict[str, Any]`

Calculate comprehensive statistics for multiple student results.

#### `export_results(results: List[Dict], filepath: str, format: str)`

Export results to file in JSON or CSV format.

### Utility Functions

#### `load_answer_key(filepath: str) -> Dict[str, Dict[str, Any]]`

Load answer key from CSV file.

## Grading Logic

### Multiple Choice Questions

- Exact letter matching (case-insensitive)
- Extracts letters from text (e.g., "Answer is A" → "A")
- Full points for correct, zero for incorrect

### Text-Based Questions

- Similarity-based scoring using SequenceMatcher
- Normalization removes punctuation, extra spaces, and standardizes case
- Configurable similarity threshold (default: 0.8)
- Partial credit based on similarity score

### Text Normalization

- Converts to lowercase
- Removes punctuation except hyphens
- Removes extra whitespace
- Handles common variations (e.g., "photo-synthesis" → "photosynthesis")

## Statistical Analysis

The system provides comprehensive statistics including:

- **Basic Stats**: Mean, median, standard deviation, min/max scores
- **Grade Distribution**: A-F grade counts based on percentage ranges
- **Question Analysis**:
  - Success rates per question
  - Difficulty classification (easy/medium/hard)
  - Common wrong answers
  - Question-specific statistics

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_grader.py -v
```

### Test Coverage

The project maintains high test coverage including:

- Unit tests for all major functions
- Edge case testing
- Statistical function validation
- File I/O error handling
- Export functionality testing

## Configuration

### Similarity Threshold

Adjust the similarity threshold for text matching:

```python
grader = Grader(answer_key)
grader.similarity_threshold = 0.7  # More lenient matching
```

### Custom Scoring

The system can be extended with custom scoring logic by subclassing the `Grader` class and overriding scoring methods.

## Examples

### Batch Processing

```python
# Process multiple students
students_data = [
    {'id': 'student_001', 'answers': {'1': 'A', '2': 'B'}},
    {'id': 'student_002', 'answers': {'1': 'B', '2': 'B'}},
]

results = []
for student in students_data:
    result = grader.grade_answers(student['answers'])
    result['student_id'] = student['id']
    results.append(result)

# Calculate class statistics
stats = grader.calculate_statistics(results)
print(f"Class average: {stats['average_score']}%")
```

### Export Results

```python
# Export to JSON
grader.export_results(results, 'class_results.json', 'json')

# Export to CSV
grader.export_results(results, 'class_results.csv', 'csv')
```

## Advanced Features

### Custom Answer Normalization

Override the normalization function for domain-specific text processing:

```python
class CustomGrader(Grader):
    def _normalize_for_comparison(self, text):
        # Custom normalization logic
        normalized = super()._normalize_for_comparison(text)
        # Add domain-specific transformations
        return normalized
```

### Integration with Learning Management Systems

The grader can be integrated with LMS platforms through the simple API:

```python
def grade_submission(student_id, answers, answer_key_path):
    answer_key = load_answer_key(answer_key_path)
    grader = Grader(answer_key)
    return grader.grade_answers(answers)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or contributions:

- Open an issue on GitHub
- Check the documentation
- Review the test files for usage examples

## Changelog

### Version 1.0.0

- Initial release
- Basic grading functionality
- Statistical analysis
- Export capabilities
- Comprehensive test suite
