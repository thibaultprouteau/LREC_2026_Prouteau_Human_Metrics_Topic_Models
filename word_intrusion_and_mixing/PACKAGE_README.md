# Word Intrusion Tools - Restructured Package

This package provides tools for generating both traditional word intrusion tasks and novel topic mixing tasks from topic model data. The package has been restructured into two main modules for better organization and maintainability.

## Package Structure

```
word_intrusion/
├── __init__.py                 # Main package interface
├── word_intrusion/             # Traditional word intrusion tasks
│   ├── __init__.py
│   ├── core.py                 # Core intrusion functions
│   ├── processors.py           # High-level processors
│   ├── file_processor.py       # File I/O utilities
│   └── cli.py                  # Command-line interface
└── topic_mixing/               # Topic mixing tasks
    ├── __init__.py
    ├── core.py                 # Core mixing functions
    └── processors.py           # Mixing processors
```

## Module Overview

### 1. Word Intrusion (`word_intrusion.word_intrusion`)

Traditional word intrusion tasks where one "intruder" word from a different topic is mixed with authentic topic words.

**Key Features:**
- Extract top/bottom words from topics
- Generate intruder candidates from other topics
- Build complete intrusion tasks with metadata
- Support for multiple file formats
- Configurable stopword filtering

**Main Classes:**
- `WordIntrusionProcessor`: High-level processor for intrusion tasks
- `FileProcessor`: Handles various input file formats

### 2. Topic Mixing (`word_intrusion.topic_mixing`)

Novel mixing tasks where words from two different topics with varying similarity levels are combined.

**Key Features:**
- Semantic similarity computation using sentence transformers
- Quartile-based topic selection (least to most similar)
- Mixing tasks with known similarity levels
- Support for different embedding models
- Comprehensive task metadata

**Main Classes:**
- `TopicMixingProcessor`: High-level processor for mixing tasks
- Core functions for similarity computation and quartile analysis

## Installation

### Option 1: Complete Installation (Recommended)
```bash
pip install -r requirements.txt
```

### Option 2: Minimal Installation (Word Intrusion Only)
```bash
pip install -r requirements-core.txt
```

### Option 3: Custom Installation
```bash
# Core functionality
pip install -r requirements-core.txt

# Add topic mixing capabilities
pip install -r requirements-mixing.txt

# Add Streamlit app interface
pip install -r requirements-app.txt

# Add development tools
pip install -r requirements-dev.txt
```

### Option 4: Manual Installation
```bash
# Core dependencies (always required)
pip install numpy>=1.19.0 pandas>=1.3.0 spacy>=3.4.0

# Topic mixing dependencies (for TopicMixingProcessor)
pip install torch>=1.9.0 sentence-transformers>=2.0.0

# Optional: Streamlit app interface
pip install streamlit>=1.28.0
```

## Usage Examples

### Word Intrusion Tasks

```python
from word_intrusion.word_intrusion import WordIntrusionProcessor

# Initialize processor
processor = WordIntrusionProcessor()

# Generate intrusion tasks
tasks = processor.process_topics(
    topics_data=your_topics,
    model_name="my_model",
    n_top_words=4,
    random_seed=42
)

# Each task contains:
# {
#   'topic_id': 0,
#   'words': ['word1', 'word2', 'word3', 'word4'],
#   'intruder': 'intruder_word',
#   'task_id': 'task_0_1'
# }
```

### Topic Mixing Tasks

```python
from word_intrusion.topic_mixing import TopicMixingProcessor

# Initialize processor
processor = TopicMixingProcessor()

# Generate mixing tasks
tasks = processor.process_mixing_tasks(
    topics_data=your_topics,
    top_n=50,           # Words for similarity computation
    mixing_n_tops=5,    # Words per topic in tasks
    random_seed=42
)

# Each task contains:
# {
#   'mixed_topics': [0, 5],
#   'topic1_words': ['word1', 'word2', 'word3'],
#   'topic2_words': ['word4', 'word5', 'word6'],
#   'mixed_words': ['word2', 'word5', 'word1', ...],
#   'similarity': 0.342,
#   'quartile': 1,
#   'task_id': 'mix_0_5_q1'
# }
```

### Combined Usage

```python
# Import both modules
from word_intrusion.word_intrusion import WordIntrusionProcessor
from word_intrusion.topic_mixing import TopicMixingProcessor

# Or import from main package
from word_intrusion import WordIntrusionProcessor, TopicMixingProcessor

# Use both processors on the same data
intrusion_processor = WordIntrusionProcessor()
mixing_processor = TopicMixingProcessor()

intrusion_tasks = intrusion_processor.process_topics(topics_data)
mixing_tasks = mixing_processor.process_mixing_tasks(topics_data)
```

## Backward Compatibility

The package maintains backward compatibility with existing code:

```python
# These imports still work
from word_intrusion import WordMixingProcessor  # Alias for TopicMixingProcessor
from word_intrusion import get_top, build_tasks  # Core functions

# But the new structure is recommended
from word_intrusion.topic_mixing import TopicMixingProcessor
from word_intrusion.word_intrusion import get_top, build_tasks
```

## Data Format

Both modules expect topic data in the same format:

```python
topics_data = [
    [  # Topic 1
        {"word": "word1", "value": 0.9},
        {"word": "word2", "value": 0.8},
        # ... sorted by value (descending)
    ],
    [  # Topic 2
        {"word": "word3", "value": 0.95},
        {"word": "word4", "value": 0.85},
        # ...
    ]
    # ... more topics
]
```

## File Processing

Both modules support processing files directly:

```python
# Word intrusion from file
intrusion_tasks = processor.process_file(
    "path/to/topics.csv",
    n_top_words=4
)

# Topic mixing from file
mixing_tasks = processor.process_file_mixing(
    "path/to/topics.csv",
    top_n=50,
    mixing_n_tops=5
)
```

## Configuration Options

### Word Intrusion Parameters
- `bottom_boundary`: Fraction or indices for bottom word selection
- `top_boundary`: Fraction or indices for top word exclusion
- `n_top_words`: Number of top words per task
- `remove_stopwords`: Whether to filter stopwords
- `language`: Language code for stopwords

### Topic Mixing Parameters
- `top_n`: Words to extract for similarity computation
- `mixing_n_tops`: Words per topic in mixing tasks
- `model_name`: Sentence transformer model name
- `remove_stopwords`: Whether to filter stopwords
- `language`: Language code for stopwords

## Examples and Testing

### Running Examples
```bash
# Demonstrate separated functionality
python example_separated.py

# Replicate original notebook functionality
python replicate_notebook.py

# Topic mixing specific examples
python example_mixing.py
```

### Running Tests
```bash
python test_mixing.py
```

## Performance Considerations

### Word Intrusion
- Lightweight and fast
- Scales linearly with number of topics
- Memory usage minimal

### Topic Mixing
- Requires sentence-transformers model loading
- GPU acceleration recommended for large datasets
- Memory usage scales with model size and number of topics
- Consider smaller models for speed: `all-MiniLM-L6-v2`

## Model Recommendations

For topic mixing tasks:

- **Fast/Light**: `sentence-transformers/all-MiniLM-L6-v2`
- **Balanced**: `sentence-transformers/all-mpnet-base-v2`
- **High Quality**: `NovaSearch/stella_en_1.5B_v5` (default)
- **Multilingual**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

## Migration Guide

### From Old Structure

**Before:**
```python
from word_intrusion.mixing_tasks import WordMixingProcessor
```

**After:**
```python
from word_intrusion.topic_mixing import TopicMixingProcessor
# or use the compatibility alias
from word_intrusion import WordMixingProcessor
```

### Method Names
All method names remain the same. Only import paths have changed.

## Troubleshooting

### Common Issues

1. **ImportError for sentence-transformers**:
   ```bash
   pip install sentence-transformers torch
   ```

2. **CUDA out of memory**:
   - Use smaller model: `all-MiniLM-L6-v2`
   - Reduce `top_n` parameter
   - Process fewer topics at once

3. **Module not found errors**:
   - Check import paths match new structure
   - Use backward compatibility aliases if needed

## Contributing

When adding new functionality:

1. **Word intrusion features**: Add to `word_intrusion/word_intrusion/`
2. **Topic mixing features**: Add to `word_intrusion/topic_mixing/`
3. **Shared utilities**: Consider creating a `common/` module
4. **Update both __init__.py files** to export new functionality

## Version History

- **v0.1.0**: Initial release with mixed functionality
- **v0.2.0**: Restructured package with separated modules
  - Separated word intrusion and topic mixing
  - Improved organization and maintainability
  - Maintained backward compatibility
  - Added comprehensive documentation
