# Streamlit Deployment Package

This directory contains all necessary files to deploy the Word Intrusion Streamlit application.

## Directory Structure

```
streamlit_deployment/
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
├── run_app.sh                    # Script to run the application
├── sample_topics.json            # Sample data for testing
├── STOPWORD_ANALYSIS_README.md   # Stopword functionality documentation
├── STOPWORD_TOOLS_GUIDE.md       # Stopword tools guide
├── PACKAGE_README.md             # Package documentation
│
└── word_intrusion/               # Main package directory
    ├── __init__.py               # Package initialization
    │
    ├── word_intrusion/           # Word intrusion module
    │   ├── __init__.py
    │   ├── core.py               # Core word intrusion logic
    │   ├── processors.py         # WordIntrusionProcessor class
    │   ├── file_processor.py     # FileProcessor class
    │   ├── word_check.py         # Word checking functionality
    │   └── cli.py                # Command-line interface
    │
    ├── topic_mixing/             # Topic mixing module
    │   ├── __init__.py
    │   ├── core.py               # Core topic mixing logic
    │   └── processors.py         # TopicMixingProcessor class
    │
    ├── task_selector/            # Task selection/sampling module
    │   ├── __init__.py
    │   └── selector.py           # TaskSelector and folder processing
    │
    ├── baml_client/              # BAML client for LLM integration
    │   └── [various Python files]
    │
    └── baml_src/                 # BAML configuration
        ├── clients.baml
        ├── generators.baml
        └── word_checker.baml
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
bash run_app.sh
```

Or directly:
```bash
streamlit run streamlit_app.py
```

## Features

The Streamlit app provides four main tabs:

1. **File Processing**: Convert various topic model formats to unified JSON format
2. **Task Generation**: Generate word intrusion tasks from processed data
3. **Topic Mixing**: Create topic mixing tasks using semantic similarity
4. **Task Sampling**: Sample tasks for human evaluation with coverage control

## Key Components

- **FileProcessor**: Handles multiple input formats (CSV, JSON, fuxpFX, fuvp, TXT)
- **WordIntrusionProcessor**: Generates word intrusion tasks with configurable parameters
- **TopicMixingProcessor**: Creates mixing tasks based on topic similarity
- **TaskSelector**: Samples tasks for evaluation with various strategies

## Dependencies

Main dependencies include:
- streamlit
- pandas
- sentence-transformers (for topic mixing)
- spacy (for stopword filtering)
- scikit-learn
- torch/transformers (for embedding models)

See `requirements.txt` for complete list.

## Usage Notes

- The app supports both single file and batch directory processing
- Stopword filtering is available for both word intrusion and topic mixing
- Control tasks can be added for quality assurance in sampling
- Multiple sampling files can be generated without overlap

## Package Structure

This deployment maintains the package structure from the main repository:
- `word_intrusion/word_intrusion/`: Traditional word intrusion tasks
- `word_intrusion/topic_mixing/`: Topic mixing functionality
- `word_intrusion/task_selector/`: Task sampling and selection
- `word_intrusion/baml_client/`: BAML integration for word checking
