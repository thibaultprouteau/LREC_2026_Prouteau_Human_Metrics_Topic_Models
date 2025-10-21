# Word Intrusion and Topic Mixing - Task Generation Interface

This directory contains the Streamlit application and core modules for generating and managing human evaluation tasks for topic models, including both traditional word intrusion and the novel Topic Word Mixing (TWM) tasks.

## Directory Structure

```
word_intrusion_and_mixing/
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
├── run_app.sh                    # Script to run the application
├── sample_topics.json            # Sample data for testing
├── .gitignore                    # Git ignore file
│
├── Documentation files:
│   ├── README.md                 # This file
│   ├── MANIFEST.md               # Detailed file manifest
│   ├── PACKAGE_README.md         # Package documentation
│   ├── STOPWORD_ANALYSIS_README.md   # Stopword functionality documentation
│   └── STOPWORD_TOOLS_GUIDE.md   # Stopword tools guide
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
    │   ├── __init__.py
    │   ├── async_client.py
    │   ├── sync_client.py
    │   ├── types.py
    │   └── [other client files]
    │
    └── baml_src/                 # BAML configuration
        ├── clients.baml          # Client configuration
        ├── generators.baml       # Generator configuration
        └── word_checker.baml     # Word checking logic
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
