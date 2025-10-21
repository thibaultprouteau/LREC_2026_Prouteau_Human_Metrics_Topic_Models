# Streamlit Deployment - File Manifest

This document lists all files copied to the `streamlit_deployment` directory.

## Date Created
October 21, 2025

## Top-Level Files

- `streamlit_app.py` - Main Streamlit application (from app/)
- `requirements.txt` - Python dependencies (from app/)
- `run_app.sh` - Shell script to run the app (from app/)
- `sample_topics.json` - Sample data for testing (from app/)
- `STOPWORD_ANALYSIS_README.md` - Stopword documentation (from app/)
- `STOPWORD_TOOLS_GUIDE.md` - Stopword tools guide (from app/)
- `PACKAGE_README.md` - Package documentation (from package/README.md)
- `README.md` - Deployment README (created)

## Package Structure: word_intrusion/

### Main Package
- `word_intrusion/__init__.py` - Package initialization and exports

### Word Intrusion Module (word_intrusion/word_intrusion/)
- `__init__.py`
- `core.py` - Core word intrusion task generation logic
- `processors.py` - WordIntrusionProcessor class
- `file_processor.py` - FileProcessor class for format conversion
- `word_check.py` - Word validation functionality
- `cli.py` - Command-line interface

### Topic Mixing Module (word_intrusion/topic_mixing/)
- `__init__.py`
- `core.py` - Core topic mixing logic
- `processors.py` - TopicMixingProcessor class

### Task Selector Module (word_intrusion/task_selector/)
- `__init__.py`
- `selector.py` - TaskSelector, process_word_intrusion_folder, process_mixing_folder

### BAML Client (word_intrusion/baml_client/)
- `__init__.py`
- `async_client.py`
- `sync_client.py`
- `config.py`
- `globals.py`
- `inlinedbaml.py`
- `parser.py`
- `runtime.py`
- `stream_types.py`
- `tracing.py`
- `type_builder.py`
- `type_map.py`
- `types.py`

### BAML Configuration (word_intrusion/baml_src/)
- `clients.baml` - BAML client configuration
- `generators.baml` - BAML generator configuration
- `word_checker.baml` - Word checker BAML definition

## Changes Made

1. **Import Path Update**: Modified `streamlit_app.py` to remove hardcoded path:
   - Removed: `sys.path.append('/home/tproutea/git/word-intrusion-tools/package')`
   - Now uses local `word_intrusion` package directly

## Dependencies

The deployment is self-contained with all necessary Python modules. Dependencies are listed in `requirements.txt`.

## Usage

From the `streamlit_deployment` directory:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
bash run_app.sh
# or
streamlit run streamlit_app.py
```

## Notes

- This is a standalone deployment directory
- No external path dependencies (all imports are relative)
- Package structure mirrors the original for easy maintenance
- Can be deployed to Streamlit Cloud or other hosting services
