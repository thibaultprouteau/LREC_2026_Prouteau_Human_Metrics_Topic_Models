# Stopword Analysis Tools - Complete Guide

This directory contains two versions of the stopword analysis tool for processing JSON topic model files.

## 📁 Files Overview

### Core Tools
- **`analyze_stopwords.py`** - Original single-file version
- **`analyze_stopwords_batch.py`** - Enhanced version with batch processing capabilities
- **`STOPWORD_ANALYSIS_README.md`** - Detailed documentation

### Test Files
- **`sample_topics.json`** - Example JSON file with 3 topics
- **`test_topics_1.json`** - Additional test file with 2 topics
- **`test_topics_2.json`** - Additional test file with 3 topics
- **`subdir/nested_topics.json`** - Test file in subdirectory for recursive testing

## 🚀 Quick Start

### Single File Analysis
```bash
# Analyze one JSON file
python analyze_stopwords_batch.py sample_topics.json

# With custom settings
python analyze_stopwords_batch.py sample_topics.json french --top-n 6
```

### Batch Processing
```bash
# Analyze all JSON files in current directory
python analyze_stopwords_batch.py .

# Include subdirectories
python analyze_stopwords_batch.py . --recursive

# Brief summary only
python analyze_stopwords_batch.py . --brief
```

## 📊 Key Features

### ✅ Supported Formats
- **List of topics**: `[[ {"word": "cat", "value": 0.9}, ... ], ...]`
- **Dictionary format**: `{"topic_0": [{"word": "cat", "value": 0.9}], ...}`
- **Alternative structures**: Tuples, different key names, etc.

### 🌍 Language Support
Works with stopwords in 9+ languages using spaCy:
- English (en): ~326 stopwords
- French (fr): ~507 stopwords
- Spanish (es): ~308 stopwords
- German (de): ~232 stopwords
- Italian (it): ~279 stopwords
- Portuguese (pt): ~203 stopwords
- Dutch (nl): ~101 stopwords
- Russian (ru): ~151 stopwords
- Danish (da): ~94 stopwords

### 🔧 Flexible Analysis
- **Configurable top-N**: Analyze any number of top words (default: 4)
- **Batch processing**: Handle multiple files simultaneously
- **Recursive search**: Include subdirectories
- **Error handling**: Continue processing if individual files fail
- **Multiple output formats**: Detailed or brief summaries

## 📈 Sample Output

### Single File
```
============================================================
STOPWORDS IN TOP 4 WORDS ANALYSIS
============================================================
📊 SUMMARY:
   Total topics analyzed: 3
   Topics with stopwords: 2
   Total stopwords found: 2
   Average stopwords per topic: 0.67

📋 DETAILED RESULTS:
Topic 1: ['machine', 'the', 'learning', 'artificial'] ⚠️ ['the']
Topic 2: ['data', 'and', 'science', 'analysis'] ⚠️ ['and'] 
Topic 3: ['book', 'read', 'story', 'author'] ✅ No stopwords
```

### Multiple Files
```
================================================================================
BATCH STOPWORDS IN TOP 4 WORDS ANALYSIS
================================================================================
📊 OVERALL SUMMARY:
   Total files processed: 3
   Successful files: 3
   Failed files: 0
   Total topics (all files): 8
   Files with stopwords: 3
   Total stopwords found: 5
   Average stopwords per file: 1.67

📋 QUICK SUMMARY BY FILE:
sample_topics.json: 3 topics, 2 stopwords in top 4
test_topics_1.json: 2 topics, 2 stopwords in top 4
test_topics_2.json: 3 topics, 1 stopwords in top 4
```

## 💡 Use Cases

### Research & Development
- **Quality Assessment**: Evaluate topic model output quality
- **Parameter Tuning**: Compare models with different preprocessing
- **Batch Evaluation**: Process multiple experimental results
- **Preprocessing Validation**: Check effectiveness of stopword filtering

### Production & Monitoring
- **Quality Control**: Monitor topic quality over time
- **Model Comparison**: Compare different modeling approaches
- **Data Pipeline**: Integrate into automated quality checks
- **Reporting**: Generate standardized quality reports

## 🛠 Technical Details

### Dependencies
- **Python 3.6+** (required)
- **spaCy** (recommended, for comprehensive stopword lists)
- **Standard library**: json, pathlib, glob, argparse

### Installation
```bash
# Recommended: Install spaCy for better stopword coverage
pip install spacy

# Tool is ready to use - uses same stopwords as core package
python analyze_stopwords_batch.py --help
```

### Error Handling
- **File not found**: Clear error messages
- **Invalid JSON**: Graceful failure with error details
- **Mixed success/failure**: Continues processing other files
- **Format validation**: Handles various JSON structures

## 📝 Command Reference

```bash
# Basic usage
python analyze_stopwords_batch.py <input_path> [language] [options]

# Options
--brief, -b           # Summary only
--top-n N, -n N      # Analyze top N words (default: 4)
--recursive, -r      # Include subdirectories
--help, -h          # Show help

# Examples
python analyze_stopwords_batch.py file.json
python analyze_stopwords_batch.py /folder/ french --brief
python analyze_stopwords_batch.py /folder/ --recursive --top-n 6
```

## 🎯 Recommendation

Use **`analyze_stopwords_batch.py`** for all analyses - it handles both single files and directories seamlessly, providing more comprehensive output and better error handling than the original version.

This tool complements the main word intrusion package by providing a quick way to assess topic quality before generating intrusion tasks. Topics with many stopwords in top positions might produce less effective intrusion tasks.
