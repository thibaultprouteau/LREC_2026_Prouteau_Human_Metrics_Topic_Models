# Stopword Analysis Tool

A simple utility to analyze how many stopwords appear in the top N words of each topic from JSON file(s). Supports both single file analysis and batch processing of multiple files in a directory.

## Usage

```bash
# Single file
python analyze_stopwords_batch.py input_file.json [language] [options]

# Directory with multiple files
python analyze_stopwords_batch.py /path/to/directory/ [language] [options]
```

## Examples

```bash
# Basic usage - analyze single file, top 4 words in English
python analyze_stopwords_batch.py topics.json

# Specify language for single file
python analyze_stopwords_batch.py topics.json french

# Analyze all JSON files in a directory
python analyze_stopwords_batch.py /path/to/folder/

# Include subdirectories (recursive search)
python analyze_stopwords_batch.py /path/to/folder/ --recursive

# Show only summary (brief mode) for multiple files
python analyze_stopwords_batch.py /path/to/folder/ --brief

# Analyze top 6 words instead of top 4
python analyze_stopwords_batch.py /path/to/folder/ --top-n 6

# Combine options
python analyze_stopwords_batch.py /path/to/folder/ en --brief --recursive --top-n 5
```

## Input Format

The tool expects a JSON file with topics in one of these formats:

### Format 1: List of topics
```json
[
  [
    {"word": "machine", "value": 0.95},
    {"word": "the", "value": 0.89},
    {"word": "learning", "value": 0.87}
  ],
  [
    {"word": "data", "value": 0.92},
    {"word": "science", "value": 0.85}
  ]
]
```

### Format 2: Dictionary of topics
```json
{
  "topic_0": [
    {"word": "machine", "value": 0.95},
    {"word": "learning", "value": 0.87}
  ],
  "topic_1": [
    {"word": "data", "value": 0.92},
    {"word": "science", "value": 0.85}
  ]
}
```

### Format 3: Alternative word-value formats
```json
[
  [
    ["machine", 0.95],
    ["learning", 0.87]
  ]
]
```

## Output

The tool provides different output formats depending on whether you're analyzing a single file or multiple files:

### Single File Output

1. **Summary Statistics**: Total topics, topics with stopwords, total stopwords found, average per topic
2. **Detailed Results**: For each topic, shows top words and which ones are stopwords
3. **Quick Summary**: Copy-paste friendly format showing counts per topic

### Multiple Files Output

1. **Overall Summary**: Combined statistics across all files
2. **Per-File Details**: Individual results for each file processed
3. **Quick Summary**: File-by-file overview for easy comparison

### Example Single File Output

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

Topic 1:
  Top 4 words: ['machine', 'the', 'learning', 'artificial']
  ⚠️  Stopwords found (1): ['the']

Topic 2:
  Top 4 words: ['data', 'and', 'science', 'analysis']
  ⚠️  Stopwords found (1): ['and']

Topic 3:
  Top 4 words: ['book', 'read', 'story', 'author']
  ✅ No stopwords found

📋 QUICK SUMMARY (for copying):
Topic 1: Found 1 stopwords in top 4 words: ['the']
Topic 2: Found 1 stopwords in top 4 words: ['and']
Topic 3: Found 0 stopwords in top 4 words: []
```

### Example Multiple Files Output

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
   Average stopwords per topic (overall): 0.62

📋 DETAILED RESULTS BY FILE:

📄 topics_1.json
   Path: /path/to/topics_1.json
   Topics: 3
   Topics with stopwords: 2
   Total stopwords: 2
   Average per topic: 0.67
   ⚠️  Topics with stopwords:
      Topic 1: ['the']
      Topic 2: ['and']

📄 topics_2.json
   Path: /path/to/topics_2.json
   Topics: 2
   Topics with stopwords: 1
   Total stopwords: 2
   Average per topic: 1.00
   ⚠️  Topics with stopwords:
      Topic 2: ['the', 'and']

📋 QUICK SUMMARY BY FILE (for copying):
topics_1.json: 3 topics, 2 stopwords in top 4
topics_2.json: 2 topics, 2 stopwords in top 4
topics_3.json: 3 topics, 1 stopwords in top 4
```

## Supported Languages

The tool supports stopwords in multiple languages using spaCy:

- **en/english** (default) - ~326 stopwords
- **fr/french** - ~507 stopwords
- **es/spanish** - ~308 stopwords
- **de/german** - ~232 stopwords
- **it/italian** - ~279 stopwords
- **pt/portuguese** - ~203 stopwords
- **nl/dutch** - ~101 stopwords
- **ru/russian** - ~151 stopwords
- **da/danish** - ~94 stopwords

Note: The exact number of stopwords may vary depending on your spaCy version.

## Dependencies

- **Python 3.6+**
- **spaCy** (recommended, for comprehensive stopword lists)

If spaCy is not installed, the tool falls back to a basic English stopword list.

To install spaCy:
```bash
pip install spacy
```

## Language Support Details

The tool uses spaCy's built-in stopword lists, which provide:
- **High quality**: Curated stopword lists for each language
- **Comprehensive coverage**: More stopwords than basic lists
- **Consistency**: Same stopwords as used in the main word intrusion package

### Stopword Counts by Language
- English (en): ~326 stopwords
- French (fr): ~507 stopwords  
- German (de): ~232 stopwords
- Spanish (es): ~308 stopwords
- Italian (it): ~279 stopwords
- Portuguese (pt): ~203 stopwords
- Dutch (nl): ~101 stopwords
- Russian (ru): ~151 stopwords
- Danish (da): ~94 stopwords

## Command Line Options

- `input_path`: JSON file or directory containing JSON files (required)
- `language`: Language for stopwords (optional, default: 'en')
- `--brief, -b`: Show only summary, not detailed results
- `--top-n N, -n N`: Number of top words to analyze per topic (default: 4)
- `--recursive, -r`: Search subdirectories recursively for JSON files (only for directories)
- `--help, -h`: Show help message

## Batch Processing Features

When processing multiple files, the tool provides:

1. **Progress Tracking**: Shows which file is being processed and results in real-time
2. **Error Handling**: Continues processing other files if one fails
3. **Combined Statistics**: Overall summary across all files
4. **Per-File Breakdown**: Individual analysis for each file
5. **Recursive Search**: Option to include JSON files in subdirectories

### Batch Processing Benefits

- **Quality Assessment at Scale**: Analyze multiple topic models simultaneously
- **Comparative Analysis**: Easy comparison of stopword presence across different models
- **Batch Reports**: Generate comprehensive reports for multiple experiments
- **Folder Organization**: Process entire directories of topic model outputs

## Use Cases

This tool is useful for:

1. **Quality Assessment**: Check if your topic models are producing too many stopwords in top positions
2. **Preprocessing Evaluation**: Determine if your stopword filtering was effective
3. **Model Comparison**: Compare different topic models based on stopword presence
4. **Data Cleaning**: Identify topics that might need better preprocessing
5. **Batch Analysis**: Process multiple topic model outputs simultaneously
6. **Experimental Validation**: Analyze results from multiple parameter configurations
7. **Quality Control**: Monitor topic quality across different datasets or time periods

## Integration with Word Intrusion Tools

This tool complements the main word intrusion package by providing a quick way to assess topic quality before generating intrusion tasks. Topics with many stopwords in top positions might produce less effective intrusion tasks.
