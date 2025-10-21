"""
Word Intrusion and Topic Mixing Package

A Python package for generating word intrusion tasks and topic mixing tasks from topic model data.

The package is organized into two main modules:
- word_intrusion: Traditional word intrusion tasks with intruder words
- topic_mixing: Topic mixing tasks with words from multiple topics
"""

# Import word intrusion functionality
from .word_intrusion import (
    get_top,
    get_bottom_pool,
    get_top_pool,
    get_intruder_candidates,
    build_tasks,
    filter_stopwords,
    WordIntrusionProcessor,
    load_frequency_data,
    FileProcessor,
    process_file,
    process_directory
)

# Import topic mixing functionality
from .topic_mixing import (
    TopicMixingProcessor,
    WordMixingProcessor,  # Backward compatibility alias
    load_embedding_model,
    compute_topic_similarities,
    find_closest_topics,
    extract_topic_words,
    topics_to_sentences,
    create_mixing_tasks_from_csv
)

__version__ = "0.2.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    # Word intrusion core functions
    'get_top',
    'get_bottom_pool', 
    'get_top_pool',
    'get_intruder_candidates',
    'build_tasks',
    'filter_stopwords',
    
    # Word intrusion processors
    'WordIntrusionProcessor',
    'load_frequency_data',
    
    # File processing
    'FileProcessor',
    'process_file',
    'process_directory',
    
    # Topic mixing processors
    'TopicMixingProcessor',
    'WordMixingProcessor',  # Backward compatibility
    
    # Topic mixing core functions
    'load_embedding_model',
    'compute_topic_similarities',
    'find_closest_topics', 
    'extract_topic_words',
    'topics_to_sentences',
    'create_mixing_tasks_from_csv'
]
