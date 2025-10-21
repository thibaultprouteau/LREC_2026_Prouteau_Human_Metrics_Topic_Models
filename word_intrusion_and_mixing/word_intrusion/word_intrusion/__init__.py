"""
Word Intrusion Task Generation

This module contains the core functionality for generating traditional word intrusion tasks
where intruder words are selected from topic models.
"""

from .core import (
    get_top,
    get_bottom_pool,
    get_top_pool,
    get_intruder_candidates,
    build_tasks,
    filter_stopwords
)

from .processors import (
    WordIntrusionProcessor,
    load_frequency_data
)

from .file_processor import (
    FileProcessor,
    process_file,
    process_directory
)

__all__ = [
    # Core functions
    'get_top',
    'get_bottom_pool', 
    'get_top_pool',
    'get_intruder_candidates',
    'build_tasks',
    'filter_stopwords',
    
    # Processors
    'WordIntrusionProcessor',
    'load_frequency_data',
    
    # File processing
    'FileProcessor',
    'process_file',
    'process_directory'
]
