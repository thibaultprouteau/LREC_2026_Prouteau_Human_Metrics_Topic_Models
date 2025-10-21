"""
Topic Mixing Task Generation

This module contains functionality for generating word mixing tasks where words from
two different topics with varying similarity levels are mixed together.
"""

from .processors import TopicMixingProcessor
from .core import (
    load_embedding_model,
    compute_topic_similarities,
    find_closest_topics,
    extract_topic_words,
    topics_to_sentences
)

# For backward compatibility
WordMixingProcessor = TopicMixingProcessor


def create_mixing_tasks_from_csv(csv_path, **kwargs):
    """
    Convenience function to create mixing tasks directly from a CSV file.
    
    Args:
        csv_path: Path to CSV file containing topic data
        **kwargs: Additional arguments passed to process_file_mixing
        
    Returns:
        List of mixing task dictionaries
    """
    processor = TopicMixingProcessor()
    return processor.process_file_mixing(csv_path, **kwargs)


def batch_process_mixing_tasks(input_directory, output_directory, **kwargs):
    """
    Convenience function for batch processing mixing tasks.
    
    Args:
        input_directory: Path to directory containing input files
        output_directory: Path to directory for output files
        **kwargs: Additional arguments passed to process_directory_mixing
        
    Returns:
        Dictionary with processing results
    """
    processor = TopicMixingProcessor()
    return processor.process_directory_mixing(input_directory, output_directory, **kwargs)


__all__ = [
    # Main processor
    'TopicMixingProcessor',
    'WordMixingProcessor',  # Backward compatibility alias
    
    # Core functions
    'load_embedding_model',
    'compute_topic_similarities', 
    'find_closest_topics',
    'extract_topic_words',
    'topics_to_sentences',
    
    # Convenience functions
    'create_mixing_tasks_from_csv',
    'batch_process_mixing_tasks'
]
