"""
Core functions for topic mixing tasks.

This module contains utility functions for computing topic similarities
and generating mixing tasks based on semantic relationships.
"""

import torch
from typing import List, Tuple
import logging

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_embedding_model(model_name: str = "intfloat/e5-small-v2", 
                        trust_remote_code: bool = False) -> SentenceTransformer:
    """
    Load a sentence transformer model for computing topic similarities.
    
    Args:
        model_name: Name of the sentence transformer model
        trust_remote_code: Whether to trust remote code for the model
        
    Returns:
        Loaded SentenceTransformer model
        
    Raises:
        ImportError: If sentence-transformers is not available
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers is required for mixing tasks. "
                        "Install it with: pip install sentence-transformers")
    
    logger.info(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name, trust_remote_code=trust_remote_code)


def compute_topic_similarities(model: SentenceTransformer, 
                             topics_as_sentences: List[str],
                             show_progress: bool = True) -> torch.Tensor:
    """
    Compute pairwise similarities between topics using sentence embeddings.
    
    Args:
        model: Loaded SentenceTransformer model
        topics_as_sentences: List of topic representations as sentences
        show_progress: Whether to show progress bar during encoding
        
    Returns:
        Tensor of pairwise similarities between topics
    """
    # Compute embeddings
    embeddings = model.encode(topics_as_sentences, show_progress_bar=show_progress)
    
    # Compute similarities
    similarities = model.similarity(embeddings, embeddings)
    
    return similarities


def find_closest_topics(similarities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the closest topic for each topic based on cosine similarity.
    
    Args:
        similarities: Tensor of pairwise topic similarities
        
    Returns:
        Tuple of (closest_indices, closest_values) where:
        - closest_indices: Index of the most similar topic for each topic (excluding self)
        - closest_values: Similarity values with the closest topics
    """
    # Create a copy to avoid modifying the original
    sim_copy = similarities.clone()
    
    # Set diagonal to -inf to exclude self-similarity
    diagonal_mask = torch.eye(similarities.size(0), dtype=torch.bool)
    sim_copy[diagonal_mask] = float('-inf')
    
    # Find the most similar topic for each topic
    closest_values, closest_indices = torch.max(sim_copy, dim=1)
    
    return closest_indices, closest_values


def extract_topic_words(topics_data: List[List[dict]], 
                       top_n: int = 50,
                       remove_stopwords: bool = False,
                       language: str = 'en') -> List[List[str]]:
    """
    Extract top words from each topic for similarity computation.
    
    Args:
        topics_data: List of topics, each topic is a list of word-value dictionaries
        top_n: Number of top words to extract per topic
        remove_stopwords: Whether to remove stopwords
        language: Language code for stopwords
        
    Returns:
        List of lists containing top words for each topic
    """
    from ..word_intrusion.core import filter_stopwords

    tops = []
    for topic in topics_data:
        top = []
        count = 0
        idx = 0
        while count < top_n and idx < len(topic):
            word = topic[idx]['word']
            if isinstance(word, str):
                top.append(word)
                count += 1
            idx += 1
        # Filter stopwords if requested
        if remove_stopwords:
            top = filter_stopwords(top, language, remove_stopwords)
        tops.append(top)
    return tops


def topics_to_sentences(topic_words: List[List[str]]) -> List[str]:
    """
    Convert topic word lists to sentence representations.
    
    Args:
        topic_words: List of word lists for each topic
        
    Returns:
        List of sentence representations (space-separated words)
    """
    return [" ".join(words) for words in topic_words]
