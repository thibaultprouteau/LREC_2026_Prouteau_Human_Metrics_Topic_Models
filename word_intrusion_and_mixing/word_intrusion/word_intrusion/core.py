"""
Core functions for word intrusion task generation.

This module contains the main functions for extracting top/bottom words,
generating intruder candidates, and building word intrusion tasks.
"""

import numpy as np
import logging
from uuid import uuid4
from typing import List, Dict, Set, Union, Tuple, Any

try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS as EN_STOPWORDS
    from spacy.lang.fr.stop_words import STOP_WORDS as FR_STOPWORDS
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    EN_STOPWORDS = set()
    FR_STOPWORDS = set()

# Set up logging
logger = logging.getLogger(__name__)


def get_stopwords(language: str = 'en') -> Set[str]:
    """
    Get stopwords for a given language using spaCy.
    
    Args:
        language: Language code ('en' for English, 'fr' for French)
        
    Returns:
        Set of stopwords
    """
    if not SPACY_AVAILABLE:
        return set()
    
    if language == 'en':
        return EN_STOPWORDS
    elif language == 'fr':
        return FR_STOPWORDS
    else:
        return set()


def filter_stopwords(words: List[str], language: str = 'en', remove_stopwords: bool = True) -> List[str]:
    """
    Filter out stopwords from a list of words.
    
    Args:
        words: List of words to filter
        language: Language code for stopwords
        remove_stopwords: If True, remove stopwords; if False, return original list
        
    Returns:
        Filtered list of words
    """
    if not remove_stopwords or not SPACY_AVAILABLE:
        return words
    
    stopwords = get_stopwords(language)
    filtered_words = [word for word in words if word.lower() not in stopwords]
    
    return filtered_words


def get_top(data: List[List[Dict[str, Any]]], n_words: int = 4, remove_stopwords: bool = False, language: str = 'en') -> List[List[str]]:
    """
    Extracts the top words from each topic in the given data.

    Args:
        data: A list where each element is a list of dictionaries,
              and each dictionary contains a 'word' key.
        n_words: Number of top words to extract from each topic.
        remove_stopwords: Whether to check and log stopwords in top words.
        language: Language code for stopwords.

    Returns:
        A list where each element is a list of the top n words (as strings)
        from the corresponding topic in the input data.
    """
    tops = []
    
    # Only check for stopwords if we're planning to remove them
    if remove_stopwords and SPACY_AVAILABLE:
        stopwords = get_stopwords(language)
        
        for topic_idx, topic in enumerate(data):
            top = []
            top_stopwords = []
            
            for w in topic[:n_words]:
                word = w['word']
                top.append(word)
                # Check if this top word is a stopword
                if word.lower() in stopwords:
                    top_stopwords.append(word)
            
            # Log stopwords found in top n words of this topic
            if top_stopwords:
                logger.info(f"Topic {topic_idx + 1}: Found {len(top_stopwords)} stopwords in top {n_words} words: {top_stopwords}")
            
            tops.append(top)
    else:
        # No stopword checking, just extract top words
        for topic in data:
            top = []
            for w in topic[:n_words]:
                top.append(w['word'])
            tops.append(top)
    
    return tops


def get_bottom_pool(data: List[List[Dict[str, Any]]], 
                   boundary: Union[float, List[int]] = 0.5,
                   remove_stopwords: bool = False,
                   language: str = 'en') -> List[List[str]]:
    """
    Extracts the bottom words from each topic in the given data.

    Args:
        data (list of list of dict): A list where each element is a list of dictionaries,
                                     and each dictionary contains a 'word' key.
        boundary (float or list): If float (0-1), fraction of words to take from bottom.
                                 If list, indices for slicing [start] or [start, end].
        remove_stopwords (bool): Whether to remove stopwords from the bottom pool.
        language (str): Language code for stopwords ('en' or 'fr').

    Returns:
        list of list of str: A list where each element is a list of words (as strings)
                             from the corresponding topic in the input data.
    """
    # Do NOT check word_exists here, just collect bottom pool
    bottom_pool = []
    for topic in data:
        bottom = []
        if isinstance(boundary, float) and 0 < boundary < 1:
            bound = len(topic) // (boundary * 100)
            subset = topic[int(bound)::]
        elif isinstance(boundary, list):
            if len(boundary) == 1:
                subset = topic[int(boundary[0])::]
            elif len(boundary) == 2:
                subset = topic[int(boundary[0]):int(boundary[1])]
            else:
                raise ValueError("Boundary must be a float between 0 and 1 or a list of indices.")
        else:
            raise ValueError("Boundary must be a float between 0 and 1 or a list of indices.")
        for w in subset:
            word = w['word']
            bottom.append(word)
        # Filter stopwords if requested
        bottom = filter_stopwords(bottom, language, remove_stopwords)
        bottom_pool.append(bottom)
    return bottom_pool


def get_top_pool(data: List[List[Dict[str, Any]]], 
                boundary: Union[float, List[int]] = 0.1,
                remove_stopwords: bool = False,
                language: str = 'en') -> Tuple[List[List[str]], List[str]]:
    """
    Extracts the top words from each topic in the given data.

    Args:
        data (list of list of dict): A list where each element is a list of dictionaries,
                                     and each dictionary contains a 'word' key.
        boundary (float or list): If float (0-1), fraction of words to take from top.
                                 If list, indices for slicing [start] or [start, end].
        remove_stopwords (bool): Whether to remove stopwords from the top pool.
        language (str): Language code for stopwords ('en' or 'fr').

    Returns:
        tuple: A tuple containing:
            - list of list of str: A list where each element is a list of the top words (as strings)
            - list of str: A flattened list of all the top words from all topics.
    """
    top_pool = []
    for topic in data:
        top = []
        if isinstance(boundary, float) and 0 < boundary < 1:
            bound = len(topic) // (boundary * 100)
            subset = topic[:int(bound)]
        elif isinstance(boundary, list):
            if len(boundary) == 1:
                subset = topic[:int(boundary[0])]
            elif len(boundary) == 2:
                subset = topic[int(boundary[0]):int(boundary[1])]
            else:
                raise ValueError("Invalid boundary value. It should be a float between 0 and 1 or a list.")
        else:
            raise ValueError("Invalid boundary value. It should be a float between 0 and 1 or a list.")
        
        for word in subset:
            top.append(word['word'])
        
        # Filter stopwords if requested
        if remove_stopwords:
            top = filter_stopwords(top, language, remove_stopwords)
        
        top_pool.append(top)
    
    all_top = [word for topic in top_pool for word in topic]
    return top_pool, all_top


def get_intruder_candidates(data: List[List[Dict[str, Any]]], 
                          bottom_boundary: Union[float, List[int]] = 0.5, 
                          top_boundary: Union[float, List[int]] = 0.1,
                          remove_stopwords: bool = False,
                          language: str = 'en') -> List[Set[str]]:
    """
    Generate a list of intruder candidates from the given data.

    Args:
        data (list of list of dict): A list containing the data required to generate the pools.
        bottom_boundary: Boundary parameter for bottom pool extraction
        top_boundary: Boundary parameter for top pool extraction
        remove_stopwords (bool): Whether to remove stopwords from the pools.
        language (str): Language code for stopwords ('en' or 'fr').

    Returns:
        list: A list of sets, where each set contains words from the bottom pool 
              that are not in the corresponding top pool.
    """
    bottom_pool = get_bottom_pool(data, boundary=bottom_boundary, 
                                 remove_stopwords=remove_stopwords, language=language)
    top_pool, all_top = get_top_pool(data, boundary=top_boundary,
                                    remove_stopwords=remove_stopwords, language=language)
    intruder_candidates = []
    
    for idx, bottom in enumerate(bottom_pool):
        current_bottom = set(all_top).difference(set(top_pool[idx]))
        intruder_candidates.append({word for word in bottom if word in current_bottom})
    
    return intruder_candidates


def build_tasks(tops: List[List[str]], 
               intruder_pool: List[Set[str]], 
               model: str,
               frequency_data: Dict[str, float] = None,
               random_seed: int = 42) -> List[Dict[str, str]]:
    """
    Build a list of tasks for word intrusion detection.

    Args:
        tops (list of list of str): A list of lists, where each inner list contains 
                                   words representing the top words of a topic.
        intruder_pool (list of set of str): A list of sets, where each set contains 
                                           potential intruder words for the corresponding topic.
        model (str): The name or identifier of the model being used.
        freqs (dict): Dictionary mapping words to their frequencies (optional).
        random_seed (int): Random seed for reproducibility.

    Returns:
        list of dict: A list of task dictionaries, each containing:
            - 'model' (str): The model name or identifier.
            - 'text' (str): A unique identifier for the task.
            - 'word1' to 'word5' (str): The words in the task, including the intruder word.
            - 'intruder' (str): The intruder word.
    """
    rng = np.random.default_rng(random_seed)
    tasks = []
    
    from .word_check import word_exists
    for top, intruders in zip(tops, intruder_pool):
        if len(top) < 4:
            print(f"Not enough top words for topic {top}.")
            continue
        # Use frequency-based selection if frequency data is available
        intruder = None
        if frequency_data is not None:
            avg_freq_top = np.mean([frequency_data[word] for word in top if word in frequency_data]).astype(int)
            freq_intruders = [word for word in intruders if word in frequency_data and not np.isnan(frequency_data[word])]
            sorted_intruders = sorted(
                freq_intruders,
                key=lambda w: abs(frequency_data[w] - avg_freq_top)
            )
            closest_intruders = sorted_intruders[:10]
            # Find first eligible intruder
            for candidate in closest_intruders:
                if word_exists(candidate):
                    intruder = candidate
                    break
        else:
            # Random order, but check eligibility
            candidates = list(intruders)
            rng.shuffle(candidates)
            for candidate in candidates:
                if word_exists(candidate):
                    intruder = candidate
                    break
        if not intruder:
            print(f"No valid intruder candidates for topic {top}.")
            continue
        # Build task
        task_data = {
            'text': str(uuid4()),
            'top': top[:4],
            'intruder': str(intruder)
        }
        task_words = [*task_data['top'], task_data['intruder']]
        rng.shuffle(task_words)
        task = {
            'model': model,
            'text': task_data['text'],
            'word1': task_words[0],
            'word2': task_words[1],
            'word3': task_words[2],
            'word4': task_words[3],
            'word5': task_words[4],
            'intruder': task_data['intruder']
        }
        tasks.append(task)
    return tasks
