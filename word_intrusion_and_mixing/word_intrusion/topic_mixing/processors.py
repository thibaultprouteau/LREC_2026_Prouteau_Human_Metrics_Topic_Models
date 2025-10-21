"""
Topic Mixing Processor

This module provides the main processor class for generating topic mixing tasks.
"""

import numpy as np
import pandas as pd
import pickle
import torch
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import logging

from ..word_intrusion.processors import WordIntrusionProcessor
from .core import (
    load_embedding_model,
    compute_topic_similarities,
    find_closest_topics,
    extract_topic_words,
    topics_to_sentences
)

logger = logging.getLogger(__name__)


class TopicMixingProcessor(WordIntrusionProcessor):
    """
    Processor for generating topic mixing tasks.
    
    This class extends WordIntrusionProcessor to add functionality for creating
    word mixing tasks where words from two topics with different similarity levels
    are mixed together for evaluation.
    """
    
    def __init__(self, 
                 frequency_data: Optional[Dict[str, float]] = None,
                 model_name: str = "intfloat/e5-small-v2",
                 trust_remote_code: bool = False):
        """
        Initialize the TopicMixingProcessor.
        
        Args:
            frequency_data: Optional dictionary mapping words to their frequencies
            model_name: Name of the sentence transformer model to use for similarity
            trust_remote_code: Whether to trust remote code for the model
        """
        super().__init__(frequency_data)
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.embedding_model = None
        
    def _get_embedding_model(self):
        """Get the embedding model, loading it if necessary."""
        if self.embedding_model is None:
            self.embedding_model = load_embedding_model(
                self.model_name, 
                self.trust_remote_code
            )
        return self.embedding_model
    
    def build_mixing_tasks(self, 
                          topic_words: List[List[str]], 
                          closest_indices: "torch.Tensor",
                          closest_values: "torch.Tensor",
                          n_tops: int = 5,
                          random_seed: int = 42) -> List[Dict[str, Any]]:
        """
        Build mixing tasks from topic words and closest topic similarities.
        
        Creates two types of tasks:
        1. Single-topic tasks: All words from the same topic with half bolded (quartile=-1, similarity=1.0)
        2. Mixed-topic tasks: Words from each topic mixed with words from its closest similar topic
        
        Args:
            topic_words: List of top words for each topic
            closest_indices: Tensor of indices for each topic's closest similar topic
            closest_values: Tensor of similarity values with the closest topics
            n_tops: Number of top words to use from each topic in mixing
            random_seed: Random seed for shuffling
            
        Returns:
            List of mixing task dictionaries including both single and mixed tasks
        """
        np.random.seed(random_seed)
        
        tasks_created = set()
        tasks = []
        
        # First, create single-topic tasks (non-mixed)
        for i in range(len(topic_words)):
            topic_words_single = topic_words[i][:n_tops * 2]  # Take double the words for single tasks
            
            if len(topic_words_single) >= n_tops * 2:  # Ensure we have enough words
                # Split words into two halves
                half_size = n_tops
                first_half = topic_words_single[:half_size]
                second_half = topic_words_single[half_size:half_size * 2]
                
                # Randomly choose which half to bold
                if np.random.rand() < 0.5:
                    bolded_words = self.bold_words_html(first_half) + second_half
                else:
                    bolded_words = first_half + self.bold_words_html(second_half)
                
                # Shuffle the combined list
                np.random.shuffle(bolded_words)
                task_string = "<br>".join(bolded_words)
                
                # Create single-topic task dictionary
                task = {
                    "mixed_topics": [i],  # Only one topic
                    "topic1_words": first_half,
                    "topic2_words": second_half,
                    "mixed_words": topic_words_single,
                    "mixed_set": task_string,
                    "quartile": -1,  # Special quartile for single-topic tasks
                    "similarity": 1.0,  # Perfect similarity (same topic)
                    "task_id": "single"
                }
                tasks.append(task)
        
        # Then, create mixed-topic tasks (only closest topics)
        for i in range(closest_indices.shape[0]):
            closest_topic_idx = int(closest_indices[i])
            
            # Avoid duplicate pairs and self-mixing
            if closest_topic_idx != i and tuple(sorted((i, closest_topic_idx))) not in tasks_created:
                tasks_created.add(tuple(sorted((i, closest_topic_idx))))
                
                # Get words from both topics
                topic1_words = topic_words[i][:n_tops]
                topic2_words = topic_words[closest_topic_idx][:n_tops]
                
                # Ensure no identical words in both sets
                topic1_words_set = set(topic1_words)
                topic2_words_unique = []
                for word in topic2_words:
                    if word not in topic1_words_set:
                        topic2_words_unique.append(word)
                    else:
                        # Find a replacement from topic_words[closest_topic_idx] that is not in topic1_words_set or topic2_words_unique
                        for candidate in topic_words[closest_topic_idx][n_tops:]:
                            if candidate not in topic1_words_set and candidate not in topic2_words_unique:
                                topic2_words_unique.append(candidate)
                                break
                        else:
                            # If no replacement found, skip this word
                            continue
                topic2_words = topic2_words_unique
                
                # Create mixed word set
                task_words = topic1_words + topic2_words
                bolded_words = self.mix_and_bold_lists(topic1_words, topic2_words)
                np.random.shuffle(bolded_words)
                task_string = "<br>".join(bolded_words)
                mixed_words = task_words.copy()
                
                # Create task dictionary
                task = {
                    "mixed_topics": [i, closest_topic_idx],
                    "topic1_words": topic1_words,
                    "topic2_words": topic2_words,
                    "mixed_words": mixed_words,
                    "mixed_set": task_string,
                    "quartile": 0,  # Always 0 since we only have closest topics
                    "similarity": float(closest_values[i]),
                    "task_id": f"mix_{i}_{closest_topic_idx}_closest"
                }
                tasks.append(task)
        
        return tasks
    
    def bold_words_html(self, words: List[str]) -> List[str]:
        """
        Replace each string in the list with the same string in bold using HTML <b> tags.

        Args:
            words: List of strings.

        Returns:
            List of strings wrapped in <b>...</b>.
        """
        return [f"<b>{word}</b>" for word in words]
    
    def mix_and_bold_lists(self, list1: List[str], list2: List[str]) -> List[str]:
        """
        Given two lists, return a single list where one is in bold (HTML <b> tags) and the other is unchanged.
        The list to bold is chosen randomly.

        Args:
            list1: First list of strings.
            list2: Second list of strings.

        Returns:
            Combined list with one list bolded.
        """
        if np.random.rand() < 0.5:
            bolded = self.bold_words_html(list1)
            return bolded + list2
        else:
            bolded = self.bold_words_html(list2)
            return list1 + bolded
    
    def process_mixing_tasks(self,
                           topics_data: List[List[Dict[str, Any]]],
                           top_n: int = 50,
                           mixing_n_tops: int = 5,
                           remove_stopwords: bool = False,
                           language: str = 'en',
                           random_seed: int = 42,
                           show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Complete pipeline for generating word mixing tasks.
        
        Creates two types of tasks:
        1. Single-topic tasks: All words from the same topic with half bolded
        2. Mixed-topic tasks: Words from two different topics based on similarity
        
        Args:
            topics_data: List of topics, each topic is a list of word-value dictionaries
            top_n: Number of top words to extract per topic for similarity computation
            mixing_n_tops: Number of top words to use from each topic in mixing tasks
            remove_stopwords: Whether to remove stopwords
            language: Language code for stopwords
            random_seed: Random seed for reproducible results
            show_progress: Whether to show progress bars
            
        Returns:
            List of mixing task dictionaries (both single and mixed tasks)
        """
        # Validate input data
        if not self.file_processor.validate_topic_data(topics_data):
            raise ValueError("Invalid topic data format")
        
        logger.info(f"Processing {len(topics_data)} topics for mixing tasks")
        
        # Extract top words for each topic
        topic_words = extract_topic_words(
            topics_data, 
            top_n=top_n,
            remove_stopwords=remove_stopwords,
            language=language
        )
        
        # Convert to sentences for similarity computation
        sentences = topics_to_sentences(topic_words)
        
        # Get embedding model
        model = self._get_embedding_model()
        
        # Compute topic similarities
        similarities = compute_topic_similarities(model, sentences, show_progress=show_progress)
        
        # Find similarity quartiles
        closest_indices, closest_values = find_closest_topics(similarities)
        
        # Build mixing tasks
        mixing_tasks = self.build_mixing_tasks(
            topic_words,
            closest_indices,
            closest_values,
            n_tops=mixing_n_tops,
            random_seed=random_seed
        )
        
        logger.info(f"Generated {len(mixing_tasks)} mixing tasks")
        
        return mixing_tasks
    
    def save_mixing_tasks(self, 
                         tasks: List[Dict[str, Any]], 
                         output_path: Union[str, Path],
                         format: str = 'csv') -> None:
        """
        Save mixing tasks to file.
        
        Args:
            tasks: List of mixing task dictionaries
            output_path: Path to save the file
            format: Output format ('csv', 'json', 'pickle')
        """
        output_path = Path(output_path)
        
        if format.lower() == 'csv':
            df = pd.DataFrame(tasks)
            df.to_csv(output_path, index=False)
        elif format.lower() == 'json':
            df = pd.DataFrame(tasks)
            df.to_json(output_path, orient='records', indent=2)
        elif format.lower() == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(tasks, f)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv', 'json', or 'pickle'")
        
        logger.info(f"Saved {len(tasks)} mixing tasks to {output_path}")
    
    def process_file_mixing(self,
                           file_path: Union[str, Path],
                           top_n: int = 50,
                           mixing_n_tops: int = 5,
                           remove_stopwords: bool = False,
                           language: str = 'en',
                           random_seed: int = 42,
                           show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Process a file and generate mixing tasks.
        
        Args:
            file_path: Path to the input file
            top_n: Number of top words to extract per topic for similarity computation
            mixing_n_tops: Number of top words to use from each topic in mixing tasks
            remove_stopwords: Whether to remove stopwords
            language: Language code for stopwords
            random_seed: Random seed for reproducible results
            show_progress: Whether to show progress bars
            
        Returns:
            List of mixing task dictionaries
        """
        # Load data using the inherited file processor
        topics_data = self.file_processor.process_file(file_path)
        
        # Process mixing tasks
        return self.process_mixing_tasks(
            topics_data,
            top_n=top_n,
            mixing_n_tops=mixing_n_tops,
            remove_stopwords=remove_stopwords,
            language=language,
            random_seed=random_seed,
            show_progress=show_progress
        )
    
    def process_directory_mixing(self,
                                directory_path: Union[str, Path],
                                output_directory: Union[str, Path],
                                top_n: int = 50,
                                mixing_n_tops: int = 5,
                                remove_stopwords: bool = False,
                                language: str = 'en',
                                random_seed: int = 42,
                                show_progress: bool = True,
                                recursive: bool = False) -> Dict[str, Any]:
        """
        Process multiple files in a directory and generate mixing tasks for each.
        
        Args:
            directory_path: Path to the directory containing input files
            output_directory: Path to the directory for output files
            top_n: Number of top words to extract per topic for similarity computation
            mixing_n_tops: Number of top words to use from each topic in mixing tasks
            remove_stopwords: Whether to remove stopwords
            language: Language code for stopwords
            random_seed: Random seed for reproducible results
            show_progress: Whether to show progress bars
            recursive: Whether to search subdirectories recursively
            
        Returns:
            Dictionary with processing results and file paths
        """
        directory_path = Path(directory_path)
        output_directory = Path(output_directory)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Create output directory if it doesn't exist
        output_directory.mkdir(parents=True, exist_ok=True)
        
        # Find supported files
        pattern = "**/*" if recursive else "*"
        supported_files = []
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.file_processor.get_supported_extensions():
                supported_files.append(file_path)
        
        if not supported_files:
            raise ValueError(f"No supported files found in {directory_path}")
        
        logger.info(f"Processing {len(supported_files)} files for mixing tasks")
        
        results = {
            'processed_files': [],
            'failed_files': [],
            'output_files': [],
            'total_tasks': 0
        }
        
        for file_path in supported_files:
            try:
                # Generate mixing tasks for this file
                mixing_tasks = self.process_file_mixing(
                    file_path,
                    top_n=top_n,
                    mixing_n_tops=mixing_n_tops,
                    remove_stopwords=remove_stopwords,
                    language=language,
                    random_seed=random_seed,
                    show_progress=show_progress
                )
                
                # Create output filename
                output_file = output_directory / f"{file_path.stem}_mixing_tasks.csv"
                
                # Save mixing tasks
                self.save_mixing_tasks(mixing_tasks, output_file, format='csv')
                
                results['processed_files'].append(str(file_path))
                results['output_files'].append(str(output_file))
                results['total_tasks'] += len(mixing_tasks)
                
                logger.info(f"Processed {file_path.name}: {len(mixing_tasks)} tasks -> {output_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                results['failed_files'].append({
                    'file': str(file_path),
                    'error': str(e)
                })
        
        logger.info(f"Batch processing complete: {len(results['processed_files'])} successful, {len(results['failed_files'])} failed")
        
        return results
