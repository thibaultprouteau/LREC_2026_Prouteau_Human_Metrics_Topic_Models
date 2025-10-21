"""
High-level processors for word intrusion tasks.

This module provides easy-to-use classes that combine the core functionality
for generating word intrusion tasks from various data sources.
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import logging

from .core import get_top, get_intruder_candidates, build_tasks
from .file_processor import FileProcessor

logger = logging.getLogger(__name__)


class WordIntrusionProcessor:
    """
    High-level processor for generating word intrusion tasks.
    
    This class provides a convenient interface for processing topic model data
    and generating word intrusion tasks with various customization options.
    """
    
    def __init__(self, frequency_data: Optional[Dict[str, float]] = None):
        """
        Initialize the WordIntrusionProcessor.
        
        Args:
            frequency_data: Optional dictionary mapping words to their frequencies
                          for better intruder selection
        """
        self.frequency_data = frequency_data
        self.file_processor = FileProcessor()
    
    def process_topics(self, 
                      topics_data: List[List[Dict[str, Any]]],
                      model_name: str = "unknown",
                      bottom_boundary: Union[float, List[int]] = 0.5,
                      top_boundary: Union[float, List[int]] = 0.1,
                      n_top_words: int = 4,
                      random_seed: int = 42,
                      remove_stopwords: bool = False,
                      language: str = 'en') -> List[Dict[str, Any]]:
        """
        Process topic data and generate word intrusion tasks.
        
        Args:
            topics_data: List of topics, each topic is a list of word-value dictionaries
            model_name: Name identifier for the model
            bottom_boundary: Boundary for selecting bottom words (intruder candidates)
            top_boundary: Boundary for excluding top words from intruders
            n_top_words: Number of top words to use per task
            random_seed: Random seed for reproducible results
            remove_stopwords: Whether to remove stopwords from pools
            language: Language code for stopwords ('en' or 'fr')
            
        Returns:
            List of word intrusion task dictionaries
        """
        # Validate input data
        if not self.file_processor.validate_topic_data(topics_data):
            raise ValueError("Invalid topic data format")
        
        # Get top words for each topic
        tops = get_top(topics_data, n_words=n_top_words, remove_stopwords=remove_stopwords, language=language)
        
        # Get intruder candidates
        intruder_candidates = get_intruder_candidates(
            topics_data, 
            bottom_boundary=bottom_boundary,
            top_boundary=top_boundary,
            remove_stopwords=remove_stopwords,
            language=language
        )
        
        # Build tasks
        tasks = build_tasks(
            tops, 
            intruder_candidates, 
            model_name,
            frequency_data=self.frequency_data,
            random_seed=random_seed
        )
        
        return tasks
    
    def process_file(self, 
                    file_path: Union[str, Path],
                    bottom_boundary: Union[float, List[int]] = 0.5,
                    top_boundary: Union[float, List[int]] = 0.1,
                    n_top_words: int = 4,
                    random_seed: int = 42,
                    remove_stopwords: bool = False,
                    language: str = 'en') -> List[Dict[str, Any]]:
        """
        Process a single file and generate word intrusion tasks.
        
        Args:
            file_path: Path to the file to process
            bottom_boundary: Boundary for selecting bottom words
            top_boundary: Boundary for excluding top words from intruders
            n_top_words: Number of top words to use per task
            random_seed: Random seed for reproducible results
            remove_stopwords: Whether to remove stopwords from pools
            language: Language code for stopwords ('en' or 'fr')
            
        Returns:
            List of word intrusion task dictionaries
        """
        file_path = Path(file_path)
        
        # Process the file using FileProcessor
        topics_data = self.file_processor.process_file(file_path)
        
        # Generate tasks
        tasks = self.process_topics(
            topics_data=topics_data,
            model_name=file_path.stem,
            bottom_boundary=bottom_boundary,
            top_boundary=top_boundary,
            n_top_words=n_top_words,
            random_seed=random_seed,
            remove_stopwords=remove_stopwords,
            language=language
        )
        
        return tasks
    
    def process_csv_file(self, 
                        csv_path: Union[str, Path],
                        bottom_boundary: Union[float, List[int]] = 0.5,
                        top_boundary: Union[float, List[int]] = 0.1,
                        n_top_words: int = 4,
                        random_seed: int = 42,
                        remove_stopwords: bool = False,
                        language: str = 'en') -> List[Dict[str, Any]]:
        """
        Process a CSV file containing topic model data.
        
        This is a convenience method specifically for CSV files.
        
        Args:
            csv_path: Path to the CSV file
            bottom_boundary: Boundary for selecting bottom words
            top_boundary: Boundary for excluding top words from intruders
            n_top_words: Number of top words to use per task
            random_seed: Random seed for reproducible results
            remove_stopwords: Whether to remove stopwords from pools
            language: Language code for stopwords ('en' or 'fr')
            
        Returns:
            List of word intrusion task dictionaries
        """
        return self.process_file(
            file_path=csv_path,
            bottom_boundary=bottom_boundary,
            top_boundary=top_boundary,
            n_top_words=n_top_words,
            random_seed=random_seed,
            remove_stopwords=remove_stopwords,
            language=language
        )
    
    def process_directory(self, 
                         directory: Union[str, Path],
                         output_dir: Optional[Union[str, Path]] = None,
                         save_format: str = 'json',
                         bottom_boundary: Union[float, List[int]] = 0.5,
                         top_boundary: Union[float, List[int]] = 0.1,
                         n_top_words: int = 4,
                         random_seed: int = 42,
                         recursive: bool = True,
                         remove_stopwords: bool = False,
                         language: str = 'en') -> Dict[str, List[Dict[str, Any]]]:
        """
        Process all supported files in a directory.
        
        Args:
            directory: Path to directory containing topic model files
            output_dir: Optional directory to save results
            save_format: Format to save results ('json' or 'csv')
            bottom_boundary: Boundary for selecting bottom words
            top_boundary: Boundary for excluding top words from intruders
            n_top_words: Number of top words to use per task
            random_seed: Random seed for reproducible results
            recursive: Whether to search subdirectories
            remove_stopwords: Whether to remove stopwords from pools
            language: Language code for stopwords ('en' or 'fr')
            
        Returns:
            Dictionary mapping filenames to their generated tasks
        """
        directory = Path(directory)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all files in directory
        file_data = self.file_processor.process_directory(directory, recursive=recursive)
        
        all_tasks = {}
        
        for filename, topics_data in file_data.items():
            try:
                # Generate tasks for this file
                tasks = self.process_topics(
                    topics_data=topics_data,
                    model_name=Path(filename).stem,
                    bottom_boundary=bottom_boundary,
                    top_boundary=top_boundary,
                    n_top_words=n_top_words,
                    random_seed=random_seed,
                    remove_stopwords=remove_stopwords,
                    language=language
                )
                
                all_tasks[filename] = tasks
                
                # Save to file if output directory specified
                if output_dir:
                    self._save_tasks(tasks, output_dir, Path(filename).stem, save_format)
                
                logger.info(f"Generated {len(tasks)} tasks for {filename}")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue
        
        return all_tasks
    
    def _save_tasks(self, 
                   tasks: List[Dict[str, Any]], 
                   output_dir: Path, 
                   filename_stem: str, 
                   save_format: str):
        """Save tasks to file in specified format."""
        if save_format.lower() == 'json':
            output_file = output_dir / f"{filename_stem}_tasks.json"
            df = pd.DataFrame(tasks)
            df.to_json(output_file, orient='records', indent=2)
        elif save_format.lower() == 'csv':
            output_file = output_dir / f"{filename_stem}_tasks.csv"
            df = pd.DataFrame(tasks)
            df.to_csv(output_file, index=False)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")
        
        logger.info(f"Saved tasks to {output_file}")
    
    def set_frequency_data(self, frequency_data: Dict[str, float]):
        """Set frequency data for better intruder selection."""
        self.frequency_data = frequency_data
    
    def load_frequency_data(self, file_path: Union[str, Path]) -> Dict[str, float]:
        """
        Load frequency data from a pickle file.
        
        Args:
            file_path: Path to the pickle file containing frequency data
            
        Returns:
            Dictionary mapping words to frequencies
        """
        frequency_data = load_frequency_data(file_path)
        self.frequency_data = frequency_data
        return frequency_data


def load_frequency_data(file_path: Union[str, Path]) -> Dict[str, float]:
    """
    Load frequency data from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary mapping words to their frequencies
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

import pickle
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Any, Optional
from .core import get_top, get_intruder_candidates, build_tasks


def load_frequency_data(file_path: Union[str, Path]) -> Dict[str, float]:
    """
    Load frequency data from a pickle file.
    
    Args:
        file_path: Path to the pickle file containing frequency data
        
    Returns:
        Dictionary mapping words to their frequencies
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)
