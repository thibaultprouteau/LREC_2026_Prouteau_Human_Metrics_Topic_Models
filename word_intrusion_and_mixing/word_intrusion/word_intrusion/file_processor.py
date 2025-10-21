"""
File processor module for handling different input file formats.

This module provides a unified interface for processing various file types
containing topic model data into a standardized format for word intrusion tasks.
"""

import json
import ast
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileProcessor:
    """
    A class to process different file types into a unified format for word intrusion tasks.
    
    Supported file types:
    - .csv: CSV files with word columns and topic probability columns
    - .fuxpFX: Custom format files with bracket-separated word-value pairs
    - .fuvp: Custom format files with bracket-separated word-value pairs
    - .json: JSON files with topic data
    - .txt: Text files with various formats
    """
    
    SUPPORTED_EXTENSIONS = {'.csv', '.fuxpfx', '.json', '.txt', '.fuvp'}
    
    def __init__(self):
        """Initialize the FileProcessor."""
        self.data = []
        self.processed_files = []
    
    def process_file(self, file_path: Union[str, Path]) -> List[List[Dict[str, Any]]]:
        """
        Process a single file and return standardized topic data.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of topics, where each topic is a list of dictionaries
            with 'word' and 'value' keys
            
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}. "
                           f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS)}")
        
        logger.info(f"Processing file: {file_path}")
        
        if extension == '.csv':
            return self._process_csv(file_path)
        elif extension in {'.fuxpfx', '.fuvp'}:
            return self._process_fuxpfx(file_path)
        elif extension == '.json':
            return self._process_json(file_path)
        elif extension == '.txt':
            return self._process_txt(file_path)
    
    def process_directory(self, directory_path: Union[str, Path], 
                         recursive: bool = True) -> Dict[str, List[List[Dict[str, Any]]]]:
        """
        Process all supported files in a directory.
        
        Args:
            directory_path: Path to the directory to process
            recursive: Whether to search subdirectories recursively
            
        Returns:
            Dictionary mapping filenames to their processed topic data
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        results = {}
        
        # Get file pattern based on recursive flag
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    results[file_path.name] = self.process_file(file_path)
                    logger.info(f"Successfully processed: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {str(e)}")
                    continue
        
        return results
    
    def _process_csv(self, file_path: Path) -> List[List[Dict[str, Any]]]:
        """
        Process CSV files with word columns and topic probability columns.
        
        Expected format:
        - First column: words
        - Subsequent columns: topic probabilities
        """
        df = pd.read_csv(file_path)
        
        # Remove any unnamed columns (like index columns)
        # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Assume first column contains words
        word_column = df.columns[0]
        # Exclude 'Outliers' column if present
        topic_columns = [col for col in df.columns[1:] if col.lower() != 'outliers']
        
        topics = []
        
        for topic_col in topic_columns:
            topic_data = []
            for idx, row in df.iterrows():
                word = str(row[word_column])
                try:
                    value = float(row[topic_col])
                    # Skip NaN values
                    if np.isnan(value):
                        continue
                    topic_data.append({'word': word, 'value': value})
                except (ValueError, TypeError) as e:
                    # Log the problematic value and skip this row
                    problematic_value = row[topic_col]
                    logging.warning(f"Could not convert value '{problematic_value}' to float for word '{word}' in topic '{topic_col}'. Skipping this entry.")
                    continue
            
            # Sort by value in descending order
            topic_data.sort(key=lambda x: x['value'], reverse=True)
            topics.append(topic_data)
        
        return topics
    
    def _process_fuxpfx(self, file_path: Path) -> List[List[Dict[str, Any]]]:
        """
        Process .fuxpFX files with custom bracket format.
        
        Expected format per line: [[word1,value1][word2,value2]...]
        """
        topics = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    topic_data = self._parse_fuxpfx_format(line)
                    topics.append(topic_data)
                except Exception as e:
                    logger.warning(f"Error parsing line {line_num} in {file_path}: {str(e)}")
                    continue
        
        return topics
    
    def _parse_fuxpfx_format(self, input_str: str) -> List[Dict[str, Any]]:
        """
        Parse the custom fuxpFX format string.
        
        Args:
            input_str: String in format [[word1,value1][word2,value2]...]
            
        Returns:
            List of dictionaries with 'word' and 'value' keys
        """
        # Remove the outer brackets
        input_str = input_str.strip('[]')
        
        # Split the string by '][' to get individual word-value pairs
        pairs = re.split(r'\]\s*\[', input_str)
        
        result = []
        
        for pair in pairs:
            # Remove any remaining brackets
            pair = pair.strip('[]')
            
            # Split the pair by ',' to separate the word and the value
            parts = pair.split(',', 1)  # Split only on first comma
            if len(parts) != 2:
                continue
            
            word, value_str = parts
            word = word.strip().strip('"\'')  # Remove quotes if present
            
            try:
                value = float(value_str.strip())
                result.append({'word': word, 'value': value})
            except ValueError:
                logger.warning(f"Could not convert value to float: {value_str}")
                continue
        
        # Sort by value in descending order
        result.sort(key=lambda x: x['value'], reverse=True)
        return result
    
    def _process_json(self, file_path: Path) -> List[List[Dict[str, Any]]]:
        """
        Process JSON files containing topic data.
        
        Expected formats:
        1. List of topics, each topic is a list of {word, value} dicts
        2. Dictionary with topic keys and word-value lists as values
        3. Single topic as list of {word, value} dicts
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if isinstance(data, list):
            # Check if it's a list of topics or a single topic
            if data and isinstance(data[0], list):
                # List of topics
                topics = []
                for topic in data:
                    topic_data = self._normalize_topic_data(topic)
                    topics.append(topic_data)
                return topics
            else:
                # Single topic
                topic_data = self._normalize_topic_data(data)
                return [topic_data]
        
        elif isinstance(data, dict):
            # Dictionary format
            topics = []
            for topic_key, topic_value in data.items():
                topic_data = self._normalize_topic_data(topic_value)
                topics.append(topic_data)
            return topics
        
        else:
            raise ValueError(f"Unsupported JSON format in {file_path}")
    
    def _process_txt(self, file_path: Path) -> List[List[Dict[str, Any]]]:
        """
        Process text files. Attempts to detect format automatically.
        
        Supported formats:
        - JSON-like strings
        - Tab-separated values
        - Custom bracket format (like fuxpFX)
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        
        # Try to parse as JSON first
        try:
            data = json.loads(content)
            return self._process_json_data(data)
        except json.JSONDecodeError:
            pass
        
        # Try to parse line by line
        lines = content.split('\n')
        topics = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try fuxpFX format
            if line.startswith('[') and line.endswith(']'):
                try:
                    topic_data = self._parse_fuxpfx_format(line)
                    topics.append(topic_data)
                    continue
                except:
                    pass
            
            # Try tab-separated format
            if '\t' in line:
                try:
                    topic_data = self._parse_tsv_line(line)
                    topics.append(topic_data)
                    continue
                except:
                    pass
            
            # Try comma-separated format
            if ',' in line:
                try:
                    topic_data = self._parse_csv_line(line)
                    topics.append(topic_data)
                    continue
                except:
                    pass
        
        if not topics:
            raise ValueError(f"Could not parse text file format: {file_path}")
        
        return topics
    
    def _normalize_topic_data(self, topic_data: List[Any]) -> List[Dict[str, Any]]:
        """
        Normalize topic data to standard format.
        
        Args:
            topic_data: Raw topic data in various formats
            
        Returns:
            Standardized list of {word, value} dictionaries
        """
        result = []
        
        for item in topic_data:
            if isinstance(item, dict):
                if 'word' in item and 'value' in item:
                    result.append({'word': str(item['word']), 'value': float(item['value'])})
                elif len(item) == 2:
                    # Assume first key is word, second is value
                    keys = list(item.keys())
                    result.append({'word': str(item[keys[0]]), 'value': float(item[keys[1]])})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                result.append({'word': str(item[0]), 'value': float(item[1])})
            else:
                logger.warning(f"Could not normalize item: {item}")
        
        # Sort by value in descending order
        result.sort(key=lambda x: x['value'], reverse=True)
        return result
    
    def _parse_tsv_line(self, line: str) -> List[Dict[str, Any]]:
        """Parse a tab-separated line into topic data."""
        parts = line.split('\t')
        result = []
        
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                word = parts[i].strip()
                value = float(parts[i + 1].strip())
                result.append({'word': word, 'value': value})
        
        result.sort(key=lambda x: x['value'], reverse=True)
        return result
    
    def _parse_csv_line(self, line: str) -> List[Dict[str, Any]]:
        """Parse a comma-separated line into topic data."""
        parts = line.split(',')
        result = []
        
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                word = parts[i].strip().strip('"\'')
                value = float(parts[i + 1].strip())
                result.append({'word': word, 'value': value})
        
        result.sort(key=lambda x: x['value'], reverse=True)
        return result
    
    def _process_json_data(self, data: Any) -> List[List[Dict[str, Any]]]:
        """Process JSON data that was loaded from a text file."""
        if isinstance(data, list):
            if data and isinstance(data[0], list):
                topics = []
                for topic in data:
                    topic_data = self._normalize_topic_data(topic)
                    topics.append(topic_data)
                return topics
            else:
                topic_data = self._normalize_topic_data(data)
                return [topic_data]
        elif isinstance(data, dict):
            topics = []
            for topic_key, topic_value in data.items():
                topic_data = self._normalize_topic_data(topic_value)
                topics.append(topic_data)
            return topics
        else:
            raise ValueError("Unsupported JSON data format")
    
    def get_supported_extensions(self) -> set:
        """Return the set of supported file extensions."""
        return self.SUPPORTED_EXTENSIONS.copy()
    
    def validate_topic_data(self, topics: List[List[Dict[str, Any]]]) -> bool:
        """
        Validate that topic data is in the correct format.
        
        Args:
            topics: List of topics to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(topics, list):
            return False
        
        for topic in topics:
            if not isinstance(topic, list):
                return False
            
            for word_data in topic:
                if not isinstance(word_data, dict):
                    return False
                if 'word' not in word_data or 'value' not in word_data:
                    return False
                if not isinstance(word_data['word'], str):
                    return False
                if not isinstance(word_data['value'], (int, float)):
                    return False
        
        return True


def process_file(file_path: Union[str, Path]) -> List[List[Dict[str, Any]]]:
    """
    Convenience function to process a single file.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        Processed topic data
    """
    processor = FileProcessor()
    return processor.process_file(file_path)


def process_directory(directory_path: Union[str, Path], 
                     recursive: bool = True) -> Dict[str, List[List[Dict[str, Any]]]]:
    """
    Convenience function to process a directory of files.
    
    Args:
        directory_path: Path to the directory to process
        recursive: Whether to search subdirectories recursively
        
    Returns:
        Dictionary mapping filenames to their processed topic data
    """
    processor = FileProcessor()
    return processor.process_directory(directory_path, recursive)
