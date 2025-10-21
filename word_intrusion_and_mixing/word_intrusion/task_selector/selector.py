"""
Task selector for human evaluation.

This module provides modular methods to sample tasks for evaluation tracks by humans,
ensuring good representativity across models and task types.
"""

import random
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path
import os


class WordIntrusionTaskSelector:
    """
    Selector for word intrusion tasks with coverage-based or count-based sampling.
    """
    
    def __init__(self, tasks: List[Dict[str, Any]], random_seed: int = 42):
        """
        Initialize the word intrusion task selector.
        
        Args:
            tasks: List of word intrusion task dictionaries
            random_seed: Random seed for reproducible sampling
        """
        self.tasks = tasks
        self.rng = random.Random(random_seed)
        np.random.seed(random_seed)
    
    def select_by_coverage(self, coverage_percentage: float, minimum_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Select tasks by coverage percentage for each model.
        
        Args:
            coverage_percentage: Percentage of tasks to select per model (0.0 to 1.0)
            minimum_tasks: Minimum number of tasks to select per model (overrides percentage if needed)
            
        Returns:
            List of selected task dictionaries
        """
        if not 0.0 <= coverage_percentage <= 1.0:
            raise ValueError("Coverage percentage must be between 0.0 and 1.0")
        
        if minimum_tasks is not None and minimum_tasks <= 0:
            raise ValueError("Minimum tasks must be positive")
        
        # Group tasks by model
        tasks_by_model = {}
        for task in self.tasks:
            model = task.get('model', 'unknown')
            if model not in tasks_by_model:
                tasks_by_model[model] = []
            tasks_by_model[model].append(task)
        
        selected_tasks = []
        for model, model_tasks in tasks_by_model.items():
            n_select = int(len(model_tasks) * coverage_percentage)
            
            # Apply minimum tasks if specified
            if minimum_tasks is not None:
                n_select = max(n_select, min(minimum_tasks, len(model_tasks)))
            
            if n_select > 0:
                selected = self.rng.sample(model_tasks, n_select)
                selected_tasks.extend(selected)
        
        return selected_tasks
    
    def select_by_count(self, tasks_per_model: int) -> List[Dict[str, Any]]:
        """
        Select a specific number of tasks per model.
        
        Args:
            tasks_per_model: Number of tasks to select per model
            
        Returns:
            List of selected task dictionaries
        """
        if tasks_per_model <= 0:
            raise ValueError("Tasks per model must be positive")
        
        # Group tasks by model
        tasks_by_model = {}
        for task in self.tasks:
            model = task.get('model', 'unknown')
            if model not in tasks_by_model:
                tasks_by_model[model] = []
            tasks_by_model[model].append(task)
        
        selected_tasks = []
        for model, model_tasks in tasks_by_model.items():
            n_select = min(tasks_per_model, len(model_tasks))
            if n_select > 0:
                selected = self.rng.sample(model_tasks, n_select)
                selected_tasks.extend(selected)
        
        return selected_tasks


class MixingTaskSelector:
    """
    Selector for topic mixing tasks with closest-topic-aware sampling.
    """
    
    def __init__(self, tasks: List[Dict[str, Any]], random_seed: int = 42):
        """
        Initialize the mixing task selector.
        
        Args:
            tasks: List of mixing task dictionaries (must contain 'quartile' field)
            random_seed: Random seed for reproducible sampling
        """
        self.tasks = tasks
        self.rng = random.Random(random_seed)
        np.random.seed(random_seed)
    
    def select_by_coverage(self, coverage_percentage: float, minimum_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Select tasks by coverage percentage, with equal distribution between single-topic 
        and closest-topic mixed tasks.
        
        Args:
            coverage_percentage: Percentage of tasks to select per model (0.0 to 1.0)
            minimum_tasks: Minimum number of tasks to select per model (overrides percentage if needed)
            
        Returns:
            List of selected task dictionaries
        """
        if not 0.0 <= coverage_percentage <= 1.0:
            raise ValueError("Coverage percentage must be between 0.0 and 1.0")
        
        if minimum_tasks is not None and minimum_tasks <= 0:
            raise ValueError("Minimum tasks must be positive")
        
        # Group tasks by model
        tasks_by_model = {}
        for task in self.tasks:
            model = task.get('model', 'unknown')
            if model not in tasks_by_model:
                tasks_by_model[model] = {}
            quartile = task.get('quartile', 'unknown')
            if quartile not in tasks_by_model[model]:
                tasks_by_model[model][quartile] = []
            tasks_by_model[model][quartile].append(task)
        
        selected_tasks = []
        
        for model, quartile_tasks in tasks_by_model.items():
            # Separate single-topic (quartile = -1) from mixed-topic (quartile = 0, closest topics)
            single_topic_tasks = quartile_tasks.get(-1, [])
            mixed_quartiles = {q: tasks for q, tasks in quartile_tasks.items() if isinstance(q, (int, float)) and q >= 0}
            
            # Calculate total tasks to select for this model
            model_total = sum(len(tasks) for tasks in quartile_tasks.values())
            n_select_total = int(model_total * coverage_percentage)
            
            # Apply minimum tasks if specified
            if minimum_tasks is not None:
                n_select_total = max(n_select_total, min(minimum_tasks, model_total))
            
            if n_select_total <= 0:
                continue
            
            # Priority allocation: 50% for single-topic, 50% for mixed-topic quartiles
            n_single_topic = int(n_select_total * 0.5) if single_topic_tasks else 0
            n_mixed_topic = n_select_total - n_single_topic
            
            # Select single-topic tasks
            if n_single_topic > 0 and single_topic_tasks:
                n_single_actual = min(n_single_topic, len(single_topic_tasks))
                selected_single = self.rng.sample(single_topic_tasks, n_single_actual)
                selected_tasks.extend(selected_single)
            
            # Select mixed-topic tasks with equal distribution across quartiles ≥ 0
            if n_mixed_topic > 0 and mixed_quartiles:
                n_quartiles = len(mixed_quartiles)
                n_per_quartile = n_mixed_topic // n_quartiles
                remainder = n_mixed_topic % n_quartiles
                
                quartile_keys = sorted(mixed_quartiles.keys())
                for i, quartile in enumerate(quartile_keys):
                    quarter_tasks = mixed_quartiles[quartile]
                    # Distribute remainder among first quartiles
                    n_select_quarter = n_per_quartile + (1 if i < remainder else 0)
                    n_select_quarter = min(n_select_quarter, len(quarter_tasks))
                    
                    if n_select_quarter > 0:
                        selected_quarter = self.rng.sample(quarter_tasks, n_select_quarter)
                        selected_tasks.extend(selected_quarter)
        
        return selected_tasks
    
    def select_by_count_per_model(self, tasks_per_model: int) -> List[Dict[str, Any]]:
        """
        Select a specific number of tasks per model, with equal distribution between single-topic
        and closest-topic mixed tasks.
        
        Args:
            tasks_per_model: Number of tasks to select per model
            
        Returns:
            List of selected task dictionaries
        """
        if tasks_per_model <= 0:
            raise ValueError("Tasks per model must be positive")
        
        # Group tasks by model
        tasks_by_model = {}
        for task in self.tasks:
            model = task.get('model', 'unknown')
            if model not in tasks_by_model:
                tasks_by_model[model] = {}
            quartile = task.get('quartile', 'unknown')
            if quartile not in tasks_by_model[model]:
                tasks_by_model[model][quartile] = []
            tasks_by_model[model][quartile].append(task)
        
        selected_tasks = []
        
        for model, quartile_tasks in tasks_by_model.items():
            # Calculate total tasks for this model
            model_total = sum(len(tasks) for tasks in quartile_tasks.values())
            n_select = min(tasks_per_model, model_total)
            
            if n_select <= 0:
                continue
            
            # Separate single-topic (quartile = -1) from mixed-topic (quartile = 0, closest topics)
            single_topic_tasks = quartile_tasks.get(-1, [])
            mixed_quartiles = {q: tasks for q, tasks in quartile_tasks.items() if isinstance(q, (int, float)) and q >= 0}
            
            # Priority allocation: 50% for single-topic, 50% for mixed-topic quartiles
            n_single_topic = int(n_select * 0.5) if single_topic_tasks else 0
            n_mixed_topic = n_select - n_single_topic
            
            # Select single-topic tasks
            if n_single_topic > 0 and single_topic_tasks:
                n_single_actual = min(n_single_topic, len(single_topic_tasks))
                selected_single = self.rng.sample(single_topic_tasks, n_single_actual)
                selected_tasks.extend(selected_single)
                
                # If we couldn't get enough single-topic tasks, add to mixed allocation
                if n_single_actual < n_single_topic:
                    n_mixed_topic += (n_single_topic - n_single_actual)
            
            # Select mixed-topic tasks with equal distribution across quartiles ≥ 0
            if n_mixed_topic > 0 and mixed_quartiles:
                n_quartiles = len(mixed_quartiles)
                n_per_quartile = n_mixed_topic // n_quartiles
                remainder = n_mixed_topic % n_quartiles
                
                quartile_keys = sorted(mixed_quartiles.keys())
                for i, quartile in enumerate(quartile_keys):
                    quarter_tasks = mixed_quartiles[quartile]
                    # Distribute remainder among first quartiles
                    n_select_quarter = n_per_quartile + (1 if i < remainder else 0)
                    n_select_quarter = min(n_select_quarter, len(quarter_tasks))
                    
                    if n_select_quarter > 0:
                        selected_quarter = self.rng.sample(quarter_tasks, n_select_quarter)
                        selected_tasks.extend(selected_quarter)
        
        return selected_tasks

    def select_by_count(self, total_tasks: int) -> List[Dict[str, Any]]:
        """
        Select a specific total number of tasks, with equal distribution for quartiles ≥0
        and prioritized selection for quartile -1 (single-topic) tasks.
        
        Args:
            total_tasks: Total number of tasks to select
            
        Returns:
            List of selected task dictionaries
        """
        if total_tasks <= 0:
            raise ValueError("Total tasks must be positive")
        
        if total_tasks >= len(self.tasks):
            return self.tasks.copy()
        
        # Separate single-topic (quartile = -1) from mixed-topic (quartile ≥ 0) across all models
        single_topic_tasks = []
        mixed_quartile_tasks = {}
        
        for task in self.tasks:
            quartile = task.get('quartile', 'unknown')
            if quartile == -1:
                single_topic_tasks.append(task)
            elif isinstance(quartile, (int, float)) and quartile >= 0:
                if quartile not in mixed_quartile_tasks:
                    mixed_quartile_tasks[quartile] = []
                mixed_quartile_tasks[quartile].append(task)
        
        # Priority allocation: 60% for single-topic, 40% for mixed-topic quartiles
        n_single_topic = int(total_tasks * 0.6) if single_topic_tasks else 0
        n_mixed_topic = total_tasks - n_single_topic
        
        selected_tasks = []
        
        # Select single-topic tasks
        if n_single_topic > 0 and single_topic_tasks:
            n_single_actual = min(n_single_topic, len(single_topic_tasks))
            selected_single = self.rng.sample(single_topic_tasks, n_single_actual)
            selected_tasks.extend(selected_single)
            
            # If we couldn't get enough single-topic tasks, add to mixed allocation
            if n_single_actual < n_single_topic:
                n_mixed_topic += (n_single_topic - n_single_actual)
        
        # Select mixed-topic tasks with equal distribution across quartiles ≥ 0
        if n_mixed_topic > 0 and mixed_quartile_tasks:
            n_quartiles = len(mixed_quartile_tasks)
            n_per_quartile = n_mixed_topic // n_quartiles
            remainder = n_mixed_topic % n_quartiles
            
            quartile_keys = sorted(mixed_quartile_tasks.keys())
            for i, quartile in enumerate(quartile_keys):
                quarter_tasks = mixed_quartile_tasks[quartile]
                # Distribute remainder among first quartiles
                n_select_quarter = n_per_quartile + (1 if i < remainder else 0)
                n_select_quarter = min(n_select_quarter, len(quarter_tasks))
                
                if n_select_quarter > 0:
                    selected_quarter = self.rng.sample(quarter_tasks, n_select_quarter)
                    selected_tasks.extend(selected_quarter)
        
        return selected_tasks


class TaskSelector:
    """
    Unified task selector for both word intrusion and mixing tasks.
    """
    
    @staticmethod
    def load_word_intrusion_tasks(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load word intrusion tasks from CSV file.
        
        Args:
            file_path: Path to CSV file containing word intrusion tasks
            
        Returns:
            List of task dictionaries
        """
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    @staticmethod
    def load_mixing_tasks(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load mixing tasks from CSV file.
        
        Args:
            file_path: Path to CSV file containing mixing tasks
            
        Returns:
            List of task dictionaries
        """
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    @staticmethod
    def process_folder(
        folder_path: Union[str, Path],
        task_type: str,
        coverage_percentage: Optional[float] = None,
        tasks_per_model: Optional[int] = None,
        total_tasks: Optional[int] = None,
        minimum_tasks: Optional[int] = None,
        output_folder: Optional[Union[str, Path]] = None,
        random_seed: int = 42
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Batch process all CSV files in a folder.
        
        Args:
            folder_path: Path to folder containing CSV files
            task_type: Type of tasks ('word_intrusion' or 'mixing')
            coverage_percentage: Percentage of tasks to select per model (0.0 to 1.0)
            tasks_per_model: Number of tasks to select per model
            total_tasks: Total number of tasks to select (mixing only)
            minimum_tasks: Minimum number of tasks to select per model (coverage only)
            output_folder: Optional folder to save selected tasks
            random_seed: Random seed for reproducible sampling
            
        Returns:
            Dictionary mapping file names to selected tasks
            
        Raises:
            ValueError: If invalid task_type or selection parameters
        """
        if task_type not in ['word_intrusion', 'mixing']:
            raise ValueError("task_type must be 'word_intrusion' or 'mixing'")
        
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
        
        # Find all CSV files
        csv_files = list(folder_path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {folder_path}")
        
        results = {}
        all_selected_tasks = []
        
        for csv_file in csv_files:
            print(f"Processing {csv_file.name}...")
            
            try:
                # Load tasks based on type
                if task_type == 'word_intrusion':
                    tasks = TaskSelector.load_word_intrusion_tasks(csv_file)
                    selected = TaskSelector.select_word_intrusion_tasks(
                        tasks, 
                        coverage_percentage=coverage_percentage,
                        tasks_per_model=tasks_per_model,
                        minimum_tasks=minimum_tasks,
                        random_seed=random_seed
                    )
                else:  # mixing
                    tasks = TaskSelector.load_mixing_tasks(csv_file)
                    selected = TaskSelector.select_mixing_tasks(
                        tasks,
                        coverage_percentage=coverage_percentage,
                        total_tasks=total_tasks,
                        tasks_per_model=tasks_per_model,
                        minimum_tasks=minimum_tasks,
                        random_seed=random_seed
                    )
                
                # Add source file info to each task
                for task in selected:
                    task['source_file'] = csv_file.name
                
                results[csv_file.name] = selected
                all_selected_tasks.extend(selected)
                
            except Exception as e:
                print(f"Error processing {csv_file.name}: {e}")
                continue
        
        # Save all selected tasks to a single file if output folder specified
        if output_folder and all_selected_tasks:
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
            
            output_file = output_path / f"selected_{task_type}_tasks.csv"
            df_all_selected = pd.DataFrame(all_selected_tasks)
            df_all_selected.to_csv(output_file, index=False)
            print(f"Saved {len(all_selected_tasks)} total selected tasks from {len(csv_files)} files to {output_file}")
        
        return results
    
    @staticmethod
    def select_word_intrusion_tasks(
        tasks: List[Dict[str, Any]], 
        coverage_percentage: Optional[float] = None,
        tasks_per_model: Optional[int] = None,
        minimum_tasks: Optional[int] = None,
        random_seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Select word intrusion tasks using coverage or count-based sampling.
        
        Args:
            tasks: List of word intrusion task dictionaries
            coverage_percentage: Percentage of tasks to select per model (0.0 to 1.0)
            tasks_per_model: Number of tasks to select per model
            minimum_tasks: Minimum number of tasks to select per model (coverage only)
            random_seed: Random seed for reproducible sampling
            
        Returns:
            List of selected task dictionaries
            
        Raises:
            ValueError: If both or neither selection method is provided
        """
        if (coverage_percentage is None) == (tasks_per_model is None):
            raise ValueError("Provide exactly one of: coverage_percentage or tasks_per_model")
        
        selector = WordIntrusionTaskSelector(tasks, random_seed)
        
        if coverage_percentage is not None:
            return selector.select_by_coverage(coverage_percentage, minimum_tasks)
        else:
            return selector.select_by_count(tasks_per_model)
    
    @staticmethod
    def select_mixing_tasks(
        tasks: List[Dict[str, Any]],
        coverage_percentage: Optional[float] = None,
        total_tasks: Optional[int] = None,
        tasks_per_model: Optional[int] = None,
        minimum_tasks: Optional[int] = None,
        random_seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Select mixing tasks using coverage, total count, or per-model count sampling with quartile awareness.
        
        Args:
            tasks: List of mixing task dictionaries (must contain 'quartile' field)
            coverage_percentage: Percentage of tasks to select per model (0.0 to 1.0)
            total_tasks: Total number of tasks to select
            tasks_per_model: Number of tasks to select per model
            minimum_tasks: Minimum number of tasks to select per model (coverage only)
            random_seed: Random seed for reproducible sampling
            
        Returns:
            List of selected task dictionaries
            
        Raises:
            ValueError: If not exactly one selection method is provided
        """
        selection_methods = [coverage_percentage is not None, total_tasks is not None, tasks_per_model is not None]
        if sum(selection_methods) != 1:
            raise ValueError("Provide exactly one of: coverage_percentage, total_tasks, or tasks_per_model")
        
        selector = MixingTaskSelector(tasks, random_seed)
        
        if coverage_percentage is not None:
            return selector.select_by_coverage(coverage_percentage, minimum_tasks)
        elif total_tasks is not None:
            return selector.select_by_count(total_tasks)
        else:
            return selector.select_by_count_per_model(tasks_per_model)


# Example usage functions
def process_word_intrusion_folder(
    folder_path: Union[str, Path],
    coverage_percentage: Optional[float] = None,
    tasks_per_model: Optional[int] = None,
    minimum_tasks: Optional[int] = None,
    output_folder: Optional[Union[str, Path]] = None,
    random_seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience function to process word intrusion task folders.
    
    Example:
        # Select 30% of tasks per model from all files
        results = process_word_intrusion_folder(
            "/path/to/word_intrusion_tasks/",
            coverage_percentage=0.3,
            output_folder="/path/to/selected_tasks/"
        )
        
        # Select 30% with minimum 10 tasks per model
        results = process_word_intrusion_folder(
            "/path/to/word_intrusion_tasks/",
            coverage_percentage=0.3,
            minimum_tasks=10,
            output_folder="/path/to/selected_tasks/"
        )
        
        # Select 100 tasks per model from all files
        results = process_word_intrusion_folder(
            "/path/to/word_intrusion_tasks/",
            tasks_per_model=100,
            output_folder="/path/to/selected_tasks/"
        )
    """
    return TaskSelector.process_folder(
        folder_path=folder_path,
        task_type='word_intrusion',
        coverage_percentage=coverage_percentage,
        tasks_per_model=tasks_per_model,
        minimum_tasks=minimum_tasks,
        output_folder=output_folder,
        random_seed=random_seed
    )


def process_mixing_folder(
    folder_path: Union[str, Path],
    coverage_percentage: Optional[float] = None,
    total_tasks: Optional[int] = None,
    tasks_per_model: Optional[int] = None,
    minimum_tasks: Optional[int] = None,
    output_folder: Optional[Union[str, Path]] = None,
    random_seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience function to process mixing task folders.
    
    Example:
        # Select 25% of tasks maintaining quartile proportions
        results = process_mixing_folder(
            "/path/to/mixing_tasks/",
            coverage_percentage=0.25,
            output_folder="/path/to/selected_tasks/"
        )
        
        # Select 25% with minimum 15 tasks per model
        results = process_mixing_folder(
            "/path/to/mixing_tasks/",
            coverage_percentage=0.25,
            minimum_tasks=15,
            output_folder="/path/to/selected_tasks/"
        )
        
        # Select 500 total tasks with proportional sampling
        results = process_mixing_folder(
            "/path/to/mixing_tasks/",
            total_tasks=500,
            output_folder="/path/to/selected_tasks/"
        )
        
        # Select 100 tasks per model maintaining quartile proportions
        results = process_mixing_folder(
            "/path/to/mixing_tasks/",
            tasks_per_model=100,
            output_folder="/path/to/selected_tasks/"
        )
    """
    return TaskSelector.process_folder(
        folder_path=folder_path,
        task_type='mixing',
        coverage_percentage=coverage_percentage,
        total_tasks=total_tasks,
        tasks_per_model=tasks_per_model,
        minimum_tasks=minimum_tasks,
        output_folder=output_folder,
        random_seed=random_seed
    )
