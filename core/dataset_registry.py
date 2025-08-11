"""
Dataset Registry

This registry provides access to standard benchmark datasets for testing models.
Users cannot add their own datasets - they must use the provided standard datasets.
"""

import os
import requests
import tarfile
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class TaskType(Enum):
    """Standard task types supported by the framework."""
    EMOTION_CLASSIFICATION = "emotion_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TEXT_CLASSIFICATION = "text_classification"
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"


@dataclass
class DatasetConfig:
    """Configuration for a standard benchmark dataset."""
    name: str
    path: str
    task_type: TaskType
    config: Dict[str, Any]
    description: str
    expected_accuracy_range: Optional[tuple] = None
    is_remote: bool = False
    download_url: Optional[str] = None
    local_cache_dir: str = "benchmark_datasets/localTestSets"
    is_compressed: bool = False
    extract_path: Optional[str] = None


class DatasetRegistry:
    """
    Registry of standard benchmark datasets.
    
    This registry provides access to curated datasets for testing models.
    Users cannot add their own datasets - they must use the provided standard datasets.
    """
    
    def __init__(self):
        """Initialize the registry with standard datasets."""
        self._datasets: Dict[TaskType, List[DatasetConfig]] = {}
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize with standard benchmark datasets."""
        # Emotion Classification Datasets
        self._datasets[TaskType.EMOTION_CLASSIFICATION] = [
            DatasetConfig(
                name="2018-E-c-En-test-gold",
                path="benchmark_datasets/localTestSets/2018-E-c-En-test-gold.txt",
                task_type=TaskType.EMOTION_CLASSIFICATION,
                config={
                    "file_format": "tsv",
                    "text_column": "Tweet",
                    "label_columns": ["anger", "anticipation", "disgust", "fear", "joy", "love", 
                                     "optimism", "pessimism", "sadness", "surprise", "trust"],
                    "task_type": "multi-label",
                    "max_length": 512
                },
                description="Multi-label emotion detection dataset with 11 emotions",
                expected_accuracy_range=(0.60, 0.85),
                is_remote=False
            ),
            DatasetConfig(
                name="GoEmotions",
                path="benchmark_datasets/localTestSets/goemotions_test.tsv",
                task_type=TaskType.EMOTION_CLASSIFICATION,
                config={
                    "file_format": "tsv",
                    "text_column": "text",
                    "label_columns": ["label"],
                    "task_type": "single-label",
                    "max_length": 512
                },
                description="Large-scale emotion dataset with 27 emotions from Reddit (single-label format)",
                expected_accuracy_range=(0.50, 0.75),
                is_remote=True,
                download_url="https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/test.tsv"
            )
        ]
        
        # Sentiment Analysis Datasets
        self._datasets[TaskType.SENTIMENT_ANALYSIS] = [
            DatasetConfig(
                name="IMDB-Sentiment",
                path="benchmark_datasets/localTestSets/imdb_sentiment.csv",
                task_type=TaskType.SENTIMENT_ANALYSIS,
                config={
                    "file_format": "csv",
                    "text_column": "text",
                    "label_columns": ["sentiment"],
                    "task_type": "single-label",
                    "max_length": 512
                },
                description="IMDB movie review sentiment analysis dataset",
                expected_accuracy_range=(0.80, 0.95),
                is_remote=False
            )
        ]
        
        # Text Classification Datasets
        self._datasets[TaskType.TEXT_CLASSIFICATION] = [
            DatasetConfig(
                name="AG-News",
                path="benchmark_datasets/localTestSets/ag_news.csv",
                task_type=TaskType.TEXT_CLASSIFICATION,
                config={
                    "file_format": "csv",
                    "text_column": "text",
                    "label_columns": ["category"],
                    "task_type": "single-label",
                    "max_length": 256
                },
                description="AG News topic classification dataset",
                expected_accuracy_range=(0.85, 0.95),
                is_remote=False
            )
        ]
    
    def get_task_types(self) -> List[TaskType]:
        """Get all available task types."""
        return list(self._datasets.keys())
    
    def get_datasets_for_task(self, task_type: TaskType) -> List[DatasetConfig]:
        """Get all datasets for a specific task type."""
        return self._datasets.get(task_type, [])
    
    def get_dataset_by_name(self, name: str) -> Optional[DatasetConfig]:
        """Get a specific dataset by name."""
        for datasets in self._datasets.values():
            for dataset in datasets:
                if dataset.name == name:
                    return dataset
        return None
    
    def ensure_dataset_available(self, dataset_config: DatasetConfig) -> bool:
        """
        Ensure a dataset is available locally.
        
        For remote datasets, this will download them if needed.
        """
        if not dataset_config.is_remote:
            return os.path.exists(dataset_config.path)
        
        # Handle remote dataset download
        if dataset_config.download_url:
            return self._download_dataset(dataset_config)
        
        return False
    
    def _download_dataset(self, dataset_config: DatasetConfig) -> bool:
        """Download a remote dataset."""
        try:
            print(f"Downloading dataset {dataset_config.name} from {dataset_config.download_url}")
            
            # Create cache directory
            os.makedirs(dataset_config.local_cache_dir, exist_ok=True)
            
            # Download the file
            response = requests.get(dataset_config.download_url, stream=True)
            response.raise_for_status()
            
            # Save to local path
            with open(dataset_config.path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Dataset {dataset_config.name} downloaded successfully")
            return True
            
        except Exception as e:
            print(f"Failed to download dataset {dataset_config.name}: {e}")
            return False
    
    def list_available_datasets(self) -> Dict[TaskType, List[str]]:
        """List all available datasets by task type."""
        return {
            task_type: [dataset.name for dataset in datasets]
            for task_type, datasets in self._datasets.items()
        }
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a dataset."""
        dataset = self.get_dataset_by_name(dataset_name)
        if not dataset:
            return None
        
        return {
            "name": dataset.name,
            "description": dataset.description,
            "task_type": dataset.task_type.value,
            "path": dataset.path,
            "config": dataset.config,
            "expected_accuracy_range": dataset.expected_accuracy_range,
            "is_remote": dataset.is_remote
        }
