"""
Dataset Registry for organizing datasets by task type.
Provides standardized test suites for different ML tasks.
Supports both local and remote datasets with automatic download.
"""

import os
import requests
import tarfile
import zipfile
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Supported ML task types."""
    EMOTION_CLASSIFICATION = "emotion-classification"
    SENTIMENT_ANALYSIS = "sentiment-analysis"
    TEXT_CLASSIFICATION = "text-classification"
    NAMED_ENTITY_RECOGNITION = "ner"
    QUESTION_ANSWERING = "qa"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"


@dataclass
class DatasetConfig:
    """Configuration for a dataset in the registry."""
    name: str
    path: str  # Can be local path or remote URL
    task_type: TaskType
    config: Dict[str, Any]
    description: str
    expected_accuracy_range: Optional[tuple] = None  # (min, max) for validation
    is_remote: bool = False
    download_url: Optional[str] = None
    local_cache_dir: str = "benchmark_datasets/localTestSets"
    is_compressed: bool = False
    extract_path: Optional[str] = None  # Path to extract compressed files


class DatasetRegistry:
    """Registry for organizing datasets by task type."""
    
    def __init__(self):
        self._datasets: Dict[TaskType, List[DatasetConfig]] = {}
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize the registry with standard datasets for each task."""
        
        # Emotion Classification Datasets
        self._datasets[TaskType.EMOTION_CLASSIFICATION] = [
            DatasetConfig(
                name="2018-E-c-En-test-gold",
                path="benchmark_datasets/localTestSets/2018-E-c-En-test-gold.txt",
                task_type=TaskType.EMOTION_CLASSIFICATION,
                config={
                    "file_format": "tsv",
                    "text_column": "Tweet",
                    "label_columns": ["anger", "anticipation", "disgust", "fear", "joy", 
                                     "love", "optimism", "pessimism", "sadness", "surprise", "trust"],
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
            ),
            # Note: Users can add their own datasets by:
            # 1. Placing dataset files in benchmark_datasets/localTestSets/
            # 2. Using the add_dataset() method
            # 3. Or modifying this registry directly
        ]
        
        # Sentiment Analysis Datasets
        self._datasets[TaskType.SENTIMENT_ANALYSIS] = [
            # Note: Users can add sentiment datasets here
        ]
        
        # Text Classification Datasets
        self._datasets[TaskType.TEXT_CLASSIFICATION] = [
            # Note: Users can add text classification datasets here
        ]
    
    def download_dataset(self, dataset_config: DatasetConfig) -> bool:
        """Download a remote dataset to local cache."""
        if not dataset_config.is_remote or not dataset_config.download_url:
            return True  # Already local
        
        try:
            print(f"Downloading dataset {dataset_config.name} from {dataset_config.download_url}")
            
            # Create cache directory if it doesn't exist
            os.makedirs(dataset_config.local_cache_dir, exist_ok=True)
            
            # Download the file
            response = requests.get(dataset_config.download_url, stream=True)
            response.raise_for_status()
            
            # Determine file extension and download path
            if dataset_config.is_compressed:
                download_path = dataset_config.path + ".tar.gz"
            else:
                download_path = dataset_config.path
            
            # Save to local path
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Handle compressed files
            if dataset_config.is_compressed:
                print(f"Extracting compressed dataset {dataset_config.name}")
                if download_path.endswith('.tar.gz'):
                    with tarfile.open(download_path, 'r:gz') as tar:
                        tar.extractall(path=dataset_config.local_cache_dir)
                elif download_path.endswith('.zip'):
                    with zipfile.ZipFile(download_path, 'r') as zip_ref:
                        zip_ref.extractall(path=dataset_config.local_cache_dir)
                
                # Clean up downloaded file
                os.remove(download_path)
                
                # Update path if extraction created a different structure
                if dataset_config.extract_path:
                    extracted_path = os.path.join(dataset_config.local_cache_dir, dataset_config.extract_path)
                    if os.path.exists(extracted_path):
                        # Copy or move files to expected location
                        import shutil
                        if os.path.isdir(extracted_path):
                            # For directories, we might need to find the actual data file
                            for root, dirs, files in os.walk(extracted_path):
                                for file in files:
                                    if file.endswith('.txt') or file.endswith('.tsv') or file.endswith('.csv'):
                                        shutil.copy2(os.path.join(root, file), dataset_config.path)
                                        break
                                break
                        else:
                            shutil.copy2(extracted_path, dataset_config.path)
            
            print(f"✓ Dataset {dataset_config.name} downloaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download dataset {dataset_config.name}: {e}")
            return False
    
    def ensure_dataset_available(self, dataset_config: DatasetConfig) -> bool:
        """Ensure dataset is available locally, downloading if necessary."""
        if os.path.exists(dataset_config.path):
            return True
        
        if dataset_config.is_remote:
            return self.download_dataset(dataset_config)
        else:
            print(f"Local dataset not found: {dataset_config.path}")
            return False
    
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
    
    def get_task_types(self) -> List[TaskType]:
        """Get all supported task types."""
        return list(self._datasets.keys())
    
    def add_dataset(self, dataset_config: DatasetConfig):
        """Add a new dataset to the registry."""
        if dataset_config.task_type not in self._datasets:
            self._datasets[dataset_config.task_type] = []
        self._datasets[dataset_config.task_type].append(dataset_config)
    
    def validate_results(self, task_type: TaskType, results: Dict[str, float]) -> Dict[str, Any]:
        """Validate benchmark results against expected ranges."""
        validation_report = {
            "task_type": task_type.value,
            "datasets_tested": [],
            "overall_assessment": "PASS"
        }
        
        datasets = self.get_datasets_for_task(task_type)
        for dataset in datasets:
            if dataset.name in results:
                accuracy = results[dataset.name]
                expected_min, expected_max = dataset.expected_accuracy_range or (0.0, 1.0)
                
                assessment = "PASS" if expected_min <= accuracy <= expected_max else "FAIL"
                validation_report["datasets_tested"].append({
                    "dataset": dataset.name,
                    "accuracy": accuracy,
                    "expected_range": (expected_min, expected_max),
                    "assessment": assessment
                })
                
                if assessment == "FAIL":
                    validation_report["overall_assessment"] = "FAIL"
        
        return validation_report
