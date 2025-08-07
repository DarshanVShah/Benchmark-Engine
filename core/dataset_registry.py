"""
Dataset Registry for organizing datasets by task type.
Provides standardized test suites for different ML tasks.
"""

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
    path: str
    task_type: TaskType
    config: Dict[str, Any]
    description: str
    expected_accuracy_range: Optional[tuple] = None  # (min, max) for validation


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
                expected_accuracy_range=(0.60, 0.85)
            ),
            DatasetConfig(
                name="ISEAR",
                path="benchmark_datasets/localTestSets/isear_dataset.txt",
                task_type=TaskType.EMOTION_CLASSIFICATION,
                config={
                    "file_format": "tsv",
                    "text_column": "text",
                    "label_columns": ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"],
                    "task_type": "single-label",
                    "max_length": 512
                },
                description="Single-label emotion classification with 7 emotions",
                expected_accuracy_range=(0.70, 0.90)
            ),
            DatasetConfig(
                name="GoEmotions",
                path="benchmark_datasets/localTestSets/goemotions_test.txt",
                task_type=TaskType.EMOTION_CLASSIFICATION,
                config={
                    "file_format": "tsv",
                    "text_column": "text",
                    "label_columns": ["admiration", "amusement", "anger", "annoyance", "approval", 
                                     "caring", "confusion", "curiosity", "desire", "disappointment", 
                                     "disapproval", "disgust", "embarrassment", "excitement", "fear", 
                                     "gratitude", "grief", "joy", "love", "nervousness", "optimism", 
                                     "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"],
                    "task_type": "multi-label",
                    "max_length": 512
                },
                description="Large-scale multi-label emotion dataset with 27 emotions",
                expected_accuracy_range=(0.50, 0.75)
            )
        ]
        
        # Sentiment Analysis Datasets
        self._datasets[TaskType.SENTIMENT_ANALYSIS] = [
            DatasetConfig(
                name="SST-2",
                path="benchmark_datasets/localTestSets/sst2_test.txt",
                task_type=TaskType.SENTIMENT_ANALYSIS,
                config={
                    "file_format": "tsv",
                    "text_column": "sentence",
                    "label_columns": ["negative", "positive"],
                    "task_type": "single-label",
                    "max_length": 512
                },
                description="Stanford Sentiment Treebank - binary sentiment classification",
                expected_accuracy_range=(0.85, 0.95)
            ),
            DatasetConfig(
                name="IMDB",
                path="benchmark_datasets/localTestSets/imdb_test.txt",
                task_type=TaskType.SENTIMENT_ANALYSIS,
                config={
                    "file_format": "tsv",
                    "text_column": "text",
                    "label_columns": ["negative", "positive"],
                    "task_type": "single-label",
                    "max_length": 512
                },
                description="IMDB movie reviews - binary sentiment classification",
                expected_accuracy_range=(0.80, 0.90)
            )
        ]
        
        # Text Classification Datasets
        self._datasets[TaskType.TEXT_CLASSIFICATION] = [
            DatasetConfig(
                name="AG-News",
                path="benchmark_datasets/localTestSets/ag_news_test.txt",
                task_type=TaskType.TEXT_CLASSIFICATION,
                config={
                    "file_format": "tsv",
                    "text_column": "text",
                    "label_columns": ["world", "sports", "business", "technology"],
                    "task_type": "single-label",
                    "max_length": 512
                },
                description="AG News dataset - 4-class news classification",
                expected_accuracy_range=(0.85, 0.95)
            )
        ]
    
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
