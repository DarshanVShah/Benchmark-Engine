"""
Local Dataset Adapter

This adapter handles local dataset files for benchmarking.
Supports various formats including TSV, CSV, and custom formats.
"""

import os
import csv
from typing import Dict, Any, List, Optional, Tuple
from core import BaseDataset, DataType


class LocalEmotionDataset(BaseDataset):
    """
    Local emotion detection dataset adapter.
    
    Supports:
    - TSV/CSV emotion detection datasets
    - Multi-label emotion classification
    - Custom text preprocessing
    """
    
    def __init__(self):
        self.data = []
        self.dataset_loaded = False
        self.dataset_path = ""
        self.emotion_columns = [
            'anger', 'anticipation', 'disgust', 'fear', 'joy', 
            'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust'
        ]
        self.text_column = 'Tweet'
        self.id_column = 'ID'
        
    @property
    def output_type(self) -> DataType:
        """Local datasets output text."""
        return DataType.TEXT
        
    def load(self, dataset_path: str) -> bool:
        """Load local dataset from file."""
        try:
            print(f"Loading local dataset: {dataset_path}")
            
            if not os.path.exists(dataset_path):
                print(f"  Error: Dataset file not found: {dataset_path}")
                return False
            
            # Load data based on file extension
            if dataset_path.endswith('.txt') or dataset_path.endswith('.tsv'):
                self.data = self._load_tsv(dataset_path)
            elif dataset_path.endswith('.csv'):
                self.data = self._load_csv(dataset_path)
            else:
                print(f"  Error: Unsupported file format: {dataset_path}")
                return False
            
            self.dataset_path = dataset_path
            self.dataset_loaded = True
            
            print(f"  ✓ Local dataset loaded: {dataset_path}")
            print(f"  Samples: {len(self.data)}")
            print(f"  Emotion columns: {self.emotion_columns}")
            
            return True
            
        except Exception as e:
            print(f"  Error loading local dataset: {e}")
            return False
    
    def _load_tsv(self, file_path: str) -> List[Dict[str, Any]]:
        """Load TSV file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                # Convert emotion values to integers
                processed_row = {}
                for key, value in row.items():
                    if key in self.emotion_columns:
                        processed_row[key] = int(value)
                    else:
                        processed_row[key] = value
                data.append(processed_row)
        return data
    
    def _load_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Load CSV file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert emotion values to integers
                processed_row = {}
                for key, value in row.items():
                    if key in self.emotion_columns:
                        processed_row[key] = int(value)
                    else:
                        processed_row[key] = value
                data.append(processed_row)
        return data
    
    def get_samples(self, num_samples: Optional[int] = None) -> List[Any]:
        """Get text samples from the dataset."""
        if not self.dataset_loaded:
            raise RuntimeError("Dataset not loaded")
        
        samples = []
        data_subset = self.data[:num_samples] if num_samples else self.data
        
        for item in data_subset:
            if self.text_column in item:
                samples.append({"text": item[self.text_column]})
        
        return samples
    
    def get_samples_with_targets(self, num_samples: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """Get (input, target) pairs for evaluation."""
        if not self.dataset_loaded:
            raise RuntimeError("Dataset not loaded")
        
        samples_with_targets = []
        data_subset = self.data[:num_samples] if num_samples else self.data
        
        for item in data_subset:
            # Extract text
            text = item.get(self.text_column, "")
            if not text:
                continue
            
            # Extract emotion targets
            emotions = {}
            for emotion in self.emotion_columns:
                if emotion in item:
                    emotions[emotion] = item[emotion]
            
            # Create input format
            input_sample = {"text": text}
            
            samples_with_targets.append((input_sample, emotions))
        
        return samples_with_targets
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return dataset information."""
        if not self.dataset_loaded:
            return {"name": "LocalEmotionDataset", "loaded": False}
        
        # Calculate emotion statistics
        emotion_stats = {}
        for emotion in self.emotion_columns:
            emotion_stats[emotion] = sum(1 for item in self.data if item.get(emotion, 0) == 1)
        
        return {
            "name": "LocalEmotionDataset",
            "type": "local",
            "source": self.dataset_path,
            "num_samples": len(self.data),
            "task": "emotion-detection",
            "loaded": True,
            "description": f"Local emotion detection dataset: {os.path.basename(self.dataset_path)}",
            "emotion_columns": self.emotion_columns,
            "emotion_statistics": emotion_stats,
            "text_column": self.text_column
        }
    
    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the expected input shape for models."""
        return (512,)  # Default max length for text


class LocalTextDataset(BaseDataset):
    """
    Generic local text dataset adapter.
    
    Supports:
    - Simple text classification datasets
    - Custom text preprocessing
    - Various file formats
    """
    
    def __init__(self):
        self.data = []
        self.dataset_loaded = False
        self.dataset_path = ""
        self.text_column = 'text'
        self.label_column = 'label'
        
    @property
    def output_type(self) -> DataType:
        """Local text datasets output text."""
        return DataType.TEXT
        
    def load(self, dataset_path: str) -> bool:
        """Load local text dataset from file."""
        try:
            print(f"Loading local text dataset: {dataset_path}")
            
            if not os.path.exists(dataset_path):
                print(f"  Error: Dataset file not found: {dataset_path}")
                return False
            
            # Load data based on file extension
            if dataset_path.endswith('.txt') or dataset_path.endswith('.tsv'):
                self.data = self._load_tsv(dataset_path)
            elif dataset_path.endswith('.csv'):
                self.data = self._load_csv(dataset_path)
            else:
                print(f"  Error: Unsupported file format: {dataset_path}")
                return False
            
            self.dataset_path = dataset_path
            self.dataset_loaded = True
            
            print(f"  ✓ Local text dataset loaded: {dataset_path}")
            print(f"  Samples: {len(self.data)}")
            
            return True
            
        except Exception as e:
            print(f"  Error loading local text dataset: {e}")
            return False
    
    def _load_tsv(self, file_path: str) -> List[Dict[str, Any]]:
        """Load TSV file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                data.append(row)
        return data
    
    def _load_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Load CSV file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data
    
    def get_samples(self, num_samples: Optional[int] = None) -> List[Any]:
        """Get text samples from the dataset."""
        if not self.dataset_loaded:
            raise RuntimeError("Dataset not loaded")
        
        samples = []
        data_subset = self.data[:num_samples] if num_samples else self.data
        
        for item in data_subset:
            if self.text_column in item:
                samples.append({"text": item[self.text_column]})
        
        return samples
    
    def get_samples_with_targets(self, num_samples: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """Get (input, target) pairs for evaluation."""
        if not self.dataset_loaded:
            raise RuntimeError("Dataset not loaded")
        
        samples_with_targets = []
        data_subset = self.data[:num_samples] if num_samples else self.data
        
        for item in data_subset:
            # Extract text
            text = item.get(self.text_column, "")
            if not text:
                continue
            
            # Extract label
            label = item.get(self.label_column, 0)
            try:
                label = int(label)
            except (ValueError, TypeError):
                label = 0
            
            # Create input format
            input_sample = {"text": text}
            
            samples_with_targets.append((input_sample, label))
        
        return samples_with_targets
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return dataset information."""
        if not self.dataset_loaded:
            return {"name": "LocalTextDataset", "loaded": False}
        
        return {
            "name": "LocalTextDataset",
            "type": "local",
            "source": self.dataset_path,
            "num_samples": len(self.data),
            "task": "text-classification",
            "loaded": True,
            "description": f"Local text dataset: {os.path.basename(self.dataset_path)}",
            "text_column": self.text_column,
            "label_column": self.label_column
        }
    
    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the expected input shape for models."""
        return (512,)  # Default max length for text