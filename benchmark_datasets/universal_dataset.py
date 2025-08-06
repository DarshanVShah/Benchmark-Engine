"""
Universal Dataset Adapter

This adapter automatically detects and handles any dataset format.
Users just provide a file path - no custom code needed!
"""

import os
import csv
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from core import BaseDataset, DataType


class UniversalDataset(BaseDataset):
    """
    Universal dataset adapter that auto-detects format and structure.
    
    Supports:
    - CSV, TSV, JSON, TXT files
    - Auto-detection of text columns and label columns
    - Multiple task types (classification, regression, multi-label)
    - Custom preprocessing configuration
    - HuggingFace datasets (via URL)
    
    Users just specify the file path - everything else is automatic!
    """
    
    def __init__(self):
        self.data = []
        self.dataset_loaded = False
        self.dataset_path = ""
        self.file_format = "auto-detected"
        self.text_column = None
        self.label_columns = []
        self.task_type = "auto-detected"
        self.max_length = 512
        self.config = {}
        
    @property
    def output_type(self) -> DataType:
        """Universal datasets output text by default."""
        return DataType.TEXT
        
    def load(self, dataset_path: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load any dataset from file path with optional configuration.
        
        Args:
            dataset_path: Path to dataset file or HuggingFace dataset name
            config: Optional configuration for custom preprocessing
        """
        try:
            print(f"Loading universal dataset: {dataset_path}")
            
            # Store configuration
            if config:
                self.config = config
                self.max_length = config.get("max_length", 512)
            
            # Check if it's a HuggingFace dataset
            if dataset_path.startswith(("http", "https")) or "/" in dataset_path and not os.path.exists(dataset_path):
                return self._load_huggingface_dataset(dataset_path)
            
            # Check if file exists
            if not os.path.exists(dataset_path):
                print(f"  Error: Dataset file not found: {dataset_path}")
                return False
            
            # Auto-detect file format and load
            self._auto_detect_and_load(dataset_path)
            
            # Auto-detect structure
            self._auto_detect_structure()
            
            self.dataset_path = dataset_path
            self.dataset_loaded = True
            
            print(f"  ✓ Universal dataset loaded: {dataset_path}")
            print(f"  Format: {self.file_format}")
            print(f"  Samples: {len(self.data)}")
            print(f"  Task: {self.task_type}")
            print(f"  Text column: {self.text_column}")
            print(f"  Label columns: {self.label_columns}")
            
            return True
            
        except Exception as e:
            print(f"  Error loading universal dataset: {e}")
            return False
    
    def _auto_detect_and_load(self, file_path: str):
        """Auto-detect file format and load data."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            self._load_csv(file_path)
        elif file_extension == '.tsv' or file_extension == '.txt':
            self._load_tsv(file_path)
        elif file_extension == '.json':
            self._load_json(file_path)
        else:
            # Try different formats
            self._try_multiple_formats(file_path)
    
    def _load_csv(self, file_path: str):
        """Load CSV file."""
        self.file_format = "csv"
        try:
            df = pd.read_csv(file_path)
            self.data = df.to_dict('records')
        except:
            # Fallback to csv module
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.data = list(reader)
    
    def _load_tsv(self, file_path: str):
        """Load TSV file."""
        self.file_format = "tsv"
        try:
            df = pd.read_csv(file_path, delimiter='\t')
            self.data = df.to_dict('records')
        except:
            # Fallback to csv module
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                self.data = list(reader)
    
    def _load_json(self, file_path: str):
        """Load JSON file."""
        self.file_format = "json"
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            
        # Handle different JSON formats
        if isinstance(content, list):
            self.data = content
        elif isinstance(content, dict):
            if "data" in content:
                data = content["data"]
                labels = content.get("labels", [])
                
                # Combine data and labels
                self.data = []
                for i, item in enumerate(data):
                    sample = {"text": item}
                    if i < len(labels):
                        sample["label"] = labels[i]
                    self.data.append(sample)
            else:
                self.data = [content]
        else:
            self.data = [content]
    
    def _try_multiple_formats(self, file_path: str):
        """Try multiple formats if auto-detection fails."""
        formats_to_try = [
            (self._load_csv, "csv"),
            (self._load_tsv, "tsv"),
            (self._load_json, "json")
        ]
        
        for load_func, format_name in formats_to_try:
            try:
                load_func(file_path)
                self.file_format = format_name
                return
            except:
                continue
        
        raise ValueError(f"Could not auto-detect format for {file_path}")
    
    def _load_huggingface_dataset(self, dataset_name: str) -> bool:
        """Load HuggingFace dataset."""
        try:
            import datasets as hf_datasets
            print(f"  Loading HuggingFace dataset: {dataset_name}")
            
            # Load dataset
            dataset = hf_datasets.load_dataset(dataset_name)
            
            # Convert to our format
            self.data = []
            for split_name, split_data in dataset.items():
                for item in split_data:
                    # Convert HuggingFace format to our format
                    converted_item = {}
                    for key, value in item.items():
                        if isinstance(value, str) and len(value) > 10:
                            converted_item["text"] = value
                        elif isinstance(value, (int, float)):
                            converted_item["label"] = value
                        else:
                            converted_item[key] = value
                    
                    if "text" in converted_item:
                        self.data.append(converted_item)
            
            self.file_format = "huggingface"
            return True
            
        except Exception as e:
            print(f"  Error loading HuggingFace dataset: {e}")
            return False
    
    def _auto_detect_structure(self):
        """Auto-detect dataset structure and task type."""
        if not self.data:
            return
        
        # Get sample to analyze structure
        sample = self.data[0]
        
        # Detect text column
        text_candidates = ["text", "content", "sentence", "input", "message", "post", "review", "comment", "Tweet"]
        for candidate in text_candidates:
            if candidate in sample:
                self.text_column = candidate
                break
        
        # If no text column found, use first string field
        if not self.text_column:
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 10:
                    self.text_column = key
                    break
        
        # Detect label columns
        label_candidates = ["label", "labels", "class", "target", "sentiment", "emotion", "category"]
        emotion_candidates = ["anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness", "surprise", "trust"]
        
        for candidate in label_candidates + emotion_candidates:
            if candidate in sample:
                self.label_columns.append(candidate)
        
        # Detect task type
        if len(self.label_columns) > 1:
            # Check if it's emotion detection
            emotion_count = sum(1 for col in self.label_columns if col in emotion_candidates)
            if emotion_count >= 5:  # At least 5 emotion columns
                self.task_type = "emotion-detection"
            else:
                self.task_type = "multi-label-classification"
        elif len(self.label_columns) == 1:
            # Check if it's sentiment analysis
            if any("sentiment" in col.lower() for col in self.label_columns):
                self.task_type = "sentiment-analysis"
            else:
                self.task_type = "text-classification"
        else:
            self.task_type = "text-processing"
    
    def get_samples(self, num_samples: Optional[int] = None) -> List[Any]:
        """Get text samples from the dataset."""
        if not self.dataset_loaded:
            raise RuntimeError("Dataset not loaded")
        
        samples = []
        data_subset = self.data[:num_samples] if num_samples else self.data
        
        for item in data_subset:
            if self.text_column and self.text_column in item:
                text = item[self.text_column]
                if isinstance(text, str) and len(text) > 0:
                    samples.append({"text": text[:self.max_length]})
        
        return samples
    
    def get_samples_with_targets(self, num_samples: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """Get (input, target) pairs for evaluation."""
        if not self.dataset_loaded:
            raise RuntimeError("Dataset not loaded")
        
        samples_with_targets = []
        data_subset = self.data[:num_samples] if num_samples else self.data
        
        for item in data_subset:
            # Extract text
            text = ""
            if self.text_column and self.text_column in item:
                text = item[self.text_column]
            
            if not text or not isinstance(text, str):
                continue
            
            # Extract targets based on task type
            target = self._extract_targets(item)
            
            # Create input format
            input_sample = {"text": text[:self.max_length]}
            
            samples_with_targets.append((input_sample, target))
        
        return samples_with_targets
    
    def _extract_targets(self, item: Dict[str, Any]) -> Any:
        """Extract targets based on detected task type."""
        if self.task_type == "emotion-detection":
            # Extract emotion targets
            emotions = {}
            for emotion in ['anger', 'anticipation', 'disgust', 'fear', 'joy', 
                          'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']:
                emotions[emotion] = item.get(emotion, 0)
            return emotions
        
        elif self.task_type == "multi-label-classification":
            # Extract multiple labels
            labels = {}
            for label_col in self.label_columns:
                if label_col in item:
                    labels[label_col] = item[label_col]
            return labels
        
        elif self.task_type == "text-classification":
            # Extract single label
            for label_col in self.label_columns:
                if label_col in item:
                    return item[label_col]
            return 0
        
        elif self.task_type == "sentiment-analysis":
            # Extract sentiment
            for label_col in self.label_columns:
                if label_col in item:
                    return item[label_col]
            return 0
        
        else:
            # Generic: return first numeric field as target
            for key, value in item.items():
                if isinstance(value, (int, float)) and key != "id":
                    return value
            return 0
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return dataset information."""
        if not self.dataset_loaded:
            return {"name": "UniversalDataset", "loaded": False}
        
        # Calculate basic statistics
        num_samples = len(self.data)
        
        # Calculate label statistics if available
        label_stats = {}
        if self.label_columns:
            for label_col in self.label_columns:
                if label_col in self.data[0]:
                    values = [item.get(label_col, 0) for item in self.data]
                    unique_values = set(values)
                    label_stats[label_col] = {
                        "unique_values": len(unique_values),
                        "value_range": f"{min(values)} - {max(values)}"
                    }
        
        return {
            "name": "UniversalDataset",
            "type": "universal",
            "source": self.dataset_path,
            "format": self.file_format,
            "num_samples": num_samples,
            "task": self.task_type,
            "loaded": True,
            "description": f"Universal dataset: {os.path.basename(self.dataset_path)}",
            "text_column": self.text_column,
            "label_columns": self.label_columns,
            "label_statistics": label_stats,
            "max_length": self.max_length
        }
    
    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the expected input shape for models."""
        return (self.max_length,) 