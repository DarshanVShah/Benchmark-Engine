"""
Local Dataset Adapter

This adapter can load local datasets in various formats (CSV, JSON, TXT)
and automatically detect the structure for benchmarking.
"""

import os
import csv
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from core import BaseDataset


class LocalTextDataset(BaseDataset):
    """
    Local dataset adapter for CSV, JSON, and text files.
    
    Supports:
    - CSV files with text/label columns
    - JSON files with text data
    - Text files (one sample per line)
    - Custom formats with auto-detection
    
    Users just specify the file path - no custom code needed!
    """
    
    def __init__(self):
        self.data = []
        self.dataset_loaded = False
        self.file_path = ""
        self.file_format = "auto-detected"
        self.text_column = None
        self.label_columns = []
        self.max_length = 512
        self.task_type = "auto-detected"
        
    def load(self, dataset_path: str) -> bool:
        """Load local dataset from file path."""
        try:
            print(f"Loading local dataset: {dataset_path}")
            
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
            self.file_path = dataset_path
            file_extension = os.path.splitext(dataset_path)[1].lower()
            
            # Load based on file extension
            if file_extension == '.csv':
                self._load_csv(dataset_path)
            elif file_extension == '.json':
                self._load_json(dataset_path)
            elif file_extension == '.txt':
                self._load_txt(dataset_path)
            else:
                # Try to auto-detect format
                self._auto_detect_and_load(dataset_path)
            
            # Auto-detect structure
            self._auto_detect_format()
            
            self.dataset_loaded = True
            print(f"  Local dataset loaded: {dataset_path}")
            print(f"  Samples: {len(self.data)}")
            print(f"  Format: {self.file_format}")
            print(f"  Detected task: {self.task_type}")
            print(f"  Text column: {self.text_column}")
            print(f"  Label columns: {self.label_columns}")
            
            return True
            
        except Exception as e:
            print(f" Failed to load dataset {dataset_path}: {e}")
            return False
    
    def _load_csv(self, file_path: str):
        """Load CSV file."""
        self.file_format = "csv"
        
        # Try pandas first (handles various CSV formats)
        try:
            df = pd.read_csv(file_path)
            self.data = df.to_dict('records')
        except:
            # Fallback to csv module
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
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
            # Assume format like {"data": [...], "labels": [...]}
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
                # Single sample or other format
                self.data = [content]
        else:
            self.data = [content]
    
    def _load_txt(self, file_path: str):
        """Load text file (one sample per line)."""
        self.file_format = "txt"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        self.data = []
        for i, line in enumerate(lines):
            line = line.strip()
            if line:  # Skip empty lines
                self.data.append({
                    "text": line,
                    "label": 0  # Default label
                })
    
    def _auto_detect_and_load(self, file_path: str):
        """Auto-detect file format and load."""
        # Try different formats
        try:
            self._load_csv(file_path)
            return
        except:
            pass
        
        try:
            self._load_json(file_path)
            return
        except:
            pass
        
        try:
            self._load_txt(file_path)
            return
        except:
            pass
        
        raise ValueError(f"Could not auto-detect format for {file_path}")
    
    def _auto_detect_format(self):
        """Auto-detect dataset structure."""
        if not self.data:
            return
        
        # Get sample to analyze structure
        sample = self.data[0]
        
        # Detect text column
        text_candidates = ["text", "content", "sentence", "input", "message", "post", "review", "comment"]
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
        for candidate in label_candidates:
            if candidate in sample:
                self.label_columns.append(candidate)
        
        # Detect task type
        if self.label_columns:
            if any("sentiment" in col.lower() for col in self.label_columns):
                self.task_type = "sentiment-analysis"
            elif any("emotion" in col.lower() for col in self.label_columns):
                self.task_type = "emotion-detection"
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
                samples.append({"text": item[self.text_column]})
            else:
                # Fallback: use first text-like field
                for key, value in item.items():
                    if isinstance(value, str) and len(value) > 10:
                        samples.append({"text": value})
                        break
        
        return samples
    
    def get_samples_with_targets(self, num_samples: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """Get (input, target) pairs for evaluation."""
        if not self.dataset_loaded:
            raise RuntimeError("Dataset not loaded")
        
        samples_with_targets = []
        data_subset = self.data[:num_samples] if num_samples else self.data
        
        for item in data_subset:
            # Extract text input
            text = ""
            if self.text_column and self.text_column in item:
                text = item[self.text_column]
            else:
                # Find first text field
                for key, value in item.items():
                    if isinstance(value, str) and len(value) > 10:
                        text = value
                        break
            
            if not text:
                continue
            
            # Extract targets
            target = self._extract_targets(item)
            
            # Create input format
            input_sample = {"text": text[:self.max_length]}
            
            samples_with_targets.append((input_sample, target))
        
        return samples_with_targets
    
    def _extract_targets(self, item: Dict[str, Any]) -> Any:
        """Extract targets based on detected task type."""
        if self.task_type == "text-classification":
            # Extract classification labels
            for label_col in self.label_columns:
                if label_col in item:
                    return item[label_col]
            return 0  # Default label
        elif self.task_type == "sentiment-analysis":
            # Extract sentiment
            for label_col in self.label_columns:
                if label_col in item:
                    return item[label_col]
            return 0  # Default sentiment
        else:
            # Generic: return first numeric field as target
            for key, value in item.items():
                if isinstance(value, (int, float)) and key != "id":
                    return value
            return 0  # Default target
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return dataset information."""
        if not self.dataset_loaded:
            return {"name": "LocalTextDataset", "loaded": False}
        
        # Calculate basic statistics
        num_samples = len(self.data)
        
        info = {
            "name": f"LocalDataset-{os.path.basename(self.file_path)}",
            "type": "local",
            "source": self.file_path,
            "format": self.file_format,
            "num_samples": num_samples,
            "task": self.task_type,
            "loaded": True,
            "description": f"Local dataset: {self.file_path}",
            "text_column": self.text_column,
            "label_columns": self.label_columns,
            "max_length": self.max_length
        }
        
        return info
    
    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the expected input shape for models."""
        return (self.max_length,)