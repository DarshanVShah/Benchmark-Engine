"""
Template Dataset Pattern

This module provides a clean template pattern for datasets.
Users explicitly select the format and structure
"""

import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from core import BaseDataset, DataType


class TemplateDataset(BaseDataset):
    """
    Template dataset that requires explicit configuration.

    Users must specify:
    - File format (csv, tsv, json)
    - Text column name
    - Label column names
    - Task type

    """

    def __init__(self):
        self.data = []
        self.dataset_loaded = False
        self.dataset_path = ""
        self.file_format = None
        self.text_column = None
        self.label_columns = []
        self.task_type = None
        self.max_length = 512

    @property
    def output_type(self) -> DataType:
        """Template datasets output text."""
        return DataType.TEXT

    def load(self, dataset_path: str, config: Dict[str, Any]) -> bool:
        """
        Load dataset with explicit configuration.

        Args:
            dataset_path: Path to dataset file
            config: Explicit configuration dict with:
                - file_format: "csv", "tsv", "json"
                - text_column: Name of text column
                - label_columns: List of label column names
                - task_type: "classification", "regression", "multi-label"
                - max_length: Optional max text length
        """
        try:
            print(f"Loading template dataset: {dataset_path}")

            # Validate configuration
            required_keys = ["file_format", "text_column", "label_columns", "task_type"]
            for key in required_keys:
                if key not in config:
                    print(f"  Error: Missing required config key: {key}")
                    return False

            # Store configuration
            self.file_format = config["file_format"]
            self.text_column = config["text_column"]
            self.label_columns = config["label_columns"]
            self.task_type = config["task_type"]
            self.max_length = config.get("max_length", 512)

            # Check if file exists
            if not os.path.exists(dataset_path):
                print(f"  Error: Dataset file not found: {dataset_path}")
                return False

            # Load data based on format
            if self.file_format == "csv":
                self._load_csv(dataset_path)
            elif self.file_format == "tsv":
                self._load_tsv(dataset_path)
            elif self.file_format == "json":
                self._load_json(dataset_path)
            else:
                print(f"  Error: Unsupported file format: {self.file_format}")
                return False

            self.dataset_path = dataset_path
            self.dataset_loaded = True

            print(f"  Template dataset loaded: {dataset_path}")
            print(f"  Format: {self.file_format}")
            print(f"  Samples: {len(self.data)}")
            print(f"  Task: {self.task_type}")
            print(f"  Text column: {self.text_column}")
            print(f"  Label columns: {self.label_columns}")

            return True

        except Exception as e:
            print(f"  Error loading template dataset: {e}")
            return False

    def _load_csv(self, file_path: str):
        """Load CSV file."""
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.data = list(reader)

    def _load_tsv(self, file_path: str):
        """Load TSV file with flexible column handling."""
        self.data = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                # Split by tab
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                # Extract text and labels
                text = parts[0].strip()
                if not text:
                    continue
                
                # Create sample with proper structure
                sample = {self.text_column: text}
                
                # Handle different label formats
                if len(parts) >= 2:
                    label_value = parts[1].strip()
                    
                    # Apply label mapping if specified
                    if hasattr(self, 'label_mapping') and self.label_mapping:
                        label_value = self.label_mapping.get(label_value, label_value)
                    
                    # Add label to sample
                    for label_col in self.label_columns:
                        sample[label_col] = label_value
                
                # Add additional columns if available
                for i, col_name in enumerate(self.label_columns[1:], start=2):
                    if i < len(parts):
                        sample[col_name] = parts[i].strip()
                
                self.data.append(sample)
        
        print(f"  Template dataset loaded: {file_path}")
        print(f"  Format: tsv")
        print(f"  Samples: {len(self.data)}")
        print(f"  Task: {self.task_type}")
        print(f"  Text column: {self.text_column}")
        print(f"  Label columns: {self.label_columns}")
        
        if self.data:
            print(f"  Sample data: {self.data[0]}")

    def _load_json(self, file_path: str):
        """Load JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
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

    def get_samples(self, num_samples: Optional[int] = None) -> List[Any]:
        """Get text samples from the dataset."""
        if not self.dataset_loaded:
            raise RuntimeError("Dataset not loaded")

        samples = []
        data_subset = self.data[:num_samples] if num_samples else self.data

        for item in data_subset:
            if self.text_column in item:
                text = item[self.text_column]
                if isinstance(text, str) and len(text) > 0:
                    samples.append({"text": text[: self.max_length]})

        return samples

    def get_samples_with_targets(
        self, num_samples: Optional[int] = None
    ) -> List[Tuple[Any, Any]]:
        """Get (input, target) pairs for evaluation."""
        if not self.dataset_loaded:
            raise RuntimeError("Dataset not loaded")

        samples_with_targets = []
        data_subset = self.data[:num_samples] if num_samples else self.data

        for item in data_subset:
            # Extract text
            text = item.get(self.text_column, "")
            if not text or not isinstance(text, str):
                continue

            # Extract targets based on task type
            target = self._extract_targets(item)

            # Create input format
            input_sample = {"text": text[: self.max_length]}

            samples_with_targets.append((input_sample, target))

        return samples_with_targets

    def _extract_targets(self, item: Dict[str, Any]) -> Any:
        """Extract targets based on configured task type."""
        if self.task_type == "multi-label":
            # Extract multiple labels
            labels = {}
            for label_col in self.label_columns:
                if label_col in item:
                    labels[label_col] = item[label_col]
            return labels

        elif self.task_type == "classification":
            # Extract single label
            for label_col in self.label_columns:
                if label_col in item:
                    return item[label_col]
            return 0

        elif self.task_type == "regression":
            # Extract regression value
            for label_col in self.label_columns:
                if label_col in item:
                    return float(item[label_col])
            return 0.0

        else:
            # Generic: return first numeric field as target
            for key, value in item.items():
                if isinstance(value, (int, float)) and key != "id":
                    return value
            return 0

    def get_dataset_info(self) -> Dict[str, Any]:
        """Return dataset information."""
        if not self.dataset_loaded:
            return {"name": "TemplateDataset", "loaded": False}

        return {
            "name": "TemplateDataset",
            "type": "template",
            "source": self.dataset_path,
            "format": self.file_format,
            "num_samples": len(self.data),
            "task": self.task_type,
            "loaded": True,
            "description": f"Template dataset: {os.path.basename(self.dataset_path)}",
            "text_column": self.text_column,
            "label_columns": self.label_columns,
            "max_length": self.max_length,
        }

    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the expected input shape for models."""
        return (self.max_length,)
