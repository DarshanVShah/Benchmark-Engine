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
            # Validate configuration
            required_keys = ["file_format", "text_column", "label_columns", "task_type"]
            for key in required_keys:
                if key not in config:
                    return False

            # Store configuration
            self.file_format = config["file_format"]
            self.text_column = config["text_column"]
            self.label_columns = config["label_columns"]
            self.task_type = config["task_type"]
            self.max_length = config.get("max_length", 512)
            self.skip_header = config.get("skip_header", False)  # Store skip_header setting

            # Check if file exists
            if not os.path.exists(dataset_path):
                return False

            # Load data based on format
            if self.file_format == "csv":
                self._load_csv(dataset_path)
            elif self.file_format == "tsv":
                self._load_tsv(dataset_path)
            elif self.file_format == "json":
                self._load_json(dataset_path)
            else:
                return False

            self.dataset_path = dataset_path
            self.dataset_loaded = True

            return True

        except Exception as e:
            return False

    def _load_csv(self, file_path: str):
        """Load CSV file with flexible column handling."""
        self.data = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            if not rows:
                return
            
            # Handle header row
            if hasattr(self, 'skip_header') and self.skip_header:
                headers = rows[0]
                rows = rows[1:]  # Skip header
                
                # Find column indices - handle both column names and indices
                text_col_idx = None
                label_col_indices = {}
                
                # Check if text_column is a name or index
                if isinstance(self.text_column, str):
                    # Column name
                    for i, header in enumerate(headers):
                        if header == self.text_column:
                            text_col_idx = i
                            break
                elif isinstance(self.text_column, int):
                    # Column index
                    text_col_idx = self.text_column
                
                # Check if label_columns contains names or indices
                if self.label_columns:
                    if isinstance(self.label_columns[0], str):
                        # Column names
                        for i, header in enumerate(headers):
                            if header in self.label_columns:
                                label_col_indices[header] = i
                    elif isinstance(self.label_columns[0], int):
                        # Column indices
                        for col_idx in self.label_columns:
                            if col_idx < len(headers):
                                label_col_indices[f"label_{col_idx}"] = col_idx
            else:
                # No header - assume first column is text, rest are labels
                text_col_idx = 0
                if isinstance(self.text_column, int):
                    text_col_idx = self.text_column
                
                # Handle label columns
                if self.label_columns:
                    if isinstance(self.label_columns[0], int):
                        label_col_indices = {f"label_{col}": col for col in self.label_columns}
                    else:
                        label_col_indices = {col: i+1 for i, col in enumerate(self.label_columns)}
                else:
                    # Default: assume second column is label
                    label_col_indices = {"label": 1}
            
            # Process data rows
            for row in rows:
                if len(row) < 2:
                    continue
                
                # Extract text
                if text_col_idx < len(row):
                    text = row[text_col_idx].strip()
                    if not text:
                        continue
                else:
                    continue
                
                # Create sample with proper structure
                sample = {"text": text}  # Always use "text" as key for compatibility
                
                # Extract labels
                for label_col, col_idx in label_col_indices.items():
                    if col_idx < len(row):
                        label_value = row[col_idx].strip()
                        # Convert to int for binary labels (0/1)
                        try:
                            sample[label_col] = int(label_value)
                        except ValueError:
                            sample[label_col] = 0  # Default to 0 if conversion fails
                    else:
                        sample[label_col] = 0  # Default to 0 if column missing
                
                self.data.append(sample)
        
        # Dataset loaded successfully

    def _load_tsv(self, file_path: str):
        """Load TSV file with flexible column handling."""
        self.data = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
            # Handle header row
            if lines and hasattr(self, 'skip_header') and self.skip_header:
                header_line = lines[0].strip()
                headers = header_line.split('\t')
                lines = lines[1:]  # Skip header
                
                # Find column indices - handle both column names and indices
                text_col_idx = None
                label_col_indices = {}
                
                # Check if text_column is a name or index
                if isinstance(self.text_column, str):
                    # Column name
                    for i, header in enumerate(headers):
                        if header == self.text_column:
                            text_col_idx = i
                            break
                elif isinstance(self.text_column, int):
                    # Column index
                    text_col_idx = self.text_column
                
                # Check if label_columns contains names or indices
                if self.label_columns:
                    if isinstance(self.label_columns[0], str):
                        # Column names
                        for i, header in enumerate(headers):
                            if header in self.label_columns:
                                label_col_indices[header] = i
                    elif isinstance(self.label_columns[0], int):
                        # Column indices
                        for col_idx in self.label_columns:
                            if col_idx < len(headers):
                                label_col_indices[f"label_{col_idx}"] = col_idx
                
                if text_col_idx is None:
                    return
            else:
                # No header - assume first column is text, rest are labels
                text_col_idx = 0
                if isinstance(self.text_column, int):
                    text_col_idx = self.text_column
                
                # Handle label columns
                if self.label_columns:
                    if isinstance(self.label_columns[0], int):
                        label_col_indices = {f"label_{col}": col for col in self.label_columns}
                    else:
                        label_col_indices = {col: i+1 for i, col in enumerate(self.label_columns)}
                else:
                    # Default: assume second column is label
                    label_col_indices = {"label": 1}
            
            # Process data lines
            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Split by tab
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                # Extract text
                if text_col_idx < len(parts):
                    text = parts[text_col_idx].strip()
                    if not text:
                        continue
                else:
                    continue
                
                # Create sample with proper structure
                sample = {"text": text}  # Always use "text" as key for compatibility
                
                # Extract labels
                for label_col, col_idx in label_col_indices.items():
                    if col_idx < len(parts):
                        label_value = parts[col_idx].strip()
                        # Convert to int for binary labels (0/1)
                        try:
                            sample[label_col] = int(label_value)
                        except ValueError:
                            sample[label_col] = 0  # Default to 0 if conversion fails
                    else:
                        sample[label_col] = 0  # Default to 0 if column missing
                
                self.data.append(sample)
        
        # Dataset loaded successfully

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
            # Always use "text" key since we standardize the data structure
            text = item.get("text", "")
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
            # Extract text - always use "text" key since we standardize the data structure
            text = item.get("text", "")
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
