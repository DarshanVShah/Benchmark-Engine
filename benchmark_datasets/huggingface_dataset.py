"""
Generic HuggingFace Dataset Adapter

"""

import random
from typing import Any, Dict, List, Optional, Tuple

# Use absolute import to avoid conflict with local datasets package
import datasets as hf_datasets

from core import BaseDataset, DataType


class HuggingFaceDataset(BaseDataset):
    """
    Generic dataset adapter for any HuggingFace Hub dataset.

    """

    def __init__(self):
        self.dataset = None
        self.dataset_loaded = False
        self.dataset_name = ""
        self.max_length = 512
        self.task_type = "auto-detected"
        self.text_column = None
        self.label_columns = []

    @property
    def output_type(self) -> DataType:
        """HuggingFace datasets output text."""
        return DataType.TEXT

    def load(self, dataset_path: str) -> bool:
        """Load any dataset from HuggingFace Hub."""
        try:
            print(f"Loading HuggingFace dataset: {dataset_path}")

            # Load dataset from HuggingFace
            self.dataset = hf_datasets.load_dataset(dataset_path)
            self.dataset_name = dataset_path

            # Get the validation split for evaluation (or fallback to train)
            if "validation" in self.dataset:
                self.data = self.dataset["validation"]
                print(f"  Using validation split for evaluation")
            elif "test" in self.dataset:
                self.data = self.dataset["test"]
                print(f"  Using test split for evaluation")
            elif "train" in self.dataset:
                self.data = self.dataset["train"]
                print(f"  Warning: Using train split (no validation/test available)")
            elif "default" in self.dataset:
                self.data = self.dataset["default"]
                print(f"  Using default split")
            else:
                # Use first available split
                split_name = list(self.dataset.keys())[0]
                self.data = self.dataset[split_name]
                print(f"  Using {split_name} split")

            # Auto-detect dataset structure
            self._auto_detect_format()

            self.dataset_loaded = True
            print(f"  HuggingFace dataset loaded: {dataset_path}")
            print(f"  Samples: {len(self.data)}")
            print(f"  Features: {list(self.data.features.keys())}")
            print(f"  Detected task: {self.task_type}")
            print(f"  Text column: {self.text_column}")
            print(f"  Label columns: {self.label_columns}")

            return True

        except Exception as e:
            print(f" Failed to load dataset {dataset_path}: {e}")
            return False

    def _auto_detect_format(self):
        """Automatically detect dataset format and structure."""
        features = self.data.features

        # Detect text column
        text_candidates = ["text", "content", "sentence", "input", "message", "post"]
        for candidate in text_candidates:
            if candidate in features:
                self.text_column = candidate
                break

        # If no text column found, use first string column
        if not self.text_column:
            for key, feature in features.items():
                if hasattr(feature, "dtype") and "string" in str(feature.dtype):
                    self.text_column = key
                    break

        # Detect label columns
        label_candidates = [
            "label",
            "labels",
            "valence",
            "arousal",
            "dominance",
            "emotion",
            "sentiment",
        ]
        for candidate in label_candidates:
            if candidate in features:
                self.label_columns.append(candidate)

        # Detect task type based on labels
        if "valence" in self.label_columns or "arousal" in self.label_columns:
            self.task_type = "emotion-detection"
        elif "label" in self.label_columns or "labels" in self.label_columns:
            self.task_type = "text-classification"
        elif "sentiment" in self.label_columns:
            self.task_type = "sentiment-analysis"
        else:
            self.task_type = "text-processing"

    def get_samples(self, num_samples: Optional[int] = None) -> List[Any]:
        """Get text samples from the dataset."""
        if not self.dataset_loaded:
            raise RuntimeError("Dataset not loaded")

        samples = []
        # Convert to list to ensure we get actual data rows
        if hasattr(self.data, "select"):
            # Use select method for HuggingFace datasets
            indices = list(range(min(num_samples or len(self.data), len(self.data))))
            data_subset = self.data.select(indices)
        else:
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

    def get_samples_with_targets(
        self, num_samples: Optional[int] = None
    ) -> List[Tuple[Any, Any]]:
        """Get (input, target) pairs for evaluation."""
        if not self.dataset_loaded:
            raise RuntimeError("Dataset not loaded")

        samples_with_targets = []
        # Convert to list to ensure we get actual data rows
        if hasattr(self.data, "select"):
            # Use select method for HuggingFace datasets
            indices = list(range(min(num_samples or len(self.data), len(self.data))))
            data_subset = self.data.select(indices)
        else:
            data_subset = self.data[:num_samples] if num_samples else self.data

        for item in data_subset:
            # Handle case where item might be a string or other format
            if isinstance(item, str):
                text = item
                target = 0  # Default target for string items
            elif isinstance(item, dict):
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

                # Extract targets based on task type
                target = self._extract_targets(item)
            else:
                # Handle other formats
                text = str(item)
                target = 0

            # Skip if text is suspicious (column names)
            if text in ["idx", "sentence", "label", "text", "content", "input"]:
                continue

            # Create input format
            input_sample = {"text": text[: self.max_length]}

            samples_with_targets.append((input_sample, target))

        return samples_with_targets

    def _extract_targets(self, item: Dict[str, Any]) -> Any:
        """Extract targets based on detected task type."""
        if self.task_type == "emotion-detection":
            # Extract VAD values
            return {
                "valence": item.get("valence", 0.0),
                "arousal": item.get("arousal", 0.0),
                "dominance": item.get("dominance", 0.0),
            }
        elif self.task_type == "text-classification":
            # Extract classification labels
            if "label" in item:
                return item["label"]
            elif "labels" in item:
                return item["labels"]
            else:
                return 0  # Default label
        elif self.task_type == "sentiment-analysis":
            # Extract sentiment
            return item.get("sentiment", 0)
        else:
            # Generic: return first numeric field as target
            for key, value in item.items():
                if isinstance(value, (int, float)) and key != "id":
                    return value
            return 0  # Default target

    def get_dataset_info(self) -> Dict[str, Any]:
        """Return dataset information."""
        if not self.dataset_loaded:
            return {"name": "HuggingFaceDataset", "loaded": False}

        # Calculate basic statistics
        num_samples = len(self.data)
        features = list(self.data.features.keys())

        info = {
            "name": f"HuggingFaceDataset-{random.randint(100, 999)}",
            "type": "huggingface",
            "source": self.dataset_name,
            "num_samples": num_samples,
            "task": self.task_type,
            "loaded": True,
            "description": f"Auto-detected HuggingFace dataset: {self.dataset_name}",
            "features": features,
            "text_column": self.text_column,
            "label_columns": self.label_columns,
            "max_length": self.max_length,
        }

        # Add task-specific statistics
        if self.task_type == "emotion-detection":
            valence_values = [
                item.get("valence", 0.0) for item in self.data if "valence" in item
            ]
            arousal_values = [
                item.get("arousal", 0.0) for item in self.data if "arousal" in item
            ]
            dominance_values = [
                item.get("dominance", 0.0) for item in self.data if "dominance" in item
            ]

            if valence_values:
                info["vad_statistics"] = {
                    "avg_valence": sum(valence_values) / len(valence_values),
                    "avg_arousal": (
                        sum(arousal_values) / len(arousal_values)
                        if arousal_values
                        else 0
                    ),
                    "avg_dominance": (
                        sum(dominance_values) / len(dominance_values)
                        if dominance_values
                        else 0
                    ),
                }

        return info

    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the expected input shape for models."""
        return (self.max_length,)
