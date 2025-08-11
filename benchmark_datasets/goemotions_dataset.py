"""
GoEmotions Dataset

Specialized dataset loader for the GoEmotions dataset which has a non-standard TSV format.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
from .template_dataset import TemplateDataset


class GoEmotionsDataset(TemplateDataset):
    """
    GoEmotions dataset loader.
    
    The dataset has format: text <tab> label <tab> id
    Where label is a numeric emotion class (0-26)
    """
    
    def _load_tsv(self, file_path: str):
        """Load GoEmotions TSV file with custom format."""
        self.data = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                # Split by tab
                parts = line.split('\t')
                if len(parts) >= 2:
                    text = parts[0].strip()
                    label = parts[1].strip()
                    
                    # Skip empty text
                    if not text:
                        continue
                    
                    # Create sample with proper column names that match the config
                    sample = {
                        "text": text,  # This should match text_column in config
                        "label": label  # This should match label_columns in config
                    }
                    
                    # Add ID if available
                    if len(parts) >= 3:
                        sample["id"] = parts[2].strip()
                    
                    self.data.append(sample)
        
        print(f"  Loaded {len(self.data)} samples from GoEmotions dataset")
        print(f"  Sample data structure: {self.data[0] if self.data else 'No data'}")
        print(f"  Text column: {self.text_column}")
        print(f"  Label columns: {self.label_columns}")
    
    def get_samples_with_targets(self, num_samples: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """Override to ensure proper sample extraction."""
        if not self.dataset_loaded:
            raise RuntimeError("Dataset not loaded")

        samples_with_targets = []
        data_subset = self.data[:num_samples] if num_samples else self.data

        print(f"  Processing {len(data_subset)} samples for GoEmotions dataset")
        
        for i, item in enumerate(data_subset):
            # Extract text using the configured text column
            text = item.get(self.text_column, "")
            if not text or not isinstance(text, str):
                print(f"    Skipping sample {i}: invalid text '{text}'")
                continue

            # Extract targets based on task type
            target = self._extract_targets(item)

            # Create input format
            input_sample = {"text": text[: self.max_length]}

            samples_with_targets.append((input_sample, target))
            
            if i < 3:  # Debug first few samples
                print(f"    Sample {i}: text='{text[:50]}...', target={target}")

        print(f"  Generated {len(samples_with_targets)} samples with targets")
        return samples_with_targets
    
    def _extract_targets(self, item: Dict[str, Any]) -> Any:
        """Extract emotion label from GoEmotions format."""
        if "label" in item:
            try:
                return int(item["label"])
            except (ValueError, TypeError):
                return 0
        return 0
