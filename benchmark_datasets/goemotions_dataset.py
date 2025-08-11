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
                    
                    # Create sample
                    sample = {
                        "text": text,
                        "label": label
                    }
                    
                    # Add ID if available
                    if len(parts) >= 3:
                        sample["id"] = parts[2].strip()
                    
                    self.data.append(sample)
        
        print(f"  Loaded {len(self.data)} samples from GoEmotions dataset")
    
    def _extract_targets(self, item: Dict[str, Any]) -> Any:
        """Extract emotion label from GoEmotions format."""
        if "label" in item:
            try:
                return int(item["label"])
            except (ValueError, TypeError):
                return 0
        return 0
