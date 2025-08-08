"""
Test script to verify local dataset functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import BenchmarkEngine
from core.dataset_registry import TaskType
from plugins import HuggingFaceAdapter
from benchmark_datasets import TemplateDataset
from metrics.template_metric import TemplateMultiLabelMetric


def test_local_emotion_dataset():
    """Test the local emotion dataset with HuggingFace model."""
    print("TESTING LOCAL EMOTION DATASET")
    
    # Create engine
    engine = BenchmarkEngine()
    
    # Register components
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_dataset("template", TemplateDataset)
    engine.register_metric("multilabel", TemplateMultiLabelMetric)
    
    # Load model
    model_path = "distilbert-base-uncased"
    model_config = {
        "device": "cpu",
        "precision": "fp32",
        "max_length": 128,
        "is_multi_label": True,  # Important for emotion dataset
        "task_type": "multi-label"
    }
    
    success = engine.load_model("huggingface", model_path, model_config)
    if not success:
        print("Failed to load model")
        return False
    
    dataset_config = {
        "file_format": "tsv",
        "text_column": "Tweet",
        "label_columns": ["anger", "anticipation", "disgust", "fear", "joy", 
                         "love", "optimism", "pessimism", "sadness", "surprise", "trust"],
        "task_type": "multi-label",
        "max_length": 128
    }
    
    dataset_path = "benchmark_datasets/localTestSets/2018-E-c-En-test-gold.txt"
    success = engine.load_dataset("template", dataset_path, dataset_config)
    if not success:
        print("Failed to load dataset")
        return False
    
    multilabel_metric = TemplateMultiLabelMetric(metric_type="accuracy", threshold=0.5)
    engine.metrics = [multilabel_metric]
    
    success = engine.validate_setup()
    if not success:
        print("Setup validation failed")
        return False
    
    print("All components are compatible!")
    print(f"  - Dataset outputs: {engine.dataset.output_type.value}")
    print(f"  - Model expects: {engine.model_adapter.input_type.value}")
    print(f"  - Model outputs: {engine.model_adapter.output_type.value}")
    print(f"  - Metric expects: {engine.metrics[0].expected_input_type.value}")
    
    try:
        engine.configure_benchmark({
            "num_samples": 5,  # Test on 5 samples
            "warmup_runs": 1,
            "batch_size": 1,
            "precision": "fp32",
            "device": "cpu"
        })
        
        results = engine.run_benchmark()
        engine.print_results()
        
        print("Local dataset test completed successfully.")
        return True
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return False


def main():
    """Main test function."""
    print("LOCAL DATASET TEST")
    print("Testing the local emotion dataset with the refactored framework")
    
    success = test_local_emotion_dataset()
    
    if success:
        print("\nAll tests passed!")
    else:
        print("\nTests failed!")


if __name__ == "__main__":
    main()
