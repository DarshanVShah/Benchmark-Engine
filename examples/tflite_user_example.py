"""
Universal TFLite Emotion Classifier Test

This example shows how a user would use the BenchmarkEngine for universal testing.
The engine selects random emotion datasets unknown to you and creates a standardized
evaluation environment. Your adapter must work with the engine's standard format.
"""

import os
import sys

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import BenchmarkEngine
from plugins.tflite_adapter import TensorFlowLiteAdapter
from benchmark_datasets.template_dataset import TemplateDataset
from metrics.template_metric import TemplateMultiLabelMetric


def main():
    """Universal test of TFLite emotion classifier."""
    
    # Create the benchmark engine
    engine = BenchmarkEngine()
    
    # Register what we need
    engine.register_adapter("tflite", TensorFlowLiteAdapter)
    engine.register_dataset("template", TemplateDataset)
    
    # Load your TFLite model
    model_path = "models/notQuantizedModel.tflite"
    model_config = {
        "device": "cpu",
        "precision": "fp32",
        "max_length": 512,  # Use larger max length for universal testing
        "input_type": "text",
        "output_type": "probabilities",
        "task_type": "multi-label",
        "is_multi_label": True
    }
    
    if not engine.load_model("tflite", model_path, model_config):
        print("Failed to load TFLite model")
        return False
    
    # Add universal evaluation metric
    universal_metric = TemplateMultiLabelMetric(metric_type="accuracy", threshold=0.5)
    engine.add_metric("universal_accuracy", universal_metric)
    
    # Run universal benchmark (engine selects random datasets unknown to you)
    print("🎯 Starting Universal Emotion Benchmark")
    print("The engine will select random emotion datasets and create a standardized format.")
    print("Your adapter must work with the engine's standard - you won't know the datasets!")
    print()
    
    results = engine.run_universal_benchmark(num_samples=500)  # Test on 500 samples per dataset
    
    if results:
        print(f"\n🎉 Universal benchmark completed successfully!")
        print(f"Your adapter achieved {results['universal_accuracy']:.1%} average accuracy")
        print(f"across {results['datasets_tested']} unknown emotion datasets.")
        return True
    else:
        print("Universal benchmark failed")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\nAll done! Check the universal benchmark results for detailed analysis.")
    else:
        print("\nSomething went wrong. Check the error messages above.")
        sys.exit(1)
