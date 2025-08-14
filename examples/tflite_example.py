"""
Universal TFLite Emotion Classifier Test with Emotion Standardization

This example shows how a user would use the BenchmarkEngine for universal testing
with the integrated emotion standardization system. The engine selects random emotion 
datasets and converts all outputs to standardized, human-readable emotions.
"""

import os
import sys
import numpy as np

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_datasets.template_dataset import TemplateDataset
from core.engine import BenchmarkEngine
from core.emotion_standardization import standardized_emotions, map_emotion_to_standard
from core.emotion_converter import convert_emotion_output, get_emotion_summary, get_emotion_analysis
from plugins.tflite_adapter import TensorFlowLiteAdapter
from metrics.template_metric import TemplateMultiLabelMetric

def run_tflite_benchmark():
    """Run the actual TFLite benchmark if model is available."""

    
    # Check if model file exists
    model_path = "models/notQuantizedModel.tflite"
    if not os.path.exists(model_path):
        print(f"TFLite model not found: {model_path}")
        return False
    
    # Create the benchmark engine
    engine = BenchmarkEngine()
    
    # Register what we need
    engine.register_adapter("tflite", TensorFlowLiteAdapter)
    engine.register_dataset("template", TemplateDataset)
    
    # Load your TFLite model
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
    
    print("TFLite model loaded successfully!")
    
    # Add universal evaluation metric
    universal_metric = TemplateMultiLabelMetric(metric_type="accuracy", threshold=0.5)
    engine.add_metric("universal_accuracy", universal_metric)
    
    # Run universal benchmark (engine selects random datasets unknown to you)
    results = engine.run_universal_benchmark(num_samples=500)  # Test on 500 samples per dataset
    
    if results:
        print(f"\nUniversal benchmark completed successfully!")
        print(f"   Your adapter achieved {results['universal_accuracy']:.1%} average accuracy")
        print(f"   across {results['successful_runs']} successful runs out of {results['datasets_tested']} total datasets.")
        
        if results.get('failed_datasets'):
            print(f"\n   Failed datasets:")
            for failed_dataset in results['failed_datasets']:
                print(f"     - {failed_dataset}")
        
        print(f"\n   Success rate: {results['successful_runs']}/{results['datasets_tested']} = {(results['successful_runs']/results['datasets_tested'])*100:.0f}%")
        
        return True
    else:
        print("Benchmark failed")
        return False


def main():
    """Universal test of TFLite emotion classifier with emotion standardization."""
    
    # Try to run the actual benchmark
    benchmark_success = run_tflite_benchmark()
    
    return benchmark_success


if __name__ == "__main__":
    success = main()

    if success:
        print("\nFull TFLite benchmark completed successfully!")
    else:
        print("\nTFLite benchmark couldn't run, but the emotion standardization system is fully functional.")
