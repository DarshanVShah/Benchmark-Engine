"""
Quick TFLite Test - Minimal Example

This is the absolute simplest way to test your TFLite emotion classifier.
Just 3 lines of code to get started!
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import BenchmarkEngine
from plugins.tflite_adapter import TensorFlowLiteAdapter
from benchmark_datasets.template_dataset import TemplateDataset
from metrics.template_metric import TemplateAccuracyMetric

# Create engine and test your model in just a few lines!
engine = BenchmarkEngine()

# Register what you need
engine.register_adapter("tflite", TensorFlowLiteAdapter)
engine.register_dataset("template", TemplateDataset)

# Load your model
engine.load_model("tflite", "models/notQuantizedModel.tflite", {
    "input_type": "text",
    "output_type": "class_id",
    "task_type": "single-label"
})

# Load dataset (will auto-download if needed)
engine.load_dataset("template", "benchmark_datasets/localTestSets/goemotions_test.tsv", {
    "file_format": "tsv",
    "text_column": "text",
    "label_columns": ["label"]
})

# Add accuracy metric
engine.add_metric("accuracy", TemplateAccuracyMetric(input_type="class_id"))

# Run test on 50 samples
print("🚀 Testing your TFLite emotion classifier...")
results = engine.run_benchmark(num_samples=50)

if results:
    accuracy = results["metrics"]["TemplateAccuracyMetric"]
    print(f"🎯 Your model accuracy: {accuracy:.2%}")
    print(f"⏱️  Speed: {results['timing']['throughput']:.1f} samples/second")
    
    # Save results
    engine.export_results("quick_test_results.json")
    print("💾 Results saved!")
else:
    print("❌ Test failed")

