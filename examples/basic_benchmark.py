"""
Basic benchmark example demonstrating the BenchmarkEngine framework.

This example shows how to:
1. Create a benchmark engine
2. Register plugins (adapters, metrics, datasets)
3. Load a model and dataset
4. Run a benchmark
5. Export results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BenchmarkEngine
from plugins import DummyModelAdapter, DummyAccuracyMetric, DummyDataset


def main():
    """Run a basic benchmark using dummy plugins."""
    
    print("BenchmarkEngine Framework Demo")
    print("=" * 50)
    
    # Create the benchmark engine
    engine = BenchmarkEngine()
    
    # Register plugins (this is where the extensibility comes in!)
    print("\nRegistering plugins...")
    engine.register_adapter("dummy", DummyModelAdapter)
    engine.register_metric("accuracy", DummyAccuracyMetric)
    engine.register_dataset("dummy", DummyDataset)
    
    # Load model and dataset
    print("\nLoading model and dataset...")
    engine.load_model("dummy", "path/to/dummy/model")
    engine.load_dataset("dummy", "path/to/dummy/dataset")
    
    # Add metrics
    print("\nAdding metrics...")
    engine.add_metric("accuracy")
    
    # Run benchmark
    print("\nRunning benchmark...")
    results = engine.run_benchmark(num_samples=20, warmup_runs=3)
    
    # Display results
    print("\nResults:")
    engine.print_results()
    
    # Export results
    print("\nExporting results...")
    engine.export_results("benchmark_results.json", format="json")
    engine.export_results("benchmark_results.md", format="markdown")
    
    print("\nBenchmark completed successfully!")
    print("\nFiles created:")
    print("  - benchmark_results.json (JSON format)")
    print("  - benchmark_results.md (Markdown format)")


if __name__ == "__main__":
    main() 