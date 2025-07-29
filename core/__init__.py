"""
BenchmarkEngine - A framework for model benchmarking with extensible adapters and metrics.

This is the core orchestration layer that handles the generic flow:
load → run → collect → report
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Tuple, Union
import json
import time
import psutil
import gc
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """Supported model types for validation."""
    HUGGINGFACE = "huggingface"
    TENSORFLOW_LITE = "tflite"
    ONNX = "onnx"
    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"
    CUSTOM = "custom"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    num_samples: Optional[int] = None
    warmup_runs: int = 5
    batch_size: int = 1
    precision: str = "fp32"  # fp32, fp16, int8
    device: str = "cpu"  # cpu, gpu, tpu
    profile_memory: bool = True
    profile_gpu: bool = False


class BaseModelAdapter(ABC):
    """Abstract interface that all model adapters must implement."""
    
    @abstractmethod
    def load(self, model_path: str) -> bool:
        """Load the model from the given path."""
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure model parameters (batch_size, precision, device, etc.)."""
        pass
    
    @abstractmethod
    def preprocess_input(self, sample: Any) -> Any:
        """Convert dataset sample to model input format."""
        pass
    
    @abstractmethod
    def run(self, inputs: Any) -> Any:
        """Run inference on the given inputs."""
        pass
    
    @abstractmethod
    def postprocess_output(self, model_output: Any) -> Any:
        """Convert model output to standardized format."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model."""
        pass
    
    @abstractmethod
    def get_model_type(self) -> ModelType:
        """Return the type of model (for validation)."""
        pass


class BaseMetric(ABC):
    """Abstract interface that all metrics must implement."""
    
    @abstractmethod
    def calculate(self, predictions: List[Any], targets: List[Any], **kwargs) -> Dict[str, float]:
        """Calculate metric values given predictions and targets."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this metric."""
        pass
    
    @abstractmethod
    def validate_inputs(self, predictions: List[Any], targets: List[Any]) -> bool:
        """Validate that inputs are compatible with this metric."""
        pass


class BaseDataset(ABC):
    """Abstract interface that all datasets must implement."""
    
    @abstractmethod
    def load(self, dataset_path: str) -> bool:
        """Load the dataset from the given path."""
        pass
    
    @abstractmethod
    def get_samples(self, num_samples: Optional[int] = None) -> List[Any]:
        """Get samples from the dataset (for compatibility)."""
        pass
    
    @abstractmethod
    def get_samples_with_targets(self, num_samples: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """Get (input, target) pairs for evaluation."""
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded dataset."""
        pass
    
    @abstractmethod
    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the expected input shape for models."""
        pass


class BenchmarkEngine:
    """
    Core benchmarking orchestration layer.
    
    Handles the generic flow: load → run → collect → report
    """
    
    def __init__(self):
        self.model_adapter: Optional[BaseModelAdapter] = None
        self.dataset: Optional[BaseDataset] = None
        self.metrics: List[BaseMetric] = []
        self.results: Dict[str, Any] = {}
        self.config: BenchmarkConfig = BenchmarkConfig()
        
        # Plugin registry for dynamic registration
        self._model_adapters: Dict[str, Type[BaseModelAdapter]] = {}
        self._metrics: Dict[str, Type[BaseMetric]] = {}
        self._datasets: Dict[str, Type[BaseDataset]] = {}
    
    def register_adapter(self, name: str, adapter_class: Type[BaseModelAdapter]):
        """Register a model adapter for dynamic loading."""
        self._model_adapters[name] = adapter_class
    
    def register_metric(self, name: str, metric_class: Type[BaseMetric]):
        """Register a metric for dynamic loading."""
        self._metrics[name] = metric_class
    
    def register_dataset(self, name: str, dataset_class: Type[BaseDataset]):
        """Register a dataset for dynamic loading."""
        self._datasets[name] = dataset_class
    
    def configure_benchmark(self, config: Dict[str, Any]):
        """Configure benchmark parameters."""
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def load_model(self, adapter_name: str, model_path: str, model_config: Optional[Dict[str, Any]] = None) -> bool:
        """Load a model using the specified adapter."""
        if adapter_name not in self._model_adapters:
            raise ValueError(f"Unknown adapter: {adapter_name}")
        
        adapter_class = self._model_adapters[adapter_name]
        self.model_adapter = adapter_class()
        
        # Load the model
        if not self.model_adapter.load(model_path):
            raise RuntimeError(f"Failed to load model from {model_path}")
        
        # Configure the model if config provided
        if model_config:
            if not self.model_adapter.configure(model_config):
                raise RuntimeError(f"Failed to configure model with config: {model_config}")
        
        return True
    
    def load_dataset(self, dataset_name: str, dataset_path: str) -> bool:
        """Load a dataset using the specified dataset loader."""
        if dataset_name not in self._datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_class = self._datasets[dataset_name]
        self.dataset = dataset_class()
        return self.dataset.load(dataset_path)
    
    def add_metric(self, metric_name: str) -> bool:
        """Add a metric to the benchmark."""
        if metric_name not in self._metrics:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        metric_class = self._metrics[metric_name]
        self.metrics.append(metric_class())
        return True
    
    def validate_setup(self) -> bool:
        """Validate that model, dataset, and metrics are compatible."""
        if not self.model_adapter:
            raise RuntimeError("No model loaded")
        if not self.dataset:
            raise RuntimeError("No dataset loaded")
        if not self.metrics:
            raise RuntimeError("No metrics configured")
        
        # Validate model type compatibility with dataset
        model_type = self.model_adapter.get_model_type()
        dataset_info = self.dataset.get_dataset_info()
        
        # Basic validation - could be extended with more sophisticated checks
        print(f"✓ Model type: {model_type.value}")
        print(f"✓ Dataset type: {dataset_info.get('type', 'unknown')}")
        print(f"✓ Metrics: {[m.get_name() for m in self.metrics]}")
        
        return True
    
    def run_benchmark(self, num_samples: Optional[int] = None, warmup_runs: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete benchmark pipeline.
        
        Args:
            num_samples: Number of samples to benchmark (None = all)
            warmup_runs: Number of warmup runs before timing
        
        Returns:
            Dictionary containing benchmark results
        """
        # Use config values if not provided
        if num_samples is None:
            num_samples = self.config.num_samples
        if warmup_runs is None:
            warmup_runs = self.config.warmup_runs
        
        # Validate setup
        self.validate_setup()
        
        # Get samples with targets
        samples_with_targets = self.dataset.get_samples_with_targets(num_samples)
        if not samples_with_targets:
            raise RuntimeError("No samples available")
        
        # Separate inputs and targets
        inputs = [sample for sample, _ in samples_with_targets]
        targets = [target for _, target in samples_with_targets]
        
        # Warmup runs
        print(f"Running {warmup_runs} warmup runs...")
        for _ in range(warmup_runs):
            preprocessed_input = self.model_adapter.preprocess_input(inputs[0])
            model_output = self.model_adapter.run(preprocessed_input)
            _ = self.model_adapter.postprocess_output(model_output)
        
        # Memory profiling setup
        initial_memory = psutil.Process().memory_info().rss if self.config.profile_memory else 0
        
        # Actual benchmark
        print(f"Running benchmark on {len(inputs)} samples...")
        start_time = time.time()
        
        predictions = []
        inference_times = []
        
        for i, (sample, target) in enumerate(samples_with_targets):
            sample_start = time.time()
            
            # Preprocess input
            preprocessed_input = self.model_adapter.preprocess_input(sample)
            
            # Run inference
            model_output = self.model_adapter.run(preprocessed_input)
            
            # Postprocess output
            prediction = self.model_adapter.postprocess_output(model_output)
            
            sample_time = time.time() - sample_start
            
            predictions.append(prediction)
            inference_times.append(sample_time)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(inputs)} samples")
        
        total_time = time.time() - start_time
        
        # Memory profiling
        final_memory = psutil.Process().memory_info().rss if self.config.profile_memory else 0
        memory_used = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Calculate metrics
        metric_results = {}
        for metric in self.metrics:
            metric_name = metric.get_name()
            
            # Validate inputs for this metric
            if not metric.validate_inputs(predictions, targets):
                print(f"Warning: Invalid inputs for metric {metric_name}")
                continue
            
            metric_values = metric.calculate(predictions, targets)
            metric_results[metric_name] = metric_values
        
        # Compile results
        self.results = {
            "model_info": self.model_adapter.get_model_info(),
            "dataset_info": self.dataset.get_dataset_info(),
            "benchmark_config": {
                "num_samples": len(inputs),
                "warmup_runs": warmup_runs,
                "batch_size": self.config.batch_size,
                "precision": self.config.precision,
                "device": self.config.device
            },
            "timing": {
                "total_time": total_time,
                "average_inference_time": sum(inference_times) / len(inference_times),
                "min_inference_time": min(inference_times),
                "max_inference_time": max(inference_times),
                "throughput": len(inputs) / total_time,
                "inference_times": inference_times  # Detailed timing data
            },
            "profiling": {
                "memory_used_mb": memory_used,
                "peak_memory_mb": final_memory / 1024 / 1024
            },
            "metrics": metric_results
        }
        
        return self.results
    
    def compare_models(self, model_configs: List[Dict], dataset_name: str, metrics: List[str]) -> Dict[str, Any]:
        """
        Compare multiple models on the same dataset and metrics.
        
        Args:
            model_configs: List of model configurations
                [{"adapter": "huggingface", "path": "model1", "config": {...}},
                 {"adapter": "tflite", "path": "model2", "config": {...}}]
            dataset_name: Name of dataset to use
            metrics: List of metric names to calculate
        
        Returns:
            Dictionary with comparison results
        """
        comparison_results = {}
        
        for i, model_config in enumerate(model_configs):
            print(f"\nBenchmarking model {i+1}/{len(model_configs)}: {model_config['adapter']}")
            
            # Create new engine instance for each model
            engine = BenchmarkEngine()
            
            # Register plugins (assuming same plugins as current engine)
            for name, adapter_class in self._model_adapters.items():
                engine.register_adapter(name, adapter_class)
            for name, metric_class in self._metrics.items():
                engine.register_metric(name, metric_class)
            for name, dataset_class in self._datasets.items():
                engine.register_dataset(name, dataset_class)
            
            # Load dataset
            engine.load_dataset(dataset_name, model_config.get("dataset_path", ""))
            
            # Add metrics
            for metric_name in metrics:
                engine.add_metric(metric_name)
            
            # Load and benchmark model
            engine.load_model(
                model_config["adapter"], 
                model_config["path"], 
                model_config.get("config")
            )
            
            results = engine.run_benchmark()
            comparison_results[f"model_{i+1}"] = {
                "config": model_config,
                "results": results
            }
        
        return comparison_results
    
    def export_results(self, output_path: str, format: str = "json"):
        """Export benchmark results to a file."""
        if not self.results:
            raise RuntimeError("No results to export. Run benchmark first.")
        
        output_file = Path(output_path)
        
        if format.lower() == "json":
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
        elif format.lower() == "markdown":
            self._export_markdown(output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Results exported to {output_file}")
    
    def _export_markdown(self, output_file: Path):
        """Export results as markdown."""
        with open(output_file, 'w') as f:
            f.write("# Benchmark Results\n\n")
            
            # Model and Dataset Info
            f.write("## Model Information\n")
            for key, value in self.results["model_info"].items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
            
            f.write("## Dataset Information\n")
            for key, value in self.results["dataset_info"].items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
            
            # Benchmark Configuration
            f.write("## Benchmark Configuration\n")
            config = self.results["benchmark_config"]
            for key, value in config.items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
            
            # Timing Results
            f.write("## Timing Results\n")
            timing = self.results["timing"]
            f.write(f"- **Total Time**: {timing['total_time']:.2f}s\n")
            f.write(f"- **Average Inference Time**: {timing['average_inference_time']:.4f}s\n")
            f.write(f"- **Min Inference Time**: {timing['min_inference_time']:.4f}s\n")
            f.write(f"- **Max Inference Time**: {timing['max_inference_time']:.4f}s\n")
            f.write(f"- **Throughput**: {timing['throughput']:.2f} samples/s\n")
            f.write("\n")
            
            # Profiling Results
            if "profiling" in self.results:
                f.write("## Profiling Results\n")
                profiling = self.results["profiling"]
                for key, value in profiling.items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")
            
            # Metrics
            f.write("## Metrics\n")
            for metric_name, metric_values in self.results["metrics"].items():
                f.write(f"### {metric_name}\n")
                for key, value in metric_values.items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")
    
    def print_results(self):
        """Print benchmark results to console."""
        if not self.results:
            print("No results available. Run benchmark first.")
            return
        
        print("\n" + "="*50)
        print("BENCHMARK RESULTS")
        print("="*50)
        
        # Model and Dataset Info
        print(f"\nModel: {self.results['model_info'].get('name', 'Unknown')}")
        print(f"Dataset: {self.results['dataset_info'].get('name', 'Unknown')}")
        print(f"Samples: {self.results['benchmark_config']['num_samples']}")
        
        # Configuration
        config = self.results['benchmark_config']
        print(f"Batch Size: {config.get('batch_size', 1)}")
        print(f"Precision: {config.get('precision', 'fp32')}")
        print(f"Device: {config.get('device', 'cpu')}")
        
        # Timing
        timing = self.results["timing"]
        print(f"\nTiming:")
        print(f"  Total Time: {timing['total_time']:.2f}s")
        print(f"  Avg Inference: {timing['average_inference_time']:.4f}s")
        print(f"  Throughput: {timing['throughput']:.2f} samples/s")
        
        # Profiling
        if "profiling" in self.results:
            profiling = self.results["profiling"]
            print(f"\nProfiling:")
            print(f"  Memory Used: {profiling.get('memory_used_mb', 0):.2f} MB")
            print(f"  Peak Memory: {profiling.get('peak_memory_mb', 0):.2f} MB")
        
        # Metrics
        print(f"\nMetrics:")
        for metric_name, metric_values in self.results["metrics"].items():
            print(f"  {metric_name}:")
            for key, value in metric_values.items():
                print(f"    {key}: {value}")
        
        print("="*50)


def main():
    """CLI entry point for the BenchmarkEngine framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BenchmarkEngine Framework")
    parser.add_argument("--example", action="store_true", help="Run basic example")
    parser.add_argument("--custom", action="store_true", help="Run custom plugins example")
    
    args = parser.parse_args()
    
    if args.example:
        # Import and run basic example
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
        from examples.basic_benchmark import main as run_basic
        run_basic()
    elif args.custom:
        # Import and run custom example
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
        from examples.custom_plugins import main as run_custom
        run_custom()
    else:
        print("BenchmarkEngine Framework")
        print("=" * 30)
        print("Use --example to run the basic benchmark")
        print("Use --custom to run the custom plugins example")
        print("\nFor more information, see the README.md file")


if __name__ == "__main__":
    main() 