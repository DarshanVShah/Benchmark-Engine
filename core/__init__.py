"""
BenchmarkEngine - A framework for model benchmarking with extensible adapters and metrics.

This is the core orchestration layer that handles the generic flow:
load → run → collect → report
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
import json
import time
from pathlib import Path


class BaseModelAdapter(ABC):
    """Abstract interface that all model adapters must implement."""
    
    @abstractmethod
    def load(self, model_path: str) -> bool:
        """Load the model from the given path."""
        pass
    
    @abstractmethod
    def run(self, inputs: Any) -> Any:
        """Run inference on the given inputs."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model."""
        pass


class BaseMetric(ABC):
    """Abstract interface that all metrics must implement."""
    
    @abstractmethod
    def calculate(self, predictions: Any, targets: Any, **kwargs) -> Dict[str, float]:
        """Calculate metric values given predictions and targets."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this metric."""
        pass


class BaseDataset(ABC):
    """Abstract interface that all datasets must implement."""
    
    @abstractmethod
    def load(self, dataset_path: str) -> bool:
        """Load the dataset from the given path."""
        pass
    
    @abstractmethod
    def get_samples(self, num_samples: Optional[int] = None) -> List[Any]:
        """Get samples from the dataset."""
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded dataset."""
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
    
    def load_model(self, adapter_name: str, model_path: str) -> bool:
        """Load a model using the specified adapter."""
        if adapter_name not in self._model_adapters:
            raise ValueError(f"Unknown adapter: {adapter_name}")
        
        adapter_class = self._model_adapters[adapter_name]
        self.model_adapter = adapter_class()
        return self.model_adapter.load(model_path)
    
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
    
    def run_benchmark(self, num_samples: Optional[int] = None, warmup_runs: int = 5) -> Dict[str, Any]:
        """
        Run the complete benchmark pipeline.
        
        Args:
            num_samples: Number of samples to benchmark (None = all)
            warmup_runs: Number of warmup runs before timing
        
        Returns:
            Dictionary containing benchmark results
        """
        if not self.model_adapter:
            raise RuntimeError("No model loaded")
        if not self.dataset:
            raise RuntimeError("No dataset loaded")
        if not self.metrics:
            raise RuntimeError("No metrics configured")
        
        # Get samples
        samples = self.dataset.get_samples(num_samples)
        if not samples:
            raise RuntimeError("No samples available")
        
        # Warmup runs
        print(f"Running {warmup_runs} warmup runs...")
        for _ in range(warmup_runs):
            self.model_adapter.run(samples[0])
        
        # Actual benchmark
        print(f"Running benchmark on {len(samples)} samples...")
        start_time = time.time()
        
        predictions = []
        inference_times = []
        
        for i, sample in enumerate(samples):
            sample_start = time.time()
            prediction = self.model_adapter.run(sample)
            sample_time = time.time() - sample_start
            
            predictions.append(prediction)
            inference_times.append(sample_time)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(samples)} samples")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        metric_results = {}
        for metric in self.metrics:
            metric_name = metric.get_name()
            metric_values = metric.calculate(predictions, samples)
            metric_results[metric_name] = metric_values
        
        # Compile results
        self.results = {
            "model_info": self.model_adapter.get_model_info(),
            "dataset_info": self.dataset.get_dataset_info(),
            "benchmark_config": {
                "num_samples": len(samples),
                "warmup_runs": warmup_runs
            },
            "timing": {
                "total_time": total_time,
                "average_inference_time": sum(inference_times) / len(inference_times),
                "min_inference_time": min(inference_times),
                "max_inference_time": max(inference_times),
                "throughput": len(samples) / total_time
            },
            "metrics": metric_results
        }
        
        return self.results
    
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
            
            # Timing Results
            f.write("## Timing Results\n")
            timing = self.results["timing"]
            f.write(f"- **Total Time**: {timing['total_time']:.2f}s\n")
            f.write(f"- **Average Inference Time**: {timing['average_inference_time']:.4f}s\n")
            f.write(f"- **Min Inference Time**: {timing['min_inference_time']:.4f}s\n")
            f.write(f"- **Max Inference Time**: {timing['max_inference_time']:.4f}s\n")
            f.write(f"- **Throughput**: {timing['throughput']:.2f} samples/s\n")
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
        
        # Timing
        timing = self.results["timing"]
        print(f"\nTiming:")
        print(f"  Total Time: {timing['total_time']:.2f}s")
        print(f"  Avg Inference: {timing['average_inference_time']:.4f}s")
        print(f"  Throughput: {timing['throughput']:.2f} samples/s")
        
        # Metrics
        print(f"\nMetrics:")
        for metric_name, metric_values in self.results["metrics"].items():
            print(f"  {metric_name}:")
            for key, value in metric_values.items():
                print(f"    {key}: {value}")
        
        print("="*50) 