"""
Core benchmarking orchestration layer.

This module contains the main BenchmarkEngine class that handles the generic flow:
load → run → collect → report
"""

import json
import time
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Type

from .types import BenchmarkConfig
from .interfaces import BaseModelAdapter, BaseMetric, BaseDataset


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
    
    def load_dataset(self, dataset_name: str, dataset_path: str, dataset_config: Optional[Dict[str, Any]] = None) -> bool:
        """Load a dataset using the specified dataset loader with optional configuration."""
        if dataset_name not in self._datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_class = self._datasets[dataset_name]
        self.dataset = dataset_class()
        
        if dataset_config:
            return self.dataset.load(dataset_path, dataset_config)
        else:
            return self.dataset.load(dataset_path)
    
    def add_metric(self, metric_name: str, metric_instance: Optional[BaseMetric] = None) -> bool:
        """Add a metric to the benchmark."""
        if metric_instance:
            # Use provided metric instance
            self.metrics.append(metric_instance)
            return True
        else:
            # Create metric instance from registered class
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
        
        # Validate model-dataset compatibility
        if not self.model_adapter.validate_compatibility(self.dataset):
            raise ValueError(
                f"Model input type ({self.model_adapter.input_type.value}) "
                f"does not match dataset output type ({self.dataset.output_type.value})"
            )
        
        # Validate model-metrics compatibility
        for metric in self.metrics:
            if self.model_adapter.output_type != metric.expected_input_type:
                raise ValueError(
                    f"Model output type ({self.model_adapter.output_type.value}) "
                    f"does not match metric expected type ({metric.expected_input_type.value})"
                )
        
        return True
        
        # Validate model type compatibility with dataset
        model_type = self.model_adapter.get_model_type()
        dataset_info = self.dataset.get_dataset_info()
        
        # Basic validation - could be extended with more sophisticated checks
        print(f"Model type: {model_type.value}")
        print(f"Dataset type: {dataset_info.get('type', 'unknown')}")
        print(f"Metrics: {[m.get_name() for m in self.metrics]}")
        
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