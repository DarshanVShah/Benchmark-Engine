"""
Enhanced Benchmark Engine with multi-dataset support.
"""

import time
import psutil
from typing import Dict, Any, List, Optional, Type
from .interfaces import BaseModelAdapter, BaseMetric, BaseDataset
from .dataset_registry import DatasetRegistry, TaskType, DatasetConfig
from .types import BenchmarkConfig


class BenchmarkEngine:
    """Enhanced benchmark engine with multi-dataset support."""
    
    def __init__(self):
        self.model_adapter: Optional[BaseModelAdapter] = None
        self.dataset: Optional[BaseDataset] = None
        self.metrics: List[BaseMetric] = []
        self.results: Optional[Dict[str, Any]] = None
        self.config = BenchmarkConfig()
        self.dataset_registry = DatasetRegistry()
        
        # Registry for components
        self._adapters: Dict[str, Type[BaseModelAdapter]] = {}
        self._metrics: Dict[str, Type[BaseMetric]] = {}
        self._datasets: Dict[str, Type[BaseDataset]] = {}
    
    def register_adapter(self, name: str, adapter_class: Type[BaseModelAdapter]):
        """Register a model adapter."""
        self._adapters[name] = adapter_class
    
    def register_metric(self, name: str, metric_class: Type[BaseMetric]):
        """Register a metric."""
        self._metrics[name] = metric_class
    
    def register_dataset(self, name: str, dataset_class: Type[BaseDataset]):
        """Register a dataset adapter."""
        self._datasets[name] = dataset_class
    
    def configure_benchmark(self, config: Dict[str, Any]):
        """Configure benchmark parameters."""
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def load_model(self, adapter_name: str, model_path: str, model_config: Optional[Dict[str, Any]] = None) -> bool:
        """Load a model using the specified adapter."""
        if adapter_name not in self._adapters:
            print(f"Unknown adapter: {adapter_name}")
            return False
        
        adapter_class = self._adapters[adapter_name]
        self.model_adapter = adapter_class()
        
        if not self.model_adapter.load(model_path):
            print(f"Failed to load model: {model_path}")
            return False
        
        if model_config:
            self.model_adapter.configure(model_config)
        
        return True
    
    def load_dataset(self, dataset_name: str, dataset_path: str, dataset_config: Optional[Dict[str, Any]] = None) -> bool:
        """Load a dataset using the specified adapter."""
        if dataset_name not in self._datasets:
            print(f"Unknown dataset adapter: {dataset_name}")
            return False
        
        dataset_class = self._datasets[dataset_name]
        self.dataset = dataset_class()
        
        if dataset_config:
            success = self.dataset.load(dataset_path, dataset_config)
        else:
            success = self.dataset.load(dataset_path)
        
        if not success:
            print(f"Failed to load dataset: {dataset_path}")
            return False
        
        return True
    
    def add_metric(self, metric_name: str, metric_instance: Optional[BaseMetric] = None) -> bool:
        """Add a metric to the benchmark."""
        if metric_instance:
            self.metrics.append(metric_instance)
            return True
        
        if metric_name not in self._metrics:
            print(f"Unknown metric: {metric_name}")
            return False
        
        metric_class = self._metrics[metric_name]
        metric_instance = metric_class()
        self.metrics.append(metric_instance)
        return True
    
    def validate_setup(self) -> bool:
        """Validate that all components are properly set up."""
        if not self.model_adapter:
            print("No model adapter loaded")
            return False
        
        if not self.dataset:
            print("No dataset loaded")
            return False
        
        if not self.metrics:
            print("No metrics added")
            return False
        
        # Validate compatibility between model and dataset
        if not self.model_adapter.validate_compatibility(self.dataset):
            print(f"Model adapter input type ({self.model_adapter.input_type.value}) is not compatible with dataset output type ({self.dataset.output_type.value})")
            return False
        
        # Validate compatibility between model output and metrics
        for metric in self.metrics:
            if self.model_adapter.output_type != metric.expected_input_type:
                print(f"Model output type {self.model_adapter.output_type.value} is not compatible with metric {metric.get_name()} expected input type {metric.expected_input_type.value}")
                return False
        
        return True
    
    def run_benchmark(self, num_samples: Optional[int] = None, warmup_runs: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete benchmark pipeline."""
        # Use config values if not provided
        if num_samples is None:
            num_samples = self.config.num_samples
        if warmup_runs is None:
            warmup_runs = self.config.warmup_runs
        
        # Validate setup
        if not self.validate_setup():
            raise RuntimeError("Benchmark setup validation failed")
        
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
            _ = self.model_adapter.predict(inputs[0])
        
        # Memory profiling setup
        initial_memory = psutil.Process().memory_info().rss if self.config.profile_memory else 0
        
        # Actual benchmark
        print(f"Running benchmark on {len(inputs)} samples...")
        start_time = time.time()
        
        predictions = []
        inference_times = []
        
        for i, (sample, target) in enumerate(samples_with_targets):
            sample_start = time.time()
            
            # Use the Template Method pattern predict() method
            prediction = self.model_adapter.predict(sample)
            
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
                "throughput": len(inputs) / total_time
            },
            "memory": {
                "memory_used_mb": memory_used
            },
            "metrics": metric_results
        }
        
        return self.results
    
    def run_multi_dataset_benchmark(self, task_type: TaskType, model_path: str, 
                                  adapter_name: str, model_config: Optional[Dict[str, Any]] = None,
                                  dataset_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run benchmark across multiple datasets for a specific task type.
        
        Args:
            task_type: The type of ML task
            model_path: Path to the model file
            adapter_name: Name of the adapter to use
            model_config: Configuration for the model
            dataset_names: Specific dataset names to test (None = all datasets for task)
        
        Returns:
            Dictionary containing results for all datasets
        """
        print(f"Running multi-dataset benchmark for task: {task_type.value}")
        
        # Get datasets for this task
        available_datasets = self.dataset_registry.get_datasets_for_task(task_type)
        if not available_datasets:
            print(f"No datasets available for task: {task_type.value}")
            return {}
        
        # Filter by dataset names if specified
        if dataset_names:
            available_datasets = [d for d in available_datasets if d.name in dataset_names]
        
        print(f"Testing on {len(available_datasets)} datasets:")
        for dataset in available_datasets:
            print(f"  - {dataset.name}: {dataset.description}")
        
        # Load model once
        if not self.load_model(adapter_name, model_path, model_config):
            return {}
        
        # Run benchmarks on each dataset
        all_results = {
            "task_type": task_type.value,
            "model_path": model_path,
            "adapter": adapter_name,
            "datasets": {}
        }
        
        for dataset_config in available_datasets:
            print(f"\n{'='*50}")
            print(f"Testing on dataset: {dataset_config.name}")
            print(f"Description: {dataset_config.description}")
            print(f"Expected accuracy range: {dataset_config.expected_accuracy_range}")
            print(f"{'='*50}")
            
            # Ensure dataset is available
            if not self.dataset_registry.ensure_dataset_available(dataset_config):
                print(f"Failed to ensure dataset availability: {dataset_config.name}")
                all_results["datasets"][dataset_config.name] = {
                    "dataset_info": dataset_config,
                    "error": "Dataset not available"
                }
                continue
            
            # Load dataset
            if not self.load_dataset("template", dataset_config.path, dataset_config.config):
                print(f"Failed to load dataset: {dataset_config.name}")
                all_results["datasets"][dataset_config.name] = {
                    "dataset_info": dataset_config,
                    "error": "Failed to load dataset"
                }
                continue
            
            # Configure model for this dataset's task type
            if model_config is None:
                model_config = {}
            
            # Update model config based on dataset task type
            dataset_task_type = dataset_config.config.get("task_type", "single-label")
            if dataset_task_type == "multi-label":
                model_config["is_multi_label"] = True
                model_config["task_type"] = "multi-label"
            else:
                model_config["is_multi_label"] = False
                model_config["task_type"] = "single-label"
            
            # Reconfigure model if needed
            if self.model_adapter:
                self.model_adapter.configure(model_config)
            
            # Add appropriate metrics based on task type
            self.metrics.clear()
            if dataset_task_type == "multi-label":
                # Use multi-label metric that expects probabilities
                from metrics.template_metric import TemplateMultiLabelMetric
                multilabel_metric = TemplateMultiLabelMetric(metric_type="accuracy", threshold=0.5)
                self.metrics.append(multilabel_metric)
            else:
                # Use single-label metric that expects class IDs
                from metrics.simple_accuracy_metric import SimpleAccuracyMetric
                accuracy_metric = SimpleAccuracyMetric()
                self.metrics.append(accuracy_metric)
            
            # Run benchmark
            try:
                results = self.run_benchmark()
                all_results["datasets"][dataset_config.name] = {
                    "dataset_info": dataset_config,
                    "benchmark_results": results
                }
                print(f"✓ Completed benchmark for {dataset_config.name}")
            except Exception as e:
                print(f"✗ Failed benchmark for {dataset_config.name}: {e}")
                all_results["datasets"][dataset_config.name] = {
                    "dataset_info": dataset_config,
                    "error": str(e)
                }
        
        # Validate results against expected ranges
        validation_results = {}
        for dataset_name, dataset_results in all_results["datasets"].items():
            if "benchmark_results" in dataset_results:
                metrics = dataset_results["benchmark_results"]["metrics"]
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, dict) and "accuracy" in metric_value:
                        validation_results[dataset_name] = metric_value["accuracy"]
        
        validation_report = self.dataset_registry.validate_results(task_type, validation_results)
        all_results["validation"] = validation_report
        
        return all_results
    
    def print_results(self):
        """Print benchmark results."""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "="*50)
        print("BENCHMARK RESULTS")
        print("="*50)
        
        # Model info
        print(f"Model: {self.results['model_info']['name']}")
        print(f"Type: {self.results['model_info']['type']}")
        
        # Dataset info
        print(f"Dataset: {self.results['dataset_info']['name']}")
        print(f"Samples: {self.results['dataset_info']['num_samples']}")
        
        # Timing
        timing = self.results['timing']
        print(f"\nTiming:")
        print(f"  Total time: {timing['total_time']:.2f}s")
        print(f"  Average inference: {timing['average_inference_time']*1000:.2f}ms")
        print(f"  Throughput: {timing['throughput']:.2f} samples/s")
        
        # Memory
        if 'memory' in self.results:
            print(f"  Memory used: {self.results['memory']['memory_used_mb']:.2f} MB")
        
        # Metrics
        print(f"\nMetrics:")
        for metric_name, metric_values in self.results['metrics'].items():
            if isinstance(metric_values, dict):
                for key, value in metric_values.items():
                    print(f"  {metric_name} - {key}: {value:.4f}")
            else:
                print(f"  {metric_name}: {metric_values:.4f}")
        
        print("="*50)
    
    def print_multi_dataset_results(self, results: Dict[str, Any]):
        """Print multi-dataset benchmark results."""
        print("\n" + "="*60)
        print("MULTI-DATASET BENCHMARK RESULTS")
        print("="*60)
        
        print(f"Task Type: {results['task_type']}")
        print(f"Model: {results['model_path']}")
        print(f"Adapter: {results['adapter']}")
        
        print(f"\nDataset Results:")
        for dataset_name, dataset_results in results['datasets'].items():
            print(f"\n{'-'*40}")
            print(f"Dataset: {dataset_name}")
            
            if 'error' in dataset_results:
                print(f"Status: FAILED - {dataset_results['error']}")
                continue
            
            dataset_info = dataset_results['dataset_info']
            print(f"Description: {dataset_info.description}")
            print(f"Expected Range: {dataset_info.expected_accuracy_range}")
            
            benchmark_results = dataset_results['benchmark_results']
            metrics = benchmark_results['metrics']
            
            print(f"Results:")
            for metric_name, metric_values in metrics.items():
                if isinstance(metric_values, dict):
                    for key, value in metric_values.items():
                        print(f"  {metric_name} - {key}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {metric_values:.4f}")
        
        # Validation report
        if 'validation' in results:
            validation = results['validation']
            print(f"\n{'='*40}")
            print(f"VALIDATION REPORT")
            print(f"{'='*40}")
            print(f"Overall Assessment: {validation['overall_assessment']}")
            
            for dataset_result in validation['datasets_tested']:
                status = "✓ PASS" if dataset_result['assessment'] == "PASS" else "✗ FAIL"
                print(f"{dataset_result['dataset']}: {status}")
                print(f"  Accuracy: {dataset_result['accuracy']:.4f}")
                print(f"  Expected: {dataset_result['expected_range'][0]:.4f} - {dataset_result['expected_range'][1]:.4f}")
        
        print("="*60)
