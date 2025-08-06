"""
Reporting and export functionality for benchmark results.

This module handles the export and display of benchmark results in various formats.
"""

import json
from pathlib import Path
from typing import Dict, Any


class ResultReporter:
    """Handles reporting and export of benchmark results."""

    def __init__(self, results: Dict[str, Any]):
        self.results = results

    def export_results(self, output_path: str, format: str = "json"):
        """Export benchmark results to a file."""
        if not self.results:
            raise RuntimeError("No results to export. Run benchmark first.")

        output_file = Path(output_path)

        if format.lower() == "json":
            with open(output_file, "w") as f:
                json.dump(self.results, f, indent=2)
        elif format.lower() == "markdown":
            self._export_markdown(output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Results exported to {output_file}")

    def _export_markdown(self, output_file: Path):
        """Export results as markdown."""
        with open(output_file, "w") as f:
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
            f.write(
                f"- **Average Inference Time**: {timing['average_inference_time']:.4f}s\n"
            )
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

        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)

        # Model and Dataset Info
        print(f"\nModel: {self.results['model_info'].get('name', 'Unknown')}")
        print(f"Dataset: {self.results['dataset_info'].get('name', 'Unknown')}")
        print(f"Samples: {self.results['benchmark_config']['num_samples']}")

        # Configuration
        config = self.results["benchmark_config"]
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

        print("=" * 50)
