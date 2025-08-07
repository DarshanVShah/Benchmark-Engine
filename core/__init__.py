"""
BenchmarkEngine - A framework for model benchmarking with extensible adapters and metrics.

This is the core orchestration layer that handles the generic flow:
load → run → collect → report
"""

# Import core types and interfaces
from .types import ModelType, BenchmarkConfig
from .interfaces import BaseModelAdapter, BaseMetric, BaseDataset, DataType, OutputType
from .engine import BenchmarkEngine
from .reporting import ResultReporter

# Import CLI functionality
from .cli import main


# Add reporting methods to BenchmarkEngine for backward compatibility
def _add_reporting_methods():
    """Add reporting methods to BenchmarkEngine for backward compatibility."""

    def export_results(self, output_path: str, format: str = "json"):
        """Export benchmark results to a file."""
        if not self.results:
            raise RuntimeError("No results to export. Run benchmark first.")

        reporter = ResultReporter(self.results)
        reporter.export_results(output_path, format)

    def print_results(self):
        """Print benchmark results to console."""
        if not self.results:
            print("No results available. Run benchmark first.")
            return

        reporter = ResultReporter(self.results)
        reporter.print_results()

    # Add methods to BenchmarkEngine class
    BenchmarkEngine.export_results = export_results
    BenchmarkEngine.print_results = print_results


# Initialize backward compatibility
_add_reporting_methods()

# Export main classes and functions
__all__ = [
    "BenchmarkEngine",
    "BaseModelAdapter",
    "BaseMetric",
    "BaseDataset",
    "ModelType",
    "BenchmarkConfig",
    "DataType",
    "OutputType",
    "ResultReporter",
    "main",
]
