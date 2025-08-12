"""
Enhanced Reporting Module

Provides comprehensive result reporting, JSON export, and administrator-style analysis
for benchmark results.
"""

import json
import os
import datetime
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path


class BenchmarkJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for benchmark results."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'value'):  # Handle enums
            return obj.value
        elif hasattr(obj, '__dict__'):  # Handle custom objects
            return obj.__dict__
        return super().default(obj)


class BenchmarkReporter:
    """
    Comprehensive reporting system for benchmark results.
    
    Provides administrator-style analysis and JSON export functionality.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize the reporter with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results_history: List[Dict[str, Any]] = []
    
    def generate_comprehensive_report(self, 
                                   benchmark_results: Dict[str, Any],
                                   model_info: Dict[str, Any],
                                   dataset_info: Dict[str, Any],
                                   metrics_info: List[Dict[str, Any]],
                                   test_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive administrator report.
        
        Args:
            benchmark_results: Raw benchmark results
            model_info: Information about the tested model
            dataset_info: Information about the test dataset
            metrics_info: Information about evaluation metrics
            test_config: Configuration used for testing
            
        Returns:
            Comprehensive report dictionary
        """
        timestamp = datetime.datetime.now().isoformat()
        
        # Extract key metrics
        accuracy = self._extract_accuracy(benchmark_results)
        performance_metrics = self._extract_performance_metrics(benchmark_results)
        
        # Generate administrator assessment
        assessment = self._generate_assessment(accuracy, performance_metrics, dataset_info)
        
        # Create comprehensive report
        report = {
            "metadata": {
                "timestamp": timestamp,
                "test_id": f"benchmark_{timestamp.replace(':', '-').replace('.', '-')}",
                "framework_version": "1.0.0",
                "test_type": "comprehensive_benchmark"
            },
            "test_configuration": test_config,
            "model_information": model_info,
            "dataset_information": dataset_info,
            "metrics_information": metrics_info,
            "benchmark_results": benchmark_results,
            "extracted_metrics": {
                "accuracy": accuracy,
                "performance": performance_metrics
            },
            "administrator_assessment": assessment,
            "summary": {
                "overall_status": assessment["overall_status"],
                "key_findings": assessment["key_findings"],
                "recommendations": assessment["recommendations"]
            }
        }
        
        # Store in history
        self.results_history.append(report)
        
        return report
    
    def _extract_accuracy(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract accuracy metrics from benchmark results."""
        accuracy_info = {
            "overall_accuracy": None,
            "per_metric_accuracy": {},
            "accuracy_breakdown": {}
        }
        
        if "metrics" in results:
            for metric_name, metric_values in results["metrics"].items():
                if isinstance(metric_values, dict):
                    if "accuracy" in metric_values:
                        accuracy_info["per_metric_accuracy"][metric_name] = metric_values["accuracy"]
                        
                        # Track overall accuracy
                        if accuracy_info["overall_accuracy"] is None:
                            accuracy_info["overall_accuracy"] = metric_values["accuracy"]
                        else:
                            # For multiple metrics, use average
                            current = accuracy_info["overall_accuracy"]
                            accuracy_info["overall_accuracy"] = (current + metric_values["accuracy"]) / 2
        
        return accuracy_info
    
    def _extract_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance metrics from benchmark results."""
        performance = {
            "total_time": None,
            "average_inference_time": None,
            "throughput": None,
            "memory_usage": None,
            "gpu_utilization": None
        }
        
        if "timing" in results:
            timing = results["timing"]
            performance.update({
                "total_time": timing.get("total_time"),
                "average_inference_time": timing.get("avg_inference_time"),
                "throughput": timing.get("throughput")
            })
        
        return performance
    
    def _generate_assessment(self, 
                           accuracy: Dict[str, Any], 
                           performance: Dict[str, Any],
                           dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate administrator assessment of results."""
        assessment = {
            "overall_status": "PENDING",
            "accuracy_grade": "PENDING",
            "performance_grade": "PENDING",
            "key_findings": [],
            "recommendations": [],
            "compliance_check": {}
        }
        
        # Assess accuracy
        if accuracy["overall_accuracy"] is not None:
            acc = accuracy["overall_accuracy"]
            if acc >= 0.9:
                assessment["accuracy_grade"] = "A+ (Excellent)"
            elif acc >= 0.8:
                assessment["accuracy_grade"] = "A (Very Good)"
            elif acc >= 0.7:
                assessment["accuracy_grade"] = "B (Good)"
            elif acc >= 0.6:
                assessment["accuracy_grade"] = "C (Acceptable)"
            else:
                assessment["accuracy_grade"] = "D (Needs Improvement)"
            
            assessment["key_findings"].append(f"Model achieved {acc:.2%} accuracy")
        
        # Assess performance
        if performance["throughput"] is not None:
            throughput = performance["throughput"]
            if throughput > 1000:
                assessment["performance_grade"] = "A+ (Excellent)"
            elif throughput > 500:
                assessment["performance_grade"] = "A (Very Good)"
            elif throughput > 100:
                assessment["performance_grade"] = "B (Good)"
            elif throughput > 50:
                assessment["performance_grade"] = "C (Acceptable)"
            else:
                assessment["performance_grade"] = "D (Needs Improvement)"
            
            assessment["key_findings"].append(f"Model processes {throughput:.1f} samples/second")
        
        # Overall status
        if assessment["accuracy_grade"].startswith("A") and assessment["performance_grade"].startswith("A"):
            assessment["overall_status"] = "EXCELLENT"
        elif assessment["accuracy_grade"].startswith("B") and assessment["performance_grade"].startswith("B"):
            assessment["overall_status"] = "GOOD"
        elif assessment["accuracy_grade"].startswith("C") and assessment["performance_grade"].startswith("C"):
            assessment["overall_status"] = "ACCEPTABLE"
        else:
            assessment["overall_status"] = "NEEDS_IMPROVEMENT"
        
        # Generate recommendations
        if assessment["overall_status"] == "NEEDS_IMPROVEMENT":
            assessment["recommendations"].append("Consider model optimization or architecture changes")
            assessment["recommendations"].append("Review training data quality and preprocessing")
        
        if assessment["performance_grade"].startswith("D"):
            assessment["recommendations"].append("Investigate performance bottlenecks")
            assessment["recommendations"].append("Consider model quantization or optimization")
        
        return assessment
    
    def export_to_json(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Export report to JSON file.
        
        Args:
            report: Report dictionary to export
            filename: Optional custom filename
            
        Returns:
            Path to exported JSON file
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_report_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, cls=BenchmarkJSONEncoder)
        
        return str(filepath)
    
    def export_summary_report(self, filename: Optional[str] = None) -> str:
        """
        Export a summary of all benchmark results.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to exported summary file
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_summary_{timestamp}.json"
        
        summary = {
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_benchmarks": len(self.results_history),
                "framework_version": "1.0.0"
            },
            "benchmark_summary": []
        }
        
        for result in self.results_history:
            summary["benchmark_summary"].append({
                "test_id": result["metadata"]["test_id"],
                "timestamp": result["metadata"]["timestamp"],
                "overall_status": result["summary"]["overall_status"],
                "accuracy": result["extracted_metrics"]["accuracy"]["overall_accuracy"],
                "performance_grade": result["administrator_assessment"]["performance_grade"]
            })
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, cls=BenchmarkJSONEncoder)
        
        return str(filepath)
    
    def print_administrator_report(self, report: Dict[str, Any]):
        """Print a formatted administrator report to console."""
        print("\n" + "="*80)
        print("ADMINISTRATOR BENCHMARK REPORT")
        print("="*80)
        
        # Basic information
        print(f"Test ID: {report['metadata']['test_id']}")
        print(f"Timestamp: {report['metadata']['timestamp']}")
        print(f"Model: {report['model_information'].get('name', 'Unknown')}")
        print(f"Dataset: {report['dataset_information'].get('name', 'Unknown')}")
        
        # Results summary
        print(f"\nOVERALL ASSESSMENT: {report['summary']['overall_status']}")
        print(f"Accuracy Grade: {report['administrator_assessment']['accuracy_grade']}")
        print(f"Performance Grade: {report['administrator_assessment']['performance_grade']}")
        
        # Key metrics
        if report['extracted_metrics']['accuracy']['overall_accuracy']:
            acc = report['extracted_metrics']['accuracy']['overall_accuracy']
            print(f"\nAccuracy: {acc:.4f} ({acc:.2%})")
        
        if report['extracted_metrics']['performance']['throughput']:
            throughput = report['extracted_metrics']['performance']['throughput']
            print(f"Throughput: {throughput:.1f} samples/second")
        
        # Key findings
        if report['administrator_assessment']['key_findings']:
            print(f"\nKEY FINDINGS:")
            for finding in report['administrator_assessment']['key_findings']:
                print(f"  • {finding}")
        
        # Recommendations
        if report['administrator_assessment']['recommendations']:
            print(f"\nRECOMMENDATIONS:")
            for rec in report['administrator_assessment']['recommendations']:
                print(f"  • {rec}")
        
        print("="*80)
    
    def get_results_history(self) -> List[Dict[str, Any]]:
        """Get all stored benchmark results."""
        return self.results_history.copy()
    
    def clear_history(self):
        """Clear the results history."""
        self.results_history.clear()
