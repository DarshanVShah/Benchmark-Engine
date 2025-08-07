"""
Core types and enums for the BenchmarkEngine framework.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


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
