"""
HuggingFace model adapter for the BenchmarkEngine framework.

This adapter provides seamless integration with HuggingFace transformers library,
supporting text classification, token classification, and other NLP tasks.

Example usage:
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.load_model("huggingface", "bert-base-uncased")
"""

import time
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    pipeline,
    Pipeline,
)
from core import BaseModelAdapter, ModelType, DataType, OutputType


class HuggingFaceAdapter(BaseModelAdapter):
    """
    HuggingFace model adapter for text-based models.

    Supports various HuggingFace model types:
    - Text classification (sequence classification)
    - Token classification (NER, POS tagging)
    - Question answering
    - Text generation
    - Custom models

    This adapter handles:
    - Model loading from HuggingFace Hub or local path
    - Text preprocessing and tokenization
    - Inference with proper error handling
    - Output postprocessing for different task types
    - Configuration for different precision and devices
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_name = "HuggingFace"
        self.model_type = "unknown"
        self.config = {}
        self.device = "cpu"
        self.task_type = "text-classification"  # Default task
        self.is_multi_label = False  # Track if this is a multi-label task

    @property
    def input_type(self) -> DataType:
        """HuggingFace models expect text input."""
        return DataType.TEXT

    @property
    def output_type(self) -> OutputType:
        """HuggingFace models output probabilities for multi-label, class IDs for single-label."""
        return OutputType.PROBABILITIES if self.is_multi_label else OutputType.CLASS_ID

    def load(self, model_path: str) -> bool:
        """
        Load a HuggingFace model and tokenizer.

        Args:
            model_path: Path to model (can be HuggingFace Hub name or local path)

        Returns:
            True if loaded successfully
        """
        try:
            print(f"Loading HuggingFace model from {model_path}...")

            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Try to determine model type based on path
            if "bert" in model_path.lower():
                self.model_type = "bert"
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            elif "roberta" in model_path.lower():
                self.model_type = "roberta"
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            elif "distilbert" in model_path.lower():
                self.model_type = "distilbert"
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            else:
                # Generic approach - try sequence classification first
                try:
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    self.model_type = "sequence_classification"
                except:
                    # Try generic model
                    self.model = AutoModel.from_pretrained(model_path)
                    self.model_type = "generic"

            # Move to device
            self.model.to(self.device)
            self.model.eval()

            print(f"  Model loaded successfully")
            print(f"  Model type: {self.model_type}")
            print(f"  Device: {self.device}")

            return True

        except Exception as e:
            print(f"  Failed to load model: {e}")
            return False

    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure model parameters.

        Args:
            config: Configuration dictionary with keys:
                - device: "cpu", "cuda", "mps"
                - precision: "fp32", "fp16", "int8"
                - batch_size: Batch size for inference
                - max_length: Maximum sequence length
                - task_type: Type of task (classification, etc.)
                - is_multi_label: Whether this is a multi-label task

        Returns:
            True if configured successfully
        """
        try:
            self.config.update(config)

            # Set device
            if "device" in config:
                self.device = config["device"]
                if self.model:
                    self.model.to(self.device)

            # Set precision
            if "precision" in config:
                precision = config["precision"]
                if precision == "fp16" and self.device != "cpu":
                    self.model = self.model.half()

            # Set task type
            if "task_type" in config:
                self.task_type = config["task_type"]

            # Set multi-label flag
            if "is_multi_label" in config:
                self.is_multi_label = config["is_multi_label"]
            elif "task_type" in config and config["task_type"] == "multi-label":
                self.is_multi_label = True

            print(f"  Model configured successfully")
            return True

        except Exception as e:
            print(f"  Failed to configure model: {e}")
            return False

    def preprocess(self, raw_input: Any) -> Any:
        """
        Preprocess text input for HuggingFace models.

        Args:
            raw_input: Raw text input from dataset (can be string or dict with 'text' key)

        Returns:
            Tokenized inputs ready for model inference
        """
        try:
            # Extract text from input
            if isinstance(raw_input, dict):
                if "text" in raw_input:
                    text = raw_input["text"]
                elif "input" in raw_input:
                    text = raw_input["input"]
                else:
                    # Try to find any string value
                    text = next((v for v in raw_input.values() if isinstance(v, str)), "")
            elif isinstance(raw_input, str):
                text = raw_input
            else:
                print(f"Warning: Expected string or dict input, got {type(raw_input)}")
                text = str(raw_input)

            if not text:
                print("Warning: Empty text input")
                return None

            # Tokenize the text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.get("max_length", 512),
                padding=True,
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            return inputs

        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None

    def run(self, preprocessed_input: Any) -> Any:
        """
        Run inference on preprocessed input.

        Args:
            preprocessed_input: Tokenized inputs from preprocess()

        Returns:
            Raw model output (logits)
        """
        try:
            with torch.no_grad():
                outputs = self.model(**preprocessed_input)
                return outputs.logits

        except Exception as e:
            print(f"Error in model inference: {e}")
            return None

    def postprocess(self, model_output: Any) -> Any:
        """
        Postprocess model output to get predictions.

        Args:
            model_output: Raw logits from model

        Returns:
            Class ID predictions for single-label, probabilities for multi-label
        """
        try:
            if model_output is None:
                return None

            # Convert logits to probabilities
            if self.is_multi_label:
                # For multi-label, use sigmoid to get probabilities
                probabilities = torch.sigmoid(model_output)
                return probabilities.cpu().numpy().tolist()
            else:
                # For single-label, use softmax and get class IDs
                probabilities = torch.softmax(model_output, dim=-1)
                predicted_class_ids = torch.argmax(probabilities, dim=-1)
                return predicted_class_ids.cpu().numpy().tolist()

        except Exception as e:
            print(f"Error in postprocessing: {e}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model."""
        return {
            "name": self.model_name,
            "type": self.model_type,
            "device": self.device,
            "task_type": self.task_type,
            "is_multi_label": self.is_multi_label,
            "output_type": self.output_type.value,
            "config": self.config,
        }

    def get_model_type(self) -> ModelType:
        """Return the type of model."""
        return ModelType.HUGGINGFACE
