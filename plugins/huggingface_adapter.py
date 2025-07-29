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
    AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForQuestionAnswering,
    pipeline, Pipeline
)
from core import BaseModelAdapter, ModelType


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
                    self.model_type = "sequence-classification"
                except:
                    # Fallback to base model
                    self.model = AutoModel.from_pretrained(model_path)
                    self.model_type = "base"
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Get model info
            self.model_name = model_path.split("/")[-1]  # Extract model name from path
            
            print(f"HuggingFace model loaded: {self.model_name}")
            print(f"  Model type: {self.model_type}")
            print(f"  Parameters: {self.model.num_parameters():,}")
            
            return True
            
        except Exception as e:
            print(f"Failed to load HuggingFace model: {e}")
            return False
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure model parameters.
        
        Args:
            config: Configuration dictionary with parameters like:
                - device: "cpu", "cuda", "mps"
                - precision: "fp32", "fp16", "int8"
                - batch_size: int
                - task_type: "text-classification", "token-classification", etc.
                
        Returns:
            True if configured successfully
        """
        try:
            print(f"Configuring HuggingFace model with: {config}")
            
            # Update configuration
            self.config.update(config)
            
            # Handle device configuration
            if "device" in config:
                self.device = config["device"]
                
                # Move model to specified device
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                    print(f"  Model moved to CUDA")
                elif self.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.model = self.model.to("mps")
                    print(f"  Model moved to MPS")
                else:
                    self.model = self.model.to("cpu")
                    print(f"  Model moved to CPU")
            
            # Handle precision configuration
            if "precision" in config:
                if config["precision"] == "fp16" and self.device != "cpu":
                    self.model = self.model.half()
                    print(f"  Model converted to FP16")
                elif config["precision"] == "int8":
                    # Note: INT8 quantization requires additional setup
                    print(f"  INT8 quantization not implemented yet")
            
            # Handle task type configuration
            if "task_type" in config:
                self.task_type = config["task_type"]
                print(f"  Task type set to: {self.task_type}")
            
            print(f"✓ HuggingFace model configured")
            return True
            
        except Exception as e:
            print(f"Failed to configure HuggingFace model: {e}")
            return False
    
    def preprocess_input(self, sample: Any) -> Any:
        """
        Convert dataset sample to HuggingFace input format.
        
        Args:
            sample: Raw sample from dataset (dict with "text" key)
            
        Returns:
            Tokenized input ready for model inference
        """
        try:
            # Extract text from sample
            if isinstance(sample, dict):
                if "text" in sample:
                    text = sample["text"]
                elif "sentence" in sample:
                    text = sample["sentence"]
                elif "input" in sample:
                    text = sample["input"]
                else:
                    # Try to find any string value
                    text = next((v for v in sample.values() if isinstance(v, str)), "")
            elif isinstance(sample, str):
                text = sample
            else:
                raise ValueError(f"Unsupported sample format: {type(sample)}")
            
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512  # Default max length
            )
            
            # Move to same device as model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            return inputs
            
        except Exception as e:
            print(f"Failed to preprocess input: {e}")
            # Return a safe fallback
            return self.tokenizer(
                "fallback text",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
    
    def run(self, inputs: Any) -> Any:
        """
        Run inference with HuggingFace model.
        
        Args:
            inputs: Preprocessed inputs from preprocess_input()
            
        Returns:
            Raw model outputs
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load() first.")
            
            # Run inference
            with torch.no_grad():  # Disable gradient computation for inference
                outputs = self.model(**inputs)
            
            return outputs
            
        except Exception as e:
            print(f"Failed to run inference: {e}")
            # Return a safe fallback output
            return {"logits": torch.zeros(1, 2)}  # Binary classification fallback
    
    def postprocess_output(self, model_output: Any) -> Any:
        """
        Convert model output to standardized format.
        
        Args:
            model_output: Raw model outputs from run()
            
        Returns:
            Standardized prediction (class label, probabilities, etc.)
        """
        try:
            if hasattr(model_output, "logits"):
                logits = model_output.logits
                
                # Apply softmax to get probabilities
                probabilities = torch.softmax(logits, dim=-1)
                
                # Get predicted class
                predicted_class = torch.argmax(logits, dim=-1)
                
                # Convert to standard format
                if self.task_type == "text-classification":
                    return {
                        "class": predicted_class.item(),
                        "probabilities": probabilities.cpu().numpy().tolist(),
                        "confidence": probabilities.max().item()
                    }
                else:
                    # Generic format
                    return {
                        "prediction": predicted_class.item(),
                        "probabilities": probabilities.cpu().numpy().tolist()
                    }
            
            elif hasattr(model_output, "last_hidden_state"):
                # For base models without classification head
                hidden_states = model_output.last_hidden_state
                # Use mean pooling as a simple approach
                pooled = torch.mean(hidden_states, dim=1)
                return {
                    "embeddings": pooled.cpu().numpy().tolist()
                }
            
            else:
                # Fallback for unknown output format
                return {"raw_output": str(model_output)}
                
        except Exception as e:
            print(f"Failed to postprocess output: {e}")
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return metadata about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            "name": self.model_name,
            "type": "huggingface",
            "model_type": self.model_type,
            "task_type": self.task_type,
            "device": self.device,
            "loaded": self.model is not None,
            "config": self.config
        }
        
        if self.model is not None:
            info["parameters"] = self.model.num_parameters()
            info["model_class"] = self.model.__class__.__name__
        
        if self.tokenizer is not None:
            info["vocab_size"] = self.tokenizer.vocab_size
            info["tokenizer_class"] = self.tokenizer.__class__.__name__
        
        return info
    
    def get_model_type(self) -> ModelType:
        """
        Return the type of model for validation.
        
        Returns:
            ModelType enum value
        """
        return ModelType.HUGGINGFACE 