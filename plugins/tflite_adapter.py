"""
TensorFlow Lite Model Adapter

This adapter handles TensorFlow Lite models for benchmarking.
Supports various input/output types and provides standardized interface.
"""

import os
import time
import numpy as np
from typing import Dict, Any, List, Optional, Union
import tensorflow as tf

from core import BaseModelAdapter, DataType, OutputType, ModelType


class TensorFlowLiteAdapter(BaseModelAdapter):
    """
    TensorFlow Lite model adapter for benchmarking.
    
    """
    
    def __init__(self):
        self.interpreter = None
        self.model_loaded = False
        self.model_path = ""
        self.input_details = []
        self.output_details = []
        self.input_shape = None
        self.output_shape = None
        self.input_type = DataType.TEXT  # Default, can be overridden
        self.output_type = OutputType.CLASS_ID  # Default, can be overridden
        self.task_type = "auto-detected"
        self.max_length = 512
        self.tokenizer = None
        
    @property
    def input_type(self) -> DataType:
        """Return the expected input type for this model."""
        return self._input_type
    
    @input_type.setter
    def input_type(self, value: DataType):
        self._input_type = value
    
    @property
    def output_type(self) -> OutputType:
        """Return the output type this model produces."""
        return self._output_type
    
    @output_type.setter
    def output_type(self, value: OutputType):
        self._output_type = value
    
    def load(self, model_path: str) -> bool:
        """Load TensorFlow Lite model from file."""
        try:
            print(f"Loading TensorFlow Lite model from {model_path}")
            
            if not os.path.exists(model_path):
                print(f"  Error: Model file not found: {model_path}")
                return False
            
            # Load the TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Extract shapes
            self.input_shape = self.input_details[0]['shape']
            self.output_shape = self.output_details[0]['shape']
            
            self.model_path = model_path
            self.model_loaded = True
            
            print(f"  TensorFlow Lite model loaded: {model_path}")
            print(f"  Input shape: {self.input_shape}")
            print(f"  Output shape: {self.output_shape}")
            print(f"  Input details: {self.input_details}")
            print(f"  Output details: {self.output_details}")
            
            return True
            
        except Exception as e:
            print(f"  Error loading TensorFlow Lite model: {e}")
            return False
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure model parameters."""
        try:
            print(f"Configuring TensorFlow Lite model with: {config}")
            
            # Set task type
            if "task_type" in config:
                self.task_type = config["task_type"]
                print(f"  Task type set to: {self.task_type}")
            
            # Set input/output types based on task
            if self.task_type == "text-classification":
                self.input_type = DataType.TEXT
                self.output_type = OutputType.CLASS_ID
            elif self.task_type == "emotion-detection":
                self.input_type = DataType.TEXT
                self.output_type = OutputType.PROBABILITIES
            elif self.task_type == "image-classification":
                self.input_type = DataType.IMAGE
                self.output_type = OutputType.CLASS_ID
            elif self.task_type == "regression":
                self.input_type = DataType.TEXT  # Could be TEXT or IMAGE
                self.output_type = OutputType.REGRESSION
            
            # Set max length for text models
            if "max_length" in config:
                self.max_length = config["max_length"]
            
            # Load tokenizer if specified
            if "tokenizer_path" in config:
                try:
                    from transformers import AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
                    print(f"  Tokenizer loaded: {config['tokenizer_path']}")
                except ImportError:
                    print("  Warning: transformers not available, using basic tokenization")
                    self.tokenizer = None
            
            print("  TensorFlow Lite model configured")
            return True
            
        except Exception as e:
            print(f"  Error configuring TensorFlow Lite model: {e}")
            return False
    
    def preprocess_input(self, raw_input: Any) -> Any:
        """
        Convert raw input to TensorFlow Lite format.
        
        Args:
            raw_input: Raw input from dataset (text, image, etc.)
            
        Returns:
            Preprocessed input ready for TFLite inference
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        if self.input_type == DataType.TEXT:
            return self._preprocess_text(raw_input)
        elif self.input_type == DataType.IMAGE:
            return self._preprocess_image(raw_input)
        elif self.input_type == DataType.TENSOR:
            return self._preprocess_tensor(raw_input)
        else:
            raise ValueError(f"Unsupported input type: {self.input_type}")
    
    def _preprocess_text(self, raw_input: Union[str, Dict[str, Any]]) -> np.ndarray:
        """Preprocess text input for TFLite model."""
        # Extract text from input
        if isinstance(raw_input, dict):
            text = raw_input.get("text", "")
        else:
            text = str(raw_input)
        
        # Truncate if needed
        if len(text) > self.max_length:
            text = text[:self.max_length]
        
        # Use tokenizer if available
        if self.tokenizer:
            # Tokenize and convert to input format
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )
            return tokens["input_ids"].astype(np.int32)
        else:
            # Basic tokenization (character-level)
            # Convert text to integer sequence
            char_to_int = {char: i for i, char in enumerate(set(text))}
            if not char_to_int:
                char_to_int = {' ': 0}
            
            # Pad or truncate to match input shape
            max_len = self.input_shape[1] if len(self.input_shape) > 1 else self.max_length
            tokenized = [char_to_int.get(c, 0) for c in text[:max_len]]
            
            # Pad with zeros
            while len(tokenized) < max_len:
                tokenized.append(0)
            
            # Ensure indices are within bounds for the model
            vocab_size = max(char_to_int.values()) + 1 if char_to_int else 1
            tokenized = [min(t, vocab_size - 1) for t in tokenized]
            
            # For BERT models, we need to provide all 3 input tensors
            if len(self.input_details) == 3:
                # This is a BERT model expecting attention_mask, input_ids, token_type_ids
                input_ids = np.array([tokenized], dtype=np.int32)
                attention_mask = np.ones_like(input_ids, dtype=np.int32)
                token_type_ids = np.zeros_like(input_ids, dtype=np.int32)
                
                # Return a list of the 3 tensors
                return [attention_mask, input_ids, token_type_ids]
            else:
                # Single input tensor
                return np.array([tokenized], dtype=np.int32)
    
    def _preprocess_image(self, raw_input: Union[str, Dict[str, Any]]) -> np.ndarray:
        """Preprocess image input for TFLite model."""
        # Extract image path or data
        if isinstance(raw_input, dict):
            image_path = raw_input.get("image", "")
        else:
            image_path = str(raw_input)
        
        # Load and preprocess image
        try:
            # Load image using PIL or tf.keras
            if os.path.exists(image_path):
                image = tf.keras.preprocessing.image.load_img(
                    image_path, 
                    target_size=(self.input_shape[1], self.input_shape[2])
                )
                image_array = tf.keras.preprocessing.image.img_to_array(image)
            else:
                # Create dummy image if path doesn't exist
                image_array = np.random.rand(*self.input_shape[1:])
            
            # Normalize to [0, 1] or [-1, 1] based on model requirements
            if image_array.max() > 1:
                image_array = image_array / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array.astype(np.float32)
            
        except Exception as e:
            print(f"  Warning: Error preprocessing image, using dummy data: {e}")
            # Return dummy data matching input shape
            return np.random.rand(*self.input_shape).astype(np.float32)
    
    def _preprocess_tensor(self, raw_input: Any) -> np.ndarray:
        """Preprocess tensor input for TFLite model."""
        if isinstance(raw_input, np.ndarray):
            return raw_input.astype(np.float32)
        elif isinstance(raw_input, list):
            return np.array(raw_input, dtype=np.float32)
        else:
            # Convert to numpy array
            return np.array(raw_input, dtype=np.float32)
    
    def run(self, preprocessed_input: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed input.
        
        Args:
            preprocessed_input: Input in TFLite format (single tensor or list of tensors)
            
        Returns:
            Raw model output
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Handle multiple input tensors (for BERT models)
        if isinstance(preprocessed_input, list):
            # Set all input tensors
            for i, tensor in enumerate(preprocessed_input):
                if i < len(self.input_details):
                    self.interpreter.set_tensor(self.input_details[i]['index'], tensor)
        else:
            # Single input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_input)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output
    
    def postprocess_output(self, model_output: np.ndarray) -> Any:
        """
        Convert raw model output to standardized format.
        
        Args:
            model_output: Raw output from TFLite model
            
        Returns:
            Standardized output matching output_type
        """
        if self.output_type == OutputType.CLASS_ID:
            # Return predicted class ID
            return int(np.argmax(model_output[0]))
        
        elif self.output_type == OutputType.PROBABILITIES:
            # Convert logits to probabilities using sigmoid
            import numpy as np
            logits = model_output[0]
            probabilities = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
            return probabilities.tolist()
        
        elif self.output_type == OutputType.LOGITS:
            # Return raw logits
            return model_output[0].tolist()
        
        elif self.output_type == OutputType.REGRESSION:
            # Return regression value
            return float(model_output[0][0])
        
        elif self.output_type == OutputType.EMBEDDINGS:
            # Return embeddings
            return model_output[0].tolist()
        
        else:
            # Return raw output
            return model_output[0].tolist()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model."""
        if not self.model_loaded:
            return {"name": "TensorFlowLiteAdapter", "loaded": False}
        
        # Get model size
        model_size = os.path.getsize(self.model_path) / (1024 * 1024)  # MB
        
        # Count parameters (approximate)
        total_params = 0
        for detail in self.input_details + self.output_details:
            if 'shape' in detail:
                shape = detail['shape']
                if len(shape) > 0:
                    total_params += np.prod(shape)
        
        return {
            "name": "TensorFlowLiteAdapter",
            "model_path": self.model_path,
            "model_size_mb": model_size,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "task_type": self.task_type,
            "input_type": self.input_type.value,
            "output_type": self.output_type.value,
            "total_parameters": int(total_params),
            "loaded": True,
            "description": f"TensorFlow Lite model: {os.path.basename(self.model_path)}"
        }
    
    def get_model_type(self) -> ModelType:
        """Return the type of model."""
        return ModelType.TENSORFLOW_LITE
    
    def validate_compatibility(self, dataset: 'BaseDataset') -> bool:
        """Validate that this model is compatible with the given dataset."""
        return dataset.output_type == self.input_type
