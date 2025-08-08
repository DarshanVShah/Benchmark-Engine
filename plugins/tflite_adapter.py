"""
TensorFlow Lite Model Adapter

This adapter handles TensorFlow Lite models for benchmarking.
Supports various input/output types and provides standardized interface.
"""

import os
import time
import numpy as np
from typing import Dict, Any, List, Optional, Union

# Conditional import to avoid errors when tensorflow is not installed
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

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
        self._input_type = DataType.TEXT  # Default, can be overridden
        self._output_type = OutputType.CLASS_ID  # Default, can be overridden
        self.task_type = "auto-detected"
        self.max_length = 512
        self.tokenizer = None

    @property
    def input_type(self) -> DataType:
        """Return the expected input type for this model."""
        return self._input_type

    @property
    def output_type(self) -> OutputType:
        """Return the output type this model produces."""
        return self._output_type

    def load(self, model_path: str) -> bool:
        """Load TensorFlow Lite model from file."""
        if not TENSORFLOW_AVAILABLE:
            print("Error: TensorFlow is not installed. Please install tensorflow: pip install tensorflow")
            return False
            
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
            self.input_shape = self.input_details[0]["shape"]
            self.output_shape = self.output_details[0]["shape"]

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

            # Set input/output types
            if "input_type" in config:
                self._input_type = DataType(config["input_type"])
                print(f"  Input type set to: {self._input_type.value}")

            if "output_type" in config:
                self._output_type = OutputType(config["output_type"])
                print(f"  Output type set to: {self._output_type.value}")

            # Set max length for text models
            if "max_length" in config:
                self.max_length = config["max_length"]
                print(f"  Max length set to: {self.max_length}")

            print(f"  TensorFlow Lite model configured")
            return True

        except Exception as e:
            print(f"  Error configuring TensorFlow Lite model: {e}")
            return False

    def preprocess(self, raw_input: Any) -> Any:
        """
        Convert raw input to model-specific format.

        Args:
            raw_input: Raw input from dataset

        Returns:
            Preprocessed input ready for model inference
        """
        try:
            # Determine input type and preprocess accordingly
            if self.input_type == DataType.TEXT:
                return self._preprocess_text(raw_input)
            elif self.input_type == DataType.IMAGE:
                return self._preprocess_image(raw_input)
            elif self.input_type == DataType.TENSOR:
                return self._preprocess_tensor(raw_input)
            else:
                # Default to tensor preprocessing
                return self._preprocess_tensor(raw_input)

        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None

    def _preprocess_text(self, raw_input: Union[str, Dict[str, Any]]) -> np.ndarray:
        """Preprocess text input for TFLite model."""
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
                text = str(raw_input)

            # Simple character-level tokenization
            # In a real implementation, you'd use a proper tokenizer
            chars = list(text.lower())
            vocab = list(set(chars))
            char_to_idx = {char: idx for idx, char in enumerate(vocab)}

            # Tokenize
            tokens = [char_to_idx.get(char, 0) for char in chars]

            # Pad/truncate to expected length
            expected_length = self.input_shape[1] if len(self.input_shape) > 1 else 1
            if len(tokens) > expected_length:
                tokens = tokens[:expected_length]
            else:
                tokens.extend([0] * (expected_length - len(tokens)))

            # Convert to numpy array
            return np.array([tokens], dtype=np.int32)

        except Exception as e:
            print(f"Error in text preprocessing: {e}")
            return np.zeros(self.input_shape, dtype=np.int32)

    def _preprocess_image(self, raw_input: Union[str, Dict[str, Any]]) -> np.ndarray:
        """Preprocess image input for TFLite model."""
        try:
            # Extract image path or data
            if isinstance(raw_input, dict):
                if "image" in raw_input:
                    image_path = raw_input["image"]
                elif "path" in raw_input:
                    image_path = raw_input["path"]
                else:
                    raise ValueError("No image path found in input")
            elif isinstance(raw_input, str):
                image_path = raw_input
            else:
                raise ValueError("Invalid image input format")

            # Load and preprocess image
            # This is a simplified implementation
            # In a real implementation, you'd use proper image loading and preprocessing
            try:
                import cv2
                
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")

                # Resize to expected input shape
                if len(self.input_shape) == 4:  # [batch, height, width, channels]
                    target_height, target_width = self.input_shape[1], self.input_shape[2]
                else:
                    target_height, target_width = 224, 224  # Default size

                resized = cv2.resize(image, (target_width, target_height))
                
                # Normalize pixel values
                normalized = resized.astype(np.float32) / 255.0
                
                # Add batch dimension if needed
                if len(normalized.shape) == 3:
                    normalized = np.expand_dims(normalized, axis=0)

                return normalized
            except ImportError:
                print("Warning: OpenCV not available, using dummy image data")
                return np.random.rand(*self.input_shape).astype(np.float32)

        except Exception as e:
            print(f"Error in image preprocessing: {e}")
            return np.zeros(self.input_shape, dtype=np.float32)

    def _preprocess_tensor(self, raw_input: Any) -> np.ndarray:
        """Preprocess tensor input for TFLite model."""
        try:
            # Convert to numpy array
            if isinstance(raw_input, np.ndarray):
                tensor = raw_input
            elif isinstance(raw_input, (list, tuple)):
                tensor = np.array(raw_input)
            else:
                tensor = np.array([raw_input])

            # Ensure correct shape
            if tensor.shape != self.input_shape:
                # Try to reshape
                try:
                    tensor = tensor.reshape(self.input_shape)
                except ValueError:
                    # Pad or truncate to match expected shape
                    if len(tensor.shape) == 1 and len(self.input_shape) == 2:
                        # Add batch dimension
                        tensor = np.expand_dims(tensor, axis=0)

            return tensor.astype(np.float32)

        except Exception as e:
            print(f"Error in tensor preprocessing: {e}")
            return np.zeros(self.input_shape, dtype=np.float32)

    def run(self, preprocessed_input: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed input.

        Args:
            preprocessed_input: Preprocessed input from preprocess()

        Returns:
            Raw model output
        """
        try:
            if not self.model_loaded:
                print("Error: Model not loaded")
                return None

            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]["index"], preprocessed_input)

            # Run inference
            self.interpreter.invoke()

            # Get output tensor
            output_data = self.interpreter.get_tensor(self.output_details[0]["index"])

            return output_data

        except Exception as e:
            print(f"Error in model inference: {e}")
            return None

    def postprocess(self, model_output: np.ndarray) -> Any:
        """
        Convert raw model output to standardized format.

        Args:
            model_output: Raw output from model

        Returns:
            Standardized prediction
        """
        try:
            if model_output is None:
                return None

            # Convert to standardized format based on output type
            if self.output_type == OutputType.CLASS_ID:
                # For classification, return class IDs
                predicted_class = np.argmax(model_output, axis=-1)
                return predicted_class.tolist()
            
            elif self.output_type == OutputType.PROBABILITIES:
                # For probabilities, return as-is
                return model_output.tolist()
            
            elif self.output_type == OutputType.LOGITS:
                # For logits, return as-is
                return model_output.tolist()
            
            else:
                # Default: return as-is
                return model_output.tolist()

        except Exception as e:
            print(f"Error in postprocessing: {e}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model."""
        info = {
            "name": os.path.basename(self.model_path),
            "type": "tensorflow_lite",
            "path": self.model_path,
            "loaded": self.model_loaded,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "input_type": self.input_type.value,
            "output_type": self.output_type.value,
            "task_type": self.task_type,
        }

        return info

    def get_model_type(self) -> ModelType:
        """Return the type of model."""
        return ModelType.TENSORFLOW_LITE

    def validate_compatibility(self, dataset: "BaseDataset") -> bool:
        """Validate that this model is compatible with the given dataset."""
        return dataset.output_type == self.input_type
