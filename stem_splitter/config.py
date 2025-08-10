import os
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path

def get_optimal_device():
    """Get the optimal device for processing."""
    if torch.cuda.is_available():
        # Check GPU memory
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            if gpu_memory >= 4.0:  # Minimum 4GB for reasonable performance
                return "cuda"
            else:
                print(f"Warning: GPU has only {gpu_memory:.1f}GB memory. Using CPU.")
                return "cpu"
        except Exception:
            return "cpu"
    else:
        return "cpu"

def get_optimal_batch_size(device: str, model_complexity: str = "medium"):
    """Get optimal batch size based on device and model complexity."""
    if device == "cpu":
        return 1  # CPU processing is typically single-batch
    
    if not torch.cuda.is_available():
        return 1
    
    try:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Adjust batch size based on GPU memory and model complexity
        if model_complexity == "simple":
            base_batch_size = max(1, int(gpu_memory_gb / 2))
        elif model_complexity == "medium":
            base_batch_size = max(1, int(gpu_memory_gb / 4))
        else:  # complex
            base_batch_size = max(1, int(gpu_memory_gb / 8))
        
        return min(base_batch_size, 4)  # Cap at 4 for stability
    except Exception:
        return 1

@dataclass
class SeparationConfig:
    model_name: str = "htdemucs"  # Default separation model
    stems: List[str] = None  # Stems to extract, None = all available
    sample_rate: int = 44100
    device: str = None  # Auto-detect if None
    output_format: str = "wav"
    batch_size: int = None  # Auto-optimize if None
    mixed_precision: bool = True  # Use mixed precision for GPU acceleration
    enable_memory_optimization: bool = True  # Enable gradient checkpointing and other optimizations
    max_gpu_memory_gb: Optional[float] = None  # Limit GPU memory usage
    
    def __post_init__(self):
        if self.stems is None:
            self.stems = ["vocals", "drums", "bass", "other"]
        
        # Auto-detect device if not specified
        if self.device is None:
            self.device = get_optimal_device()
        # Override with environment variable if set
        elif os.environ.get("CUDA_VISIBLE_DEVICES") and self.device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if self.batch_size is None:
            complexity = self._get_model_complexity()
            self.batch_size = get_optimal_batch_size(self.device, complexity)
        
        # Set mixed precision only for GPU
        if self.device == "cpu":
            self.mixed_precision = False
    
    def _get_model_complexity(self) -> str:
        """Estimate model complexity for memory planning."""
        if "htdemucs" in self.model_name:
            if "6s" in self.model_name:
                return "complex"
            return "medium"
        elif "demucs" in self.model_name:
            return "medium"
        else:
            return "simple"
    
    def optimize_for_gpu(self):
        """Optimize settings specifically for GPU processing."""
        if self.device == "cuda" and torch.cuda.is_available():
            # Enable optimizations
            self.mixed_precision = True
            self.enable_memory_optimization = True
            
            # Set optimal memory limit
            if self.max_gpu_memory_gb is None:
                try:
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    self.max_gpu_memory_gb = total_memory * 0.8  # Use 80% of available memory
                except Exception:
                    self.max_gpu_memory_gb = 4.0  # Default fallback
            
            print(f"GPU optimization enabled:")
            print(f"  - Mixed precision: {self.mixed_precision}")
            print(f"  - Memory optimization: {self.enable_memory_optimization}")
            print(f"  - Max GPU memory: {self.max_gpu_memory_gb:.1f}GB")
            print(f"  - Batch size: {self.batch_size}")

@dataclass
class AnalysisConfig:
    detect_key: bool = True
    detect_tempo: bool = True
    detect_chord_progression: bool = False
    enable_gpu_acceleration: bool = True  # Use GPU for analysis if available
    chunk_size: int = 30  # Process audio in chunks (seconds) to manage memory
    advanced_features: bool = False  # Enable advanced analysis features