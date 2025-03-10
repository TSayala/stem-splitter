from typing import Dict, Optional, Union
import torch
import torchaudio
from pathlib import Path
import os
import importlib
import json

class ModelLoader:
    """Handles loading and caching of separation models."""
    
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or Path(os.path.expanduser("~/.stem_splitter/models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models = {}
        
    def load_model(self, model_name: str, device: str = "cpu"):
        """
        Load a separation model.
        
        Args:
            model_name: Name of the model to load
            device: Device to load the model on ('cpu' or 'cuda')
            
        Returns:
            The loaded model
        """
        # Check if model is already loaded
        cache_key = f"{model_name}_{device}"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        # Load appropriate model based on name
        if model_name.startswith("htdemucs"):
            model = self._load_htdemucs(model_name, device)
        elif model_name.startswith("demucs"):
            model = self._load_demucs(model_name, device)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Cache model
        self.loaded_models[cache_key] = model
        return model
    
    def _load_htdemucs(self, model_name: str, device: str):
        """Load HT-Demucs model."""
        try:
            # Import here to avoid dependency if not using this model
            from demucs.pretrained import get_model as get_demucs_model
            from demucs.apply import apply_model
            
            # Get the model
            model = get_demucs_model(model_name)
            model.to(device)
            model.eval()
            
            # Wrap the model for easier interface
            def separation_fn(audio_tensor):
                # Apply model expects tensor of shape [batch, channels, time]
                stems = apply_model(model, audio_tensor)
                
                # Convert to dictionary of stems
                stem_names = model.sources
                stems_dict = {name: stems[:, i] for i, name in enumerate(stem_names)}
                
                return stems_dict
            
            return separation_fn
            
        except ImportError:
            raise ImportError("HT-Demucs requires the 'demucs' package. Install with: pip install demucs")
    
    def _load_demucs(self, model_name: str, device: str):
        """Load Demucs model."""
        try:
            # Similar to HT-Demucs
            from demucs.pretrained import get_model as get_demucs_model
            from demucs.apply import apply_model
            
            model = get_demucs_model(model_name)
            model.to(device)
            model.eval()
            
            def separation_fn(audio_tensor):
                stems = apply_model(model, audio_tensor)
                stem_names = model.sources
                stems_dict = {name: stems[:, i] for i, name in enumerate(stem_names)}
                return stems_dict
            
            return separation_fn
            
        except ImportError:
            raise ImportError("Demucs requires the 'demucs' package. Install with: pip install demucs")