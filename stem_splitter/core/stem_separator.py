from typing import Dict, List, Optional
import numpy as np
import torch
from pathlib import Path
import os
from ..config import SeparationConfig
from ..models.model_loader import ModelLoader
from .audio_processor import AudioProcessor

class StemSeparator:
    """Core class for separating audio into stems."""
    
    def __init__(self, config: SeparationConfig):
        self.config = config
        self.model_loader = ModelLoader()
        self.audio_processor = AudioProcessor(config)
        self.model = None
        
    def load_model(self):
        """Load the separation model."""
        self.model = self.model_loader.load_model(
            model_name=self.config.model_name,
            device=self.config.device
        )
        
    def separate(self, 
                 input_file: Path, 
                 output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Separate audio file into stems.
        
        Args:
            input_file: Path to input audio file
            output_dir: Directory to save separated stems
            
        Returns:
            Dictionary mapping stem names to output file paths
        """
        if self.model is None:
            self.load_model()
            
        # Load audio
        audio, sr = self.audio_processor.load_audio(input_file)
        
        # Prepare output directory
        if output_dir is None:
            output_dir = Path(input_file).parent / "stems"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to tensor for model input
        if isinstance(audio, np.ndarray):
            # Ensure correct shape: [channels, samples]
            if audio.ndim == 1:
                audio = np.expand_dims(audio, axis=0)
            elif audio.ndim > 1 and audio.shape[0] > 2:
                # Assuming it's in [samples, channels] format
                audio = audio.T
                
            # Convert to tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            
            # Add batch dimension if needed
            if audio_tensor.ndim == 2:
                audio_tensor = audio_tensor.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")
        
        # Move to device
        audio_tensor = audio_tensor.to(self.config.device)
        
        # Perform separation
        with torch.no_grad():
            stems_dict = self.model(audio_tensor)
        
        # Process and save stems
        output_files = {}
        
        for stem_name, stem_audio in stems_dict.items():
            if stem_name in self.config.stems:
                # Move back to CPU and convert to numpy
                stem_np = stem_audio.cpu().numpy().squeeze()
                
                # Generate output filename
                output_file = output_dir / f"{Path(input_file).stem}_{stem_name}.{self.config.output_format}"
                
                # Save the stem
                saved_path = self.audio_processor.save_audio(stem_np, output_file, sr)
                output_files[stem_name] = saved_path
                
        return output_files