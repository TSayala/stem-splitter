import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path

@dataclass
class SeparationConfig:
    model_name: str = "htdemucs"  # Default separation model
    stems: List[str] = None  # Stems to extract, None = all available
    sample_rate: int = 44100
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    output_format: str = "wav"
    
    def __post_init__(self):
        if self.stems is None:
            self.stems = ["vocals", "drums", "bass", "other"]


@dataclass
class AnalysisConfig:
    detect_key: bool = True
    detect_tempo: bool = True
    detect_chord_progression: bool = False