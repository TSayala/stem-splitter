import numpy as np
import librosa
import soundfile as sf
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union

def load_audio(file_path: Union[str, Path], 
               sr: Optional[int] = None, 
               mono: bool = False) -> Tuple[np.ndarray, int]:
    """
    Load audio file with support for various formats.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate (None for original)
        mono: Convert to mono
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        # Try loading with librosa (supports more formats)
        audio, sr_orig = librosa.load(file_path, sr=sr, mono=mono)
        
        # If stereo is needed but we got mono
        if not mono and audio.ndim == 1:
            audio = np.stack([audio, audio])
            
    except Exception as e:
        # Fall back to soundfile
        audio, sr_orig = sf.read(file_path)
        
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
            
        # Resample if needed
        if sr is not None and sr != sr_orig:
            audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=sr)
            sr_orig = sr
            
        # Handle mono/stereo conversion
        if mono and audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        elif not mono and audio.ndim == 1:
            audio = np.stack([audio, audio])
            
    return audio, sr_orig

def save_audio(audio: np.ndarray, 
               file_path: Union[str, Path], 
               sr: int, 
               format: Optional[str] = None):
    """
    Save audio to file.
    
    Args:
        audio: Audio data
        file_path: Output file path
        sr: Sample rate
        format: Output format (None to infer from extension)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Infer format from extension if not provided
    if format is None:
        format = file_path.suffix.lower().strip('.')
        if not format:
            format = 'wav'
            
    # Ensure audio has correct shape for soundfile
    if audio.ndim == 1:
        # Mono: [samples]
        pass
    elif audio.ndim == 2:
        if audio.shape[0] == 1 or audio.shape[0] == 2:
            # [channels, samples] -> [samples, channels]
            audio = audio.T
        # Else assume already in [samples, channels] format
    
    # Normalize audio if needed
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))
        
    # Save audio
    sf.write(file_path, audio, sr, format=format)
    
def convert_audio_format(input_file: Union[str, Path], 
                         output_format: str,
                         output_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Convert audio file to a different format.
    
    Args:
        input_file: Input audio file
        output_format: Target format (wav, mp3, etc.)
        output_dir: Directory for output (None for same as input)
        
    Returns:
        Path to converted file
    """
    input_file = Path(input_file)
    
    # Determine output path
    if output_dir is None:
        output_dir = input_file.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
    output_file = output_dir / f"{input_file.stem}.{output_format}"
    
    # Load audio
    audio, sr = load_audio(input_file)
    
    # Save in new format
    save_audio(audio, output_file, sr, format=output_format)
    
    return output_file

def get_audio_info(file_path: Union[str, Path]) -> Dict:
    """
    Get basic information about an audio file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with audio information
    """
    file_path = Path(file_path)
    
    # Get file info
    try:
        info = sf.info(file_path)
        
        # Load a small portion for additional analysis
        y, sr = librosa.load(file_path, sr=None, duration=30)
        
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "peak_amplitude": float(np.max(np.abs(y))),
            "rms_level_db": float(20 * np.log10(np.mean(rms) + 1e-8)),
            "file_size_mb": file_path.stat().st_size / (1024 * 1024)
        }
    except Exception as e:
        return {"error": str(e)}
