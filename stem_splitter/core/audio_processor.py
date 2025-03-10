import numpy as np
import librosa
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path
import soundfile as sf
from ..config import SeparationConfig, AnalysisConfig

class AudioProcessor:
    """Handles loading, processing, and saving audio files."""
    
    def __init__(self, config: SeparationConfig):
        self.config = config
        
    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file and return the waveform and sample rate."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Load audio with librosa
        audio, sr = librosa.load(file_path, sr=self.config.sample_rate, mono=False)
        
        # Convert mono to stereo if needed
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        
        return audio, sr
    
    def save_audio(self, 
                   audio: np.ndarray, 
                   output_path: Union[str, Path], 
                   sample_rate: Optional[int] = None):
        """Save audio to disk."""
        if sample_rate is None:
            sample_rate = self.config.sample_rate
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure audio is correctly shaped
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        
        sf.write(output_path, audio.T, sample_rate)
        return output_path
    
    def analyze_audio(self, audio: np.ndarray, sr: int, analysis_config: AnalysisConfig) -> Dict:
        """Analyze audio to extract musical features."""
        results = {}
        
        # Convert to mono for analysis if stereo
        if audio.ndim > 1 and audio.shape[0] > 1:
            y_mono = np.mean(audio, axis=0)
        else:
            y_mono = audio.squeeze()
        
        if analysis_config.detect_key:
            key = self._detect_key(y_mono, sr)
            results['key'] = key
            
        if analysis_config.detect_tempo:
            tempo = self._detect_tempo(y_mono, sr)
            results['tempo'] = tempo
            
        if analysis_config.detect_chord_progression:
            chords = self._detect_chords(y_mono, sr)
            results['chord_progression'] = chords
            
        return results
    
    def _detect_key(self, y_mono: np.ndarray, sr: int) -> str:
        """Detect musical key of audio."""
        # Use librosa's key detection (chroma features + key profile correlation)
        chroma = librosa.feature.chroma_cqt(y=y_mono, sr=sr)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Major and minor profiles
        maj_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        min_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        
        # Calculate mean chroma vector
        chroma_mean = np.mean(chroma, axis=1)
        
        # Correlate with each possible key
        max_corr = -1
        key_result = None
        
        for i in range(12):  # For each possible key
            # Correlate with major profile
            maj_corr = np.corrcoef(np.roll(maj_profile, i), chroma_mean)[0, 1]
            # Correlate with minor profile
            min_corr = np.corrcoef(np.roll(min_profile, i), chroma_mean)[0, 1]
            
            if maj_corr > max_corr:
                max_corr = maj_corr
                key_result = f"{key_names[i]} Major"
            
            if min_corr > max_corr:
                max_corr = min_corr
                key_result = f"{key_names[i]} Minor"
        
        return key_result
    
    def _detect_tempo(self, y_mono: np.ndarray, sr: int) -> float:
        """Detect tempo (BPM) of audio."""
        onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        return round(tempo, 1)
    
    def _detect_chords(self, y_mono: np.ndarray, sr: int) -> List[Tuple[float, str]]:
        """Detect chord progression in audio."""
        # This is a simplified implementation - more sophisticated chord detection
        # would likely require a dedicated model
        chroma = librosa.feature.chroma_cqt(y=y_mono, sr=sr)
        
        # Get frame times
        frames = np.arange(chroma.shape[1])
        times = librosa.frames_to_time(frames, sr=sr)
        
        chord_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chord_progression = []
        
        # Simple chord detection based on strongest notes in chroma
        for i, time in enumerate(times):
            if i % 10 == 0:  # Sample at regular intervals to avoid too many chords
                chroma_frame = chroma[:, i]
                root = np.argmax(chroma_frame)
                
                # Determine major or minor based on third
                third = (root + 4) % 12  # Major third
                minor_third = (root + 3) % 12  # Minor third
                
                if chroma_frame[minor_third] > chroma_frame[third]:
                    chord = f"{chord_names[root]}m"
                else:
                    chord = chord_names[root]
                
                chord_progression.append((time, chord))
        
        return chord_progression