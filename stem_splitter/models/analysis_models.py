from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import librosa

class KeyDetector:
    """Detects musical key from audio."""
    
    def __init__(self):
        # Key profiles (Krumhansl-Schmuckler key profiles)
        self.major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        self.minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
    def detect_key(self, audio: np.ndarray, sr: int) -> str:
        """
        Detect the musical key of an audio track.
        
        Args:
            audio: Audio as numpy array
            sr: Sample rate
            
        Returns:
            String representation of the detected key (e.g., "C Major")
        """
        # Ensure mono audio
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
            
        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, bins_per_octave=12, n_octaves=7)
        
        # Average over time
        chroma_avg = np.mean(chroma, axis=1)
        
        # Normalize
        chroma_normalized = chroma_avg / chroma_avg.sum()
        
        # Correlate with key profiles for all possible keys
        correlations = []
        for i in range(12):  # 12 possible keys (C, C#, D, etc.)
            # Shift major profile to each key and correlate
            major_corr = np.corrcoef(np.roll(self.major_profile, i), chroma_normalized)[0, 1]
            
            # Shift minor profile to each key and correlate
            minor_corr = np.corrcoef(np.roll(self.minor_profile, i), chroma_normalized)[0, 1]
            
            correlations.append((i, "Major", major_corr))
            correlations.append((i, "Minor", minor_corr))
            
        # Find key with highest correlation
        best_key_idx, best_key_mode, best_corr = max(correlations, key=lambda x: x[2])
        
        # Map index to key name
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_name = f"{key_names[best_key_idx]} {best_key_mode}"
        
        return key_name


class TempoDetector:
    """Detects tempo (BPM) from audio."""
    
    def detect_tempo(self, audio: np.ndarray, sr: int) -> float:
        """
        Detect the tempo of an audio track in beats per minute (BPM).
        
        Args:
            audio: Audio as numpy array
            sr: Sample rate
            
        Returns:
            Tempo in BPM
        """
        # Ensure mono audio
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
            
        # Compute onset strength envelope
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        
        # Dynamic tempo estimation
        prior_bpm = 120.0  # Prior on 120 BPM
        dtempo = librosa.beat.tempo(
            onset_envelope=onset_env, 
            sr=sr,
            prior_bpm=prior_bpm,
            aggregate=None
        )
        
        # Use dynamic programming beat tracker for more accurate results
        beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[1]
        
        if len(beats) > 1:
            # Calculate tempo from beat positions
            beat_times = librosa.frames_to_time(beats, sr=sr)
            beat_diffs = np.diff(beat_times)
            
            # Calculate average beat duration and convert to BPM
            avg_beat_duration = np.mean(beat_diffs)
            tempo = 60.0 / avg_beat_duration
        else:
            # Fall back to estimated tempo if beat tracking fails
            tempo = np.median(dtempo)
            
        return float(tempo)


class ChordDetector:
    """Detects chord progressions from audio."""
    
    def __init__(self):
        # Chord templates (simplified major/minor triads)
        self.chord_templates = {
            'major': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),  # Root, major third, fifth
            'minor': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),  # Root, minor third, fifth
            '7': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),      # Root, major third, fifth, minor seventh
            'm7': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]),     # Root, minor third, fifth, minor seventh
        }
        
        self.root_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
    def detect_chords(self, audio: np.ndarray, sr: int) -> List[Tuple[float, str]]:
        """
        Detect chord progression in audio.
        
        Args:
            audio: Audio as numpy array
            sr: Sample rate
            
        Returns:
            List of (time, chord) tuples
        """
        # Ensure mono audio
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
            
        # Compute chroma features
        hop_length = 512
        chroma = librosa.feature.chroma_cqt(
            y=audio, 
            sr=sr,
            hop_length=hop_length,
            bins_per_octave=12
        )
        
        # Get frame times
        frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
        
        # Segment into beats for chord detection
        # First, detect beats
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=hop_length)
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
        
        # If beat detection fails, use fixed segment size
        if len(beats) < 2:
            segment_size = int(sr * 0.5 / hop_length)  # 0.5 second segments
            segments = np.arange(0, chroma.shape[1], segment_size)
            segment_times = librosa.frames_to_time(segments, sr=sr, hop_length=hop_length)
        else:
            segments = beats
            segment_times = beat_times
            
        # Initialize chord progression
        chord_progression = []
        
        # Detect chord for each segment
        for i in range(len(segments) - 1):
            start_frame = segments[i]
            end_frame = segments[i + 1]
            
            # Average chroma over segment
            segment_chroma = np.mean(chroma[:, start_frame:end_frame], axis=1)
            
            # Find best matching chord
            best_chord = None
            best_score = -1
            
            for root in range(12):
                for chord_type, template in self.chord_templates.items():
                    # Shift template to current root
                    shifted_template = np.roll(template, root)
                    
                    # Compute correlation
                    corr = np.corrcoef(segment_chroma, shifted_template)[0, 1]
                    
                    if corr > best_score:
                        best_score = corr
                        if chord_type == 'major':
                            best_chord = self.root_names[root]
                        elif chord_type == 'minor':
                            best_chord = f"{self.root_names[root]}m"
                        elif chord_type == '7':
                            best_chord = f"{self.root_names[root]}7"
                        elif chord_type == 'm7':
                            best_chord = f"{self.root_names[root]}m7"
            
            # Add to progression
            chord_progression.append((segment_times[i], best_chord))
            
        return chord_progression