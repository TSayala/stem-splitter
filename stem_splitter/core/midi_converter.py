import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import mido
from mido import Message, MidiFile, MidiTrack
import librosa
from ..config import SeparationConfig

class MidiConverter:
    """Converts audio stems to MIDI."""
    
    def __init__(self, config: SeparationConfig):
        self.config = config
        
    def audio_to_midi(self, 
                      audio: np.ndarray, 
                      sr: int, 
                      output_file: Path,
                      instrument_type: str = "melodic",
                      velocity_sensitivity: float = 1.0) -> Path:
        """
        Convert audio stem to MIDI.
        
        Args:
            audio: Audio data as numpy array
            sr: Sample rate
            output_file: Path to save MIDI file
            instrument_type: "melodic" or "percussive"
            velocity_sensitivity: Controls how dynamic the velocities are
            
        Returns:
            Path to the saved MIDI file
        """
        # Create MIDI file
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        # Set tempo (assuming we've analyzed the audio beforehand)
        tempo = self._estimate_tempo(audio, sr)
        tempo_in_microsec = int(60_000_000 / tempo)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo_in_microsec, time=0))
        
        # Process differently based on instrument type
        if instrument_type == "melodic":
            self._convert_melodic(audio, sr, track, velocity_sensitivity)
        elif instrument_type == "percussive":
            self._convert_percussive(audio, sr, track, velocity_sensitivity)
        else:
            raise ValueError(f"Unsupported instrument type: {instrument_type}")
        
        # Save the MIDI file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        mid.save(str(output_file))
        
        return output_file
    
    def _estimate_tempo(self, audio: np.ndarray, sr: int) -> float:
        """Estimate tempo from audio."""
        # Ensure audio is mono
        if audio.ndim > 1:
            y_mono = np.mean(audio, axis=0)
        else:
            y_mono = audio
            
        onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        return tempo
    
    def _convert_melodic(self, 
                         audio: np.ndarray, 
                         sr: int, 
                         track: MidiTrack,
                         velocity_sensitivity: float):
        """Convert melodic audio (like vocals, bass) to MIDI notes."""
        # Ensure mono for melody extraction
        if audio.ndim > 1:
            y_mono = np.mean(audio, axis=0)
        else:
            y_mono = audio
            
        # Extract pitch
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y_mono, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Convert pitch to MIDI notes
        midi_notes = []
        current_note = None
        note_start_frame = 0
        prev_note = None
        
        for i, (pitch, is_voiced) in enumerate(zip(f0, voiced_flag)):
            if is_voiced and not np.isnan(pitch):
                midi_pitch = int(round(librosa.hz_to_midi(pitch)))
                
                # If this is a new note or a different note
                if current_note is None or current_note != midi_pitch:
                    # End the previous note if there was one
                    if current_note is not None:
                        # Calculate the note duration in MIDI ticks
                        duration_frames = i - note_start_frame
                        duration_seconds = duration_frames * librosa.frames_to_time(1, sr=sr)
                        duration_ticks = int(duration_seconds * 480)  # Assuming 480 ticks per quarter note
                        
                        # Calculate velocity based on the amplitude during the note
                        segment = y_mono[note_start_frame:i]
                        if len(segment) > 0:
                            velocity = min(127, max(1, int(np.mean(np.abs(segment)) * 500 * velocity_sensitivity)))
                        else:
                            velocity = 64
                        
                        # Add the note to our list
                        midi_notes.append({
                            'note': current_note,
                            'start': note_start_frame,
                            'end': i,
                            'duration': duration_ticks,
                            'velocity': velocity
                        })
                    
                    # Start the new note
                    current_note = midi_pitch
                    note_start_frame = i
                    
            # If not voiced but we have a current note, end it
            elif not is_voiced and current_note is not None:
                # Calculate duration
                duration_frames = i - note_start_frame
                duration_seconds = duration_frames * librosa.frames_to_time(1, sr=sr)
                duration_ticks = int(duration_seconds * 480)
                
                # Calculate velocity
                segment = y_mono[note_start_frame:i]
                if len(segment) > 0:
                    velocity = min(127, max(1, int(np.mean(np.abs(segment)) * 500 * velocity_sensitivity)))
                else:
                    velocity = 64
                
                # Add the note
                midi_notes.append({
                    'note': current_note,
                    'start': note_start_frame,
                    'end': i,
                    'duration': duration_ticks,
                    'velocity': velocity
                })
                
                current_note = None
        
        # End final note if needed
        if current_note is not None:
            duration_frames = len(f0) - note_start_frame
            duration_seconds = duration_frames * librosa.frames_to_time(1, sr=sr)
            duration_ticks = int(duration_seconds * 480)
            
            segment = y_mono[note_start_frame:]
            if len(segment) > 0:
                velocity = min(127, max(1, int(np.mean(np.abs(segment)) * 500 * velocity_sensitivity)))
            else:
                velocity = 64
            
            midi_notes.append({
                'note': current_note,
                'start': note_start_frame,
                'end': len(f0),
                'duration': duration_ticks,
                'velocity': velocity
            })
        
        # Sort notes by start time
        midi_notes.sort(key=lambda x: x['start'])
        
        # Add notes to the track
        current_time = 0
        for note in midi_notes:
            # Calculate delta time from the previous event
            delta_time = note['start'] - current_time if current_time < note['start'] else 0
            delta_ticks = int(delta_time * librosa.frames_to_time(1, sr=sr) * 480)
            
            # Note on
            track.append(Message('note_on', note=note['note'], velocity=note['velocity'], time=delta_ticks))
            
            # Note off (delta time is the duration)
            track.append(Message('note_off', note=note['note'], velocity=0, time=note['duration']))
            
            current_time = note['start'] + note['duration']
    
    def _convert_percussive(self, 
                           audio: np.ndarray, 
                           sr: int, 
                           track: MidiTrack,
                           velocity_sensitivity: float):
        """Convert percussive audio (like drums) to MIDI notes."""
        # Ensure mono
        if audio.ndim > 1:
            y_mono = np.mean(audio, axis=0)
        else:
            y_mono = audio
            
        # Detect onsets
        onset_frames = librosa.onset.onset_detect(y=y_mono, sr=sr)
        
        # Drum mapping - simplified version
        # General MIDI drum mapping: 35=Acoustic Bass Drum, 38=Acoustic Snare, 42=Closed Hi-Hat, 46=Open Hi-Hat
        drum_mapping = {
            'kick': 36,    # Bass Drum
            'snare': 38,   # Snare Drum
            'hihat': 42,   # Closed Hi-Hat
            'tom': 45      # Tom
        }
        
        # For each onset, classify the drum type and add a MIDI note
        current_time = 0
        
        for i, frame in enumerate(onset_frames):
            # Calculate delta time
            time_sec = librosa.frames_to_time(frame, sr=sr)
            delta_ticks = int(time_sec * 480)  # Convert to MIDI ticks
            
            # Get a short segment after the onset
            segment_start = frame
            segment_end = min(len(y_mono), frame + int(0.1 * sr))  # 100ms segment
            segment = y_mono[segment_start:segment_end]
            
            # Simplified drum classification based on spectral features
            spec = np.abs(librosa.stft(segment))
            spec_db = librosa.amplitude_to_db(spec)
            
            # Simple heuristic to classify drum sounds
            # Low frequency content for kick, mid for snare, high for hi-hat
            low_energy = np.mean(spec[:10, :])
            mid_energy = np.mean(spec[10:30, :])
            high_energy = np.mean(spec[30:, :])
            
            # Determine drum type
            if low_energy > mid_energy and low_energy > high_energy:
                drum_type = 'kick'
            elif mid_energy > high_energy:
                drum_type = 'snare'
            else:
                drum_type = 'hihat'
                
            # Calculate velocity based on the peak amplitude
            velocity = min(127, max(1, int(np.max(np.abs(segment)) * 300 * velocity_sensitivity)))
            
            # Add note to track
            note = drum_mapping[drum_type]
            track.append(Message('note_on', note=note, velocity=velocity, time=delta_ticks - current_time))
            track.append(Message('note_off', note=note, velocity=0, time=10))  # Short duration for drums
            
            current_time = delta_ticks + 10