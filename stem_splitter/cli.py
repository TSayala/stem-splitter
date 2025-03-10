import argparse
import os
import sys
from pathlib import Path
import json
from typing import List, Dict, Optional

from .config import SeparationConfig, AnalysisConfig
from .core.stem_separator import StemSeparator
from .core.audio_processor import AudioProcessor
from .core.midi_converter import MidiConverter
from .utils.audio_utils import get_audio_info
from .utils.file_utils import save_metadata

def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Stem Splitter - Audio separation tool")
    
    # Define subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Separation command
    sep_parser = subparsers.add_parser("separate", help="Separate audio into stems")
    sep_parser.add_argument("input", help="Input audio file")
    sep_parser.add_argument("--output-dir", "-o", help="Output directory")
    sep_parser.add_argument("--model", default="htdemucs", help="Separation model to use")
    sep_parser.add_argument("--stems", nargs="+", help="Stems to extract (default: all)")
    sep_parser.add_argument("--format", default="wav", help="Output format (wav, mp3, etc.)")
    
    # Analysis command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze audio")
    analyze_parser.add_argument("input", help="Input audio file")
    analyze_parser.add_argument("--output", "-o", help="Output JSON file for results")
    analyze_parser.add_argument("--detect-key", action="store_true", help="Detect musical key")
    analyze_parser.add_argument("--detect-tempo", action="store_true", help="Detect tempo")
    analyze_parser.add_argument("--detect-chords", action="store_true", help="Detect chord progression")
    
    # MIDI conversion command
    midi_parser = subparsers.add_parser("to-midi", help="Convert audio to MIDI")
    midi_parser.add_argument("input", help="Input audio file")
    midi_parser.add_argument("--output", "-o", help="Output MIDI file")
    midi_parser.add_argument("--instrument-type", choices=["melodic", "percussive"], 
                           default="melodic", help="Instrument type")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get audio information")
    info_parser.add_argument("input", help="Input audio file")
    
    # List models command
    list_models_parser = subparsers.add_parser("list-models", help="List available separation models")
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Handle commands
        if args.command == "separate":
            run_separation(args)
        elif args.command == "analyze":
            run_analysis(args)
        elif args.command == "to-midi":
            run_midi_conversion(args)
        elif args.command == "info":
            run_info(args)
        elif args.command == "list-models":
            list_models()
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        
def run_separation(args):
    """Run audio separation."""
    input_file = Path(args.input)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_file.parent / "stems" / input_file.stem
        
    # Create config
    config = SeparationConfig(
        model_name=args.model,
        stems=args.stems,
        output_format=args.format
    )
    
    # Initialize separator
    separator = StemSeparator(config)
    
    print(f"Separating {input_file.name} using {args.model}...")
    
    # Run separation
    output_files = separator.separate(input_file, output_dir)
    
    print("Separation complete. Stems saved to:")
    for stem, path in output_files.items():
        print(f"  - {stem}: {path}")

def run_analysis(args):
    """Run audio analysis."""
    input_file = Path(args.input)
    
    # Create configs
    separation_config = SeparationConfig()
    analysis_config = AnalysisConfig(
        detect_key=args.detect_key,
        detect_tempo=args.detect_tempo,
        detect_chord_progression=args.detect_chords
    )
    
    # Initialize processor
    processor = AudioProcessor(separation_config)
    
    print(f"Analyzing {input_file.name}...")
    
    # Load audio
    audio, sr = processor.load_audio(input_file)
    
    # Run analysis
    results = processor.analyze_audio(audio, sr, analysis_config)
    
    # Display results
    print("\nAnalysis Results:")
    for key, value in results.items():
        if key == 'chord_progression':
            print("\nChord Progression:")
            for time, chord in value:
                print(f"  {time:.2f}s: {chord}")
        else:
            print(f"  {key}: {value}")
    
    # Save to file if requested
    if args.output:
        output_file = Path(args.output)
        save_metadata(results, output_file)
        print(f"\nResults saved to {output_file}")

def run_midi_conversion(args):
    """Run audio to MIDI conversion."""
    input_file = Path(args.input)
    
    # Setup output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.with_suffix('.mid')
    
    # Create config and initialize converter
    config = SeparationConfig()
    processor = AudioProcessor(config)
    converter = MidiConverter(config)
    
    print(f"Converting {input_file.name} to MIDI...")
    
    # Load audio
    audio, sr = processor.load_audio(input_file)
    
    # Convert to MIDI
    midi_file = converter.audio_to_midi(
        audio, 
        sr, 
        output_file,
        instrument_type=args.instrument_type
    )
    
    print(f"Conversion complete. MIDI saved to {midi_file}")

def run_info(args):
    """Display audio file information."""
    input_file = Path(args.input)
    
    print(f"Getting information for {input_file.name}...")
    
    # Get info
    info = get_audio_info(input_file)
    
    # Display info
    print("\nAudio Information:")
    for key, value in info.items():
        if key == 'duration':
            minutes = int(value // 60)
            seconds = int(value % 60)
            print(f"  Duration: {minutes}:{seconds:02d}")
        elif key == 'rms_level_db':
            print(f"  RMS Level: {value:.1f} dB")
        elif key == 'peak_amplitude':
            print(f"  Peak Amplitude: {value:.4f}")
        elif key == 'file_size_mb':
            print(f"  File Size: {value:.2f} MB")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")

def list_models():
    """List available separation models."""
    print("Available Separation Models:")
    print("\nDemucs Models:")
    print("  - htdemucs       (High quality, 4 stems: vocals, drums, bass, other)")
    print("  - htdemucs_ft    (Fine-tuned version)")
    print("  - htdemucs_6s    (6 stems: vocals, drums, bass, guitar, piano, other)")
    print("  - demucs         (Original Demucs v2)")
    print("  - demucs_extra   (Demucs with extra training)")
    
    print("\nFor more information on model capabilities, use the --help flag.")

if __name__ == "__main__":
    main()