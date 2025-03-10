import os
import sys
import time
import logging
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable

from .config import SeparationConfig, AnalysisConfig
from .core.stem_separator import StemSeparator
from .core.audio_processor import AudioProcessor
from .core.midi_converter import MidiConverter
from .utils.audio_utils import get_audio_info
from .utils.file_utils import ensure_directory, save_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('stem_splitter')

class StemSplitterApp:
    """Main application class for the Stem Splitter project."""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize the application.
        
        Args:
            config_file: Path to JSON configuration file (optional)
        """
        # Load config from file if provided
        if config_file:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                self.separation_config = SeparationConfig(**config_data.get('separation', {}))
                self.analysis_config = AnalysisConfig(**config_data.get('analysis', {}))
        else:
            # Use default configs
            self.separation_config = SeparationConfig()
            self.analysis_config = AnalysisConfig()
            
        # Initialize components
        self.separator = StemSeparator(self.separation_config)
        self.audio_processor = AudioProcessor(self.separation_config)
        self.midi_converter = MidiConverter(self.separation_config)
        
        # Task tracking
        self.active_tasks = {}
        self.task_counter = 0
        self.task_lock = threading.Lock()

    def separate_audio(self, 
                       input_file: Union[str, Path],
                       output_dir: Optional[Union[str, Path]] = None,
                       callback: Optional[Callable] = None) -> str:
        """
        Separate audio into stems.
        
        Args:
            input_file: Path to input audio file
            output_dir: Directory to save separated stems (optional)
            callback: Function to call with results when complete (optional)
            
        Returns:
            Task ID for tracking progress
        """
        input_file = Path(input_file)
        
        # Create task ID
        task_id = self._create_task("separation", input_file)
        
        # Run in background thread
        threading.Thread(
            target=self._run_separation,
            args=(task_id, input_file, output_dir, callback),
            daemon=True
        ).start()
        
        return task_id
    
    def _run_separation(self, 
                       task_id: str,
                       input_file: Path,
                       output_dir: Optional[Path],
                       callback: Optional[Callable]):
        """Run the separation task in a background thread."""
        try:
            # Update task status
            self._update_task_status(task_id, "running", "Loading audio...")
            
            # Create output directory if needed
            if output_dir is None:
                output_dir = input_file.parent / "stems" / input_file.stem
            
            output_dir = ensure_directory(output_dir)
            
            # Run separation
            self._update_task_status(task_id, "running", "Separating stems...")
            output_files = self.separator.separate(input_file, output_dir)
            
            # Create result metadata
            results = {
                "input_file": str(input_file),
                "output_dir": str(output_dir),
                "model": self.separation_config.model_name,
                "stems": {stem: str(path) for stem, path in output_files.items()},
                "timestamp": time.time()
            }
            
            # Save metadata
            metadata_path = output_dir / "metadata.json"
            save_metadata(results, metadata_path)
            results["metadata_file"] = str(metadata_path)
            
            # Mark task as complete
            self._update_task_status(task_id, "completed", "Separation complete", results)
            
            # Call callback if provided
            if callback:
                callback(task_id, "completed", results)
                
        except Exception as e:
            logger.error(f"Separation error: {e}", exc_info=True)
            error_message = str(e)
            self._update_task_status(task_id, "failed", error_message)
            
            if callback:
                callback(task_id, "failed", {"error": error_message})
    
    def analyze_audio(self,
                     input_file: Union[str, Path],
                     output_file: Optional[Union[str, Path]] = None,
                     config: Optional[AnalysisConfig] = None,
                     callback: Optional[Callable] = None) -> str:
        """
        Analyze audio to extract musical features.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to save results as JSON (optional)
            config: Analysis configuration (optional, uses default if not provided)
            callback: Function to call with results when complete (optional)
            
        Returns:
            Task ID for tracking progress
        """
        input_file = Path(input_file)
        
        if output_file:
            output_file = Path(output_file)
            
        # Use provided config or default
        analysis_config = config or self.analysis_config
        
        # Create task ID
        task_id = self._create_task("analysis", input_file)
        
        # Run in background thread
        threading.Thread(
            target=self._run_analysis,
            args=(task_id, input_file, output_file, analysis_config, callback),
            daemon=True
        ).start()
        
        return task_id
    
    def _run_analysis(self,
                     task_id: str,
                     input_file: Path,
                     output_file: Optional[Path],
                     config: AnalysisConfig,
                     callback: Optional[Callable]):
        """Run the analysis task in a background thread."""
        try:
            # Update task status
            self._update_task_status(task_id, "running", "Loading audio...")
            
            # Load audio
            audio, sr = self.audio_processor.load_audio(input_file)
            
            # Run analysis
            self._update_task_status(task_id, "running", "Analyzing audio...")
            results = self.audio_processor.analyze_audio(audio, sr, config)
            
            # Add file info
            results["file_info"] = get_audio_info(input_file)
            results["input_file"] = str(input_file)
            results["timestamp"] = time.time()
            
            # Save results if requested
            if output_file:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                save_metadata(results, output_file)
                results["output_file"] = str(output_file)
            
            # Mark task as complete
            self._update_task_status(task_id, "completed", "Analysis complete", results)
            
            # Call callback if provided
            if callback:
                callback(task_id, "completed", results)
                
        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            error_message = str(e)
            self._update_task_status(task_id, "failed", error_message)
            
            if callback:
                callback(task_id, "failed", {"error": error_message})
    
    def convert_to_midi(self,
                       input_file: Union[str, Path],
                       output_file: Optional[Union[str, Path]] = None,
                       instrument_type: str = "melodic",
                       callback: Optional[Callable] = None) -> str:
        """
        Convert audio to MIDI.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to save MIDI file (optional)
            instrument_type: Type of instrument ("melodic" or "percussive")
            callback: Function to call with results when complete (optional)
            
        Returns:
            Task ID for tracking progress
        """
        input_file = Path(input_file)
        
        # Set default output file if not provided
        if output_file is None:
            output_file = input_file.with_suffix('.mid')
        else:
            output_file = Path(output_file)
            
        # Create task ID
        task_id = self._create_task("midi_conversion", input_file)
        
        # Run in background thread
        threading.Thread(
            target=self._run_midi_conversion,
            args=(task_id, input_file, output_file, instrument_type, callback),
            daemon=True
        ).start()
        
        return task_id
    
    def _run_midi_conversion(self,
                            task_id: str,
                            input_file: Path,
                            output_file: Path,
                            instrument_type: str,
                            callback: Optional[Callable]):
        """Run the MIDI conversion task in a background thread."""
        try:
            # Update task status
            self._update_task_status(task_id, "running", "Loading audio...")
            
            # Load audio
            audio, sr = self.audio_processor.load_audio(input_file)
            
            # Convert to MIDI
            self._update_task_status(task_id, "running", "Converting to MIDI...")
            midi_file = self.midi_converter.audio_to_midi(
                audio, 
                sr, 
                output_file,
                instrument_type=instrument_type
            )
            
            # Create result metadata
            results = {
                "input_file": str(input_file),
                "output_file": str(midi_file),
                "instrument_type": instrument_type,
                "timestamp": time.time()
            }
            
            # Mark task as complete
            self._update_task_status(task_id, "completed", "Conversion complete", results)
            
            # Call callback if provided
            if callback:
                callback(task_id, "completed", results)
                
        except Exception as e:
            logger.error(f"MIDI conversion error: {e}", exc_info=True)
            error_message = str(e)
            self._update_task_status(task_id, "failed", error_message)
            
            if callback:
                callback(task_id, "failed", {"error": error_message})
    
    def get_task_status(self, task_id: str) -> Dict:
        """
        Get the status of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status dictionary
        """
        with self.task_lock:
            if task_id in self.active_tasks:
                return self.active_tasks[task_id].copy()
            else:
                return {"status": "not_found", "message": "Task not found"}
    
    def get_all_tasks(self) -> Dict[str, Dict]:
        """
        Get all tasks.
        
        Returns:
            Dictionary mapping task IDs to task status dictionaries
        """
        with self.task_lock:
            return {task_id: status.copy() for task_id, status in self.active_tasks.items()}
    
    def _create_task(self, task_type: str, input_file: Path) -> str:
        """Create and register a new task."""
        with self.task_lock:
            self.task_counter += 1
            task_id = f"{task_type}_{self.task_counter}_{int(time.time())}"
            
            self.active_tasks[task_id] = {
                "id": task_id,
                "type": task_type,
                "input_file": str(input_file),
                "status": "pending",
                "message": "Task created",
                "created_at": time.time(),
                "updated_at": time.time()
            }
            
        return task_id
    
    def _update_task_status(self, 
                           task_id: str, 
                           status: str, 
                           message: str,
                           results: Optional[Dict] = None):
        """Update the status of a task."""
        with self.task_lock:
            if task_id in self.active_tasks:
                self.active_tasks[task_id].update({
                    "status": status,
                    "message": message,
                    "updated_at": time.time()
                })
                
                if results:
                    self.active_tasks[task_id]["results"] = results
                    
                if status in ["completed", "failed"]:
                    self.active_tasks[task_id]["completed_at"] = time.time()


# Package entry point
def create_app(config_file: Optional[Union[str, Path]] = None) -> StemSplitterApp:
    """
    Create and initialize the application.
    
    Args:
        config_file: Path to configuration file (optional)
        
    Returns:
        Initialized StemSplitterApp instance
    """
    return StemSplitterApp(config_file)


# Usage example (if run directly)
if __name__ == "__main__":
    # Example usage as a script
    import argparse
    
    parser = argparse.ArgumentParser(description="Stem Splitter Application")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("command", choices=["separate", "analyze", "to-midi"], help="Command to run")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--output", "-o", help="Output file or directory")
    
    args = parser.parse_args()
    
    # Create app
    app = create_app(args.config)
    
    # Define callback to handle task completion
    def task_callback(task_id, status, results):
        if status == "completed":
            print(f"\nTask {task_id} completed successfully!")
            if "stems" in results:
                print("Separated stems:")
                for stem, path in results["stems"].items():
                    print(f"  - {stem}: {path}")
            elif "key" in results:
                print("Analysis results:")
                print(f"  - Key: {results.get('key', 'Unknown')}")
                print(f"  - Tempo: {results.get('tempo', 'Unknown')} BPM")
            elif "output_file" in results:
                print(f"Output saved to: {results['output_file']}")
        else:
            print(f"\nTask {task_id} failed: {results.get('error', 'Unknown error')}")
    
    # Run command
    if args.command == "separate":
        task_id = app.separate_audio(args.input, args.output, callback=task_callback)
    elif args.command == "analyze":
        task_id = app.analyze_audio(args.input, args.output, callback=task_callback)
    elif args.command == "to-midi":
        task_id = app.convert_to_midi(args.input, args.output, callback=task_callback)
    
    print(f"Started task {task_id}. Processing...")
    
    # Wait for task completion (in a real app, this would be handled by the event loop)
    while True:
        status = app.get_task_status(task_id)
        if status["status"] in ["completed", "failed"]:
            break
        time.sleep(0.5)