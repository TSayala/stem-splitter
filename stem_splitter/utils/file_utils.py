import os
import shutil
import tempfile
import zipfile
import json
from pathlib import Path
from typing import List, Dict, Optional, Union

def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def clean_directory(directory: Union[str, Path], pattern: Optional[str] = None):
    """
    Remove files from a directory.
    
    Args:
        directory: Directory path
        pattern: File pattern to match (None for all files)
    """
    directory = Path(directory)
    if not directory.exists():
        return
    
    if pattern:
        for file in directory.glob(pattern):
            if file.is_file():
                file.unlink()
    else:
        for file in directory.iterdir():
            if file.is_file():
                file.unlink()

def create_temp_directory() -> Path:
    """
    Create a temporary directory.
    
    Returns:
        Path to temporary directory
    """
    temp_dir = Path(tempfile.mkdtemp())
    return temp_dir

def zip_directory(directory: Union[str, Path], 
                 output_file: Union[str, Path],
                 file_pattern: Optional[str] = None) -> Path:
    """
    Zip the contents of a directory.
    
    Args:
        directory: Directory to zip
        output_file: Output zip file
        file_pattern: Pattern to match files (None for all files)
        
    Returns:
        Path to zip file
    """
    directory = Path(directory)
    output_file = Path(output_file)
    
    if not output_file.suffix:
        output_file = output_file.with_suffix('.zip')
        
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        if file_pattern:
            files = list(directory.glob(file_pattern))
        else:
            files = [f for f in directory.rglob('*') if f.is_file()]
            
        for file in files:
            zipf.write(file, file.relative_to(directory))
            
    return output_file

def save_metadata(metadata: Dict, 
                 output_file: Union[str, Path]):
    """
    Save metadata to JSON file.
    
    Args:
        metadata: Dictionary with metadata
        output_file: Output JSON file
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_metadata(file_path: Union[str, Path]) -> Dict:
    """
    Load metadata from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary with metadata
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {}
    
    with open(file_path, 'r') as f:
        metadata = json.load(f)
        
    return metadata
