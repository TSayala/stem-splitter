from setuptools import setup, find_packages

setup(
    name="stem_splitter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "mido>=1.2.10",
    ],
    extras_require={
        "demucs": ["demucs>=4.0.0"],
        "all": ["demucs>=4.0.0"]
    },
    entry_points={
        "console_scripts": [
            "stem-splitter=stem_splitter.cli:main",
        ],
    },
    author="Tanay Sayala",
    description="A modular audio stem separation tool",
    keywords="audio, stem separation, music, midi",
    python_requires=">=3.8",
)