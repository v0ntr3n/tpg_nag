"""Setup script for DualGuide-SDXL package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="dualguide-sdxl",
    version="1.0.0",
    author="DualGuide-SDXL Contributors",
    description="Dual Guidance for Stable Diffusion XL: Combining Token Perturbation Guidance (TPG) and Normalized Attention Guidance (NAG) for enhanced image generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DualGuide-SDXL",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "diffusers>=0.25.0",
        "accelerate>=0.20.0",
        "Pillow>=9.5.0",
        "safetensors>=0.3.1",
        "omegaconf>=2.3.0",
    ],
    extras_require={
        "xformers": ["xformers>=0.0.20"],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add CLI scripts here if needed
        ],
    },
)
