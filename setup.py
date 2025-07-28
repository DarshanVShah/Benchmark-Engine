"""
Setup script for the BenchmarkEngine framework.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="benchmark-engine",
    version="0.1.0",
    author="BenchmarkEngine Team",
    description="A flexible, extensible framework for model benchmarking with plugin-based architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/BenchmarkEngine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        # Core dependencies (minimal for framework)
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "examples": [
            # Dependencies for example plugins
            "transformers>=4.0.0",  # For HuggingFace adapter
            "torch>=1.8.0",         # For PyTorch support
            "tensorflow>=2.4.0",    # For TensorFlow support
        ],
    },
    entry_points={
        "console_scripts": [
            "benchmark-engine=core:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 