#!/usr/bin/env python
"""
CCL + Julia Bridge & WaveCaster System
Setup script for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = [
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ]

# Separate dev requirements
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "pytest-asyncio>=0.18.0",
    "black>=22.0.0",
    "mypy>=0.950",
    "flake8>=4.0.0",
    "pylint>=2.12.0",
    "pre-commit>=2.15.0",
    "isort>=5.10.0",
    "hypothesis>=6.36.0",
    "faker>=11.3.0",
]

# Optional dependencies
optional_requirements = {
    "full": [
        "requests>=2.26.0",
        "matplotlib>=3.4.0",
        "sounddevice>=0.4.0",
        "pycryptodome>=3.10.0",
    ],
    "web": [
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
    ],
    "ml": [
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "pandas>=1.3.0",
    ],
    "docs": [
        "sphinx>=4.3.0",
        "sphinx-rtd-theme>=1.0.0",
        "myst-parser>=0.17.0",
    ],
    "dev": dev_requirements,
}

# All optional dependencies combined
optional_requirements["all"] = [
    dep
    for deps in optional_requirements.values()
    for dep in deps
]

setup(
    name="ccl-wavecaster",
    version="0.2.0",
    author="CCL + WaveCaster Team",
    author_email="",
    description="CCL + Julia Bridge System with WaveCaster signal modulation engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/9x25dillon/Implementing_julia_bridge_al-ULS",
    project_urls={
        "Bug Tracker": "https://github.com/9x25dillon/Implementing_julia_bridge_al-ULS/issues",
        "Documentation": "https://github.com/9x25dillon/Implementing_julia_bridge_al-ULS/tree/main/docs",
        "Source Code": "https://github.com/9x25dillon/Implementing_julia_bridge_al-ULS",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    py_modules=["wavecaster", "ccl", "mock_al_uls_server", "ta_uls_trainer"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Networking",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=optional_requirements,
    entry_points={
        "console_scripts": [
            "wavecaster=wavecaster:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "signal-processing",
        "modulation",
        "llm",
        "julia",
        "categorical-coherence",
        "code-analysis",
        "machine-learning",
    ],
)
