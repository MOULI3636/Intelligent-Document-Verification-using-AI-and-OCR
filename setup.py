"""
DocVision AI: Setuptools Package Configuration.

Allows installing docvision-ai as an editable package ('pip install -e .')
or building distribution wheels for production deployment.
"""

from setuptools import setup, find_packages
import os

# Read long description from README.md
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

if __name__ == "__main__":
    setup(
    name="docvision-ai",
    version="1.0.0",
    description="Intelligent Document Verification, Multi-Engine OCR Benchmarking & Fraud Detection Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DocVision AI Research Team",
    author_email="research@docvision.ai",
    url="https://github.com/docvision-ai/docvision-ai",
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "Pillow>=9.5.0",
        "albumentations>=1.3.1",
        "transformers>=4.30.0",
        "easyocr>=1.7.0",
        "torchmetrics>=1.0.0",
        "jiwer>=3.0.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "streamlit>=1.25.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "docvision-train=train:main",
            "docvision-eval=evaluate:main",
            "docvision-app=app:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
