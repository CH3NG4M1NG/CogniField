"""
setup.py
=========
CogniField v10 — Python Package Setup

Install (development):
    pip install -e .

Install (release):
    pip install cognifield
"""

from setuptools import setup, find_packages
import os

HERE = os.path.dirname(__file__)
README = open(os.path.join(HERE, "README.md"), encoding="utf-8").read()

setup(
    name="cognifield",
    version="11.0.0",
    author="CogniField Project",
    description=(
        "Adaptive Multi-Agent Cognitive Intelligence Framework — "
        "reason, decide, and collaborate without deep learning"
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/cognifield",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "Pillow>=9.0",
    ],
    extras_require={
        "api":  ["flask>=2.0"],
        "dev":  ["flask>=2.0", "pytest>=7.0"],
        "all":  ["flask>=2.0", "pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "cognifield=cognifield.cli.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "cognitive-ai", "multi-agent", "reasoning", "decision-making",
        "belief-system", "uncertainty", "llm-integration", "consensus",
    ],
    include_package_data=True,
    zip_safe=False,
)
