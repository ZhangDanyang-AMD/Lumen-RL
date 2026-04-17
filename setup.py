from pathlib import Path

from setuptools import find_packages, setup

this_dir = Path(__file__).parent

setup(
    name="lumenrl",
    version="0.1.0",
    description="High-performance RL post-training framework for LLMs on AMD GPUs",
    long_description=(this_dir / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/ZhangDanyang-AMD/Lumen-RL",
    packages=find_packages(exclude=["tests*", "lumenrl-docs*"]),
    python_requires=">=3.10",
    install_requires=[
        "ray[default]>=2.9",
        "omegaconf>=2.3",
        "torch>=2.4",
        "transformers>=4.40",
        "datasets>=2.18",
        "accelerate>=0.28",
    ],
    extras_require={
        # NOTE: lumen and atom are installed from source (submodules or editable).
        # They are NOT on PyPI. Install them separately:
        #   pip install -e third_party/Lumen       (training engine)
        #   pip install -e third_party/verl         (RL orchestrator)
        # AITER/mori are built from Lumen's third_party/.
        "test": [
            "pytest>=8.0",
            "pytest-xdist>=3.5",
        ],
        "dev": [
            "pytest>=8.0",
            "pytest-xdist>=3.5",
            "ruff>=0.3",
            "mypy>=1.8",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
