import os
import re
from pathlib import Path

from setuptools import find_namespace_packages, setup

version = None
# Read version without importing the package
with open(os.path.join("src/dhenara/ai", "__init__.py")) as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string")


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="dhenara",
    version=version,
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src", include=["dhenara.*"]),
    install_requires=[
        "httpx>=0.28.0",
        "requests>=2.32.1",
        "asgiref>=3.8.0",
        "cryptography>=44.0.0",
        "aiohttp>=3.11.0",
        "pydantic>=2.10.0",
        "openai>=1.65.0",
        "google-genai>=1.4.0",
        "anthropic>=0.49.0",
        # Cloud dependecies in extra
        "azure-ai-inference>=1.0.0b9",
        "boto3>=1.37.7",  # AWS
        "botocore>=1.37.7",  # AWS
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio",
            "pytest-cov",
            "black",
            "ruff",
            "add-trailing-comma",
        ],
        # TODO_FUTURE
        # "azure": [
        #    "azure-ai-inference>=1.0.0",
        # ],
        # "aws": [
        #    "boto3>=1.37.7",
        #    "botocore>=1.37.7",
        # ],
    },
    python_requires=">=3.10",
    description="Dhenara Package for Multi Provider AI-Model API calls",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Dhenara",
    author_email="support@dhenara.com",
    url="https://github.com/dhenara/dhenara",
    license="MIT",  # Replace with your actual license
    keywords="ai, llm, machine learning, language models",
    project_urls={
        "Homepage": "https://dhenara.com",
        "Documentation": "https://docs.dhenara.com/",
        "Bug Reports": "https://github.com/dhenara/dhenara/issues",
        "Source Code": "https://github.com/dhenara/dhenara",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True,
)
