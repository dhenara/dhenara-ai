from setuptools import find_namespace_packages, setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="dhenara",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src", include=["dhenara.*"]),
    install_requires=[
        "httpx>=0.24.0",
        "requests>=2.25.1",
        "asgiref",
        "cryptography",
        "aiohttp",  # For async HTTP requests
        "pydantic>=2.0.0",
        # AI Models
        "openai",
        "google-genai",
        "anthropic",
        # Cloud dependecies are extra
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
        "azure": [
            "azure-ai-inference",
        ],
        "aws": [
            "boto3",
            "botocore",
        ],
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
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True,
)
