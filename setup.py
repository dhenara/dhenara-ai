from setuptools import find_namespace_packages, setup

setup(
    name="dhenara-ai",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src", include=["dhenara.*"]),
    install_requires=[
        "httpx>=0.24.0",
        "requests>=2.25.1",
        "pydantic>=2.0.0",
    ],
    python_requires=">=3.10",
    description="Dhenara Inc AI Model Clients",
    author="Dhenara",
    author_email="support@dhenara.com",
    url="https://github.com/dhenara/dhen-ai",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
)
