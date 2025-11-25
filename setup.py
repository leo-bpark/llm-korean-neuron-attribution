# Author : Bumjin Park

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="LKN",
    packages=find_packages(),
    version="1.0.0",
    install_requires=requirements,
    python_requires=">=3.8",
    author="Bumjin Park",
    description="LLM Korean Neuron Attribution Tool - Measure and visualize MLP neuron contributions",
    long_description=long_description,
    long_description_content_type="text/markdown",
)