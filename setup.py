"""Setup script for rainbow-tensor package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rainbow-tensor",
    version="0.1.0",
    author="Niox1337",
    description="A Python package for IPython/Jupyter notebooks with numpy support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Niox1337/rainbow-tensor",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Framework :: IPython",
        "Framework :: Jupyter",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "ipython>=7.0.0",
    ],
)
