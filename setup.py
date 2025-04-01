from setuptools import setup, find_packages

setup(
    name="tsilva-notebook-utils",
    version="0.1.9",
    packages=find_packages(),
    install_requires=[
        "ipython>=7.0.0",
        "opencv-python>=4.0.0",
        "imageio>=2.0.0",
        "numpy>=1.19.0",
    ],
    author="Tiago Silva",
    description="Utility functions for Jupyter/Colab notebooks",
    python_requires=">=3.6",
)
