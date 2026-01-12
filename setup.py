from setuptools import setup, find_packages

setup(
    name="gcvit-classifier",
    version="0.1.0",
    description="GCViT image classification with reproducible training configs",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.12",
        "numpy",
        "opencv-python",
        "scikit-learn",
        "tqdm",
    ],
)
