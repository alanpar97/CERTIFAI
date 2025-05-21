"""Python setup.py for certifai package"""
from setuptools import setup

req = ["numpy",
"pandas",
"scikit-learn",
"tqdm"
]

setup(
    name="certifai",
    version="0.1.0",
    description="A python implementation of CERTIFAI framework for machine learning models' explainability",
    url="https://github.com/alanpar97/CERTIFAI/",
    author="Alan Paredes",
    packages=["certifai"],
    install_requires=req,
)
