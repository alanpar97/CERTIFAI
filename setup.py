"""Python setup.py for project_name package"""
import io
import os
from setuptools import find_packages, setup

req = ["numpy",
"pandas",
"scikit-learn",
"tqdm"
]

def read_requirements(path):
    return [
        line
        for line in read(path).splitlines()
        if line                                   # <-- guard against blanks
        and not line.startswith(("#", "-", "git+"))
    ]


setup(
    name="certifai",
    version="0.1.0",
    description="A python implementation of CERTIFAI framework for machine learning models' explainability",
    url="https://github.com/alanpar97/CERTIFAI/",
    author="Alan Paredes",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=req,
)
