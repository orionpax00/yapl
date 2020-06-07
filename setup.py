from setuptools import setup, find_packages

import yapl 

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name="yapl",
    packages = find_packages(),
    version=yapl.__version__,
    author="Durgesh",
    author_email="dkumar@ce.iitr.ac.in",
    description="yet another python library",
    url="https://github.com/orionpax00/yapl",
    license="APACHE",
    install_requires=[            # I get to this in a second
        'yacs==0.1.6',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
