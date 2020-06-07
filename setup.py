from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name="yapl",
    version="0.0.1",
    author="Durgesh",
    author_email="dkumar@ce.iitr.ac.in",
    description="yet another python library",
    url="https://github.com/orionpax00/yapl",
    packages=setuptools.find_packages(),
    license="APACHE",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
