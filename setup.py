from setuptools import setup, find_packages

setup(
    name="evo-hexapod-py",
    version="0.1.0",
    description="Evolutionary hexapod robot simulation",
    author="Anhar Risnumawan",
    author_email="risnumawan-anhar@ed.tmu.ac.jp",
    packages=find_packages(),
    install_requires=[
        "pybullet>=3.2.1",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
    ],
    python_requires=">=3.7",
)