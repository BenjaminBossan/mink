from setuptools import setup
from setuptools import find_packages


with open('VERSION', 'r') as f:
    version = f.read().strip()


setup(
    name='mink',
    version=version,
    author='Benjamin Bossan',
    packages=find_packages(),
)
