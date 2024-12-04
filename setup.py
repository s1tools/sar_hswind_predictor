import os
from setuptools import setup, find_packages


# Get long description text from README.rst.
with open('README.md', 'r') as f:
    readme = f.read()

with open('VERSION.txt', 'r') as f:
    version = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name="s1_hswind_predictor",
    python_requires=">=3.9",
    version=version,
    author="Amine Benchaabane",
    author_email="abenchaabane@groupcls.com",
    description=("Estimate Hs wind Sea from official ESA L2 OCN product with deep learning model"),
    license="",
    url="https://github.com/s1tools/sar_hswind_predictor.git",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=requirements,
    long_description=readme,
)
