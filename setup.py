#!/usr/bin/env python
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='OXEMU',
    version='0.0.1',
    description='OXEMU',
    url='https://github.com/Harry45/OXEMU',
    author='Arrykrishna Mootoovaloo',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    packages=['oxemu'],
    install_requires=['pandas', 'numpy', 'torch', 'matplotlib'],
    python_requires='>=3.6',
)
