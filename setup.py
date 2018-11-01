#!/usr/bin/env python

from setuptools import setup

setup(name='imutil',
    version='0.1.18',
    description='Swiss army knife for displaying images',
    long_description='Swiss army knife for converting and displaying numpy arrays, PyTorch tensors, and other image-like objects',
    author='Larry Neal',
    author_email='nealla@lwneal.com',
    url="https://github.com/lwneal/imutil",
    python_requires='>3',
    packages=[
        'imutil',
    ],
    install_requires=[
        'numpy',
        'Pillow',
        'scikit-image',
    ],
    package_data={
        'imutil': [
            'DejaVuSansMono.ttf',
        ]
    },
)
