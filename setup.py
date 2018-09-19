#!/usr/bin/env python

from setuptools import setup

setup(name='imutil',
    version='0.1.13',
    description='Swiss army knife for displaying images',
    author='Larry Neal',
    author_email='nealla@lwneal.com',
    url="https://github.com/lwneal/imutil",
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
