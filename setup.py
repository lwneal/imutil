#!/usr/bin/env python

from setuptools import setup

setup(name='imutil',
        version='0.1.0',
        description='Swiss army knife for displaying images',
        author='Larry Neal',
        author_email='nealla@lwneal.com',
        packages=[
            'imutil',
        ],
        scripts=['scripts/gnomehat_server',
                 'scripts/gnomehat_worker',
                 'scripts/gnomehat_run',
                 'scripts/gnomehat_cleanup',
                 'scripts/gnomehat'],
      install_requires=[
          "numpy",
          "Pillow",
      ],
)
