#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import re
from pathlib import Path

from setuptools import find_packages, setup

# Get list of requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

url = 'https://desc-sn-pwv.readthedocs.io/en/latest/'

# Get package version
init_path = Path(__file__).resolve().parent / 'snat_sim/__init__.py'
with init_path.open('r') as f:
    s = f.read()

versionRegExp = re.compile("__version__ = '(.*?)'")
__version__ = versionRegExp.findall(s)[0]

setup(name='snat_sim',
      version=__version__,
      packages=find_packages(),
      keywords='Supernova Atmsophere Chromatic Effects',
      description='Simulation tools for atmophseric effects in phtomoteric supernova observations',
      # long_description=long_description,
      # long_description_content_type='text/markdown',
      author='Daniel Perrefort',
      author_email='djperrefort@pitt.edu',
      url=url,
      license='GPL v3',
      python_requires='>=3.7',
      install_requires=requirements,
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      include_package_data=True
      )
