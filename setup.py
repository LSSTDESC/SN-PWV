import re
from pathlib import Path

from setuptools import find_packages, setup

# Get list of requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Get package version
init_path = Path(__file__).resolve().parent / 'snat_sim/__init__.py'
with init_path.open('r') as f:
    source = f.read()

versionRegExp = re.compile("__version__ = '(.*?)'")
__version__ = versionRegExp.findall(source)[0]

setup(name='snat_sim',
      version=__version__,
      packages=find_packages(),
      keywords='Supernova Atmosphere Chromatic Effects',
      description='Simulation tools for atmospheric effects in photometric supernova observations.',
      author='The Dark Energy Science Collaboration (DESC)',
      author_email='djperrefort@pitt.edu',
      url='https://desc-sn-pwv.readthedocs.io/en/latest/',
      license='GPL v3',
      python_requires='>=3.8',
      install_requires=requirements,
      include_package_data=True
      )
