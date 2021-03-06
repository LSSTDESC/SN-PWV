import os
import re
from pathlib import Path

from setuptools import find_packages, setup

root_dir = Path(__file__).resolve().parent
data_dir = root_dir / 'data'
init_path = root_dir / 'snat_sim/__init__.py'

# Get list of package requirements
with open('requirements.txt') as f:
    core_requirements = f.read().splitlines()

# Extra requirements for building docs
with open('docs/requirements.txt') as f:
    doc_requirements = f.read().splitlines()

# Get package version
with init_path.open('r') as f:
    __version__ = re.compile("__version__ = '(.*?)'").findall(f.read())[0]


def gen_data_files(*dirs):
    """Create a list of data files to include with the package"""

    results = []
    for src_dir in dirs:
        for root, dirs, files in os.walk(src_dir):
            results.append((root, map(lambda f: root + '/' + f, files)))

    return results


setup(name='snat_sim',
      version=__version__,
      packages=find_packages() + ['data'],
      data_files=gen_data_files('data'),
      keywords='Supernova Atmosphere Chromatic Effects',
      description='Simulation tools for atmospheric effects in photometric supernova observations.',
      author='The Dark Energy Science Collaboration (DESC)',
      author_email='djperrefort@pitt.edu',
      url='https://desc-sn-pwv.readthedocs.io/en/latest/',
      license='GPL v3',
      python_requires='>=3.8',
      install_requires=core_requirements,
      include_package_data=True,
      extras_require={
          'tests': ['sndata'],
          'docs': doc_requirements
      }
      )
