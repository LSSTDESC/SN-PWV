"""Package installation logic"""

import re
from pathlib import Path
from typing import Union

from setuptools import find_packages, setup

# File paths for source code and supporting artifacts
root_dir = Path(__file__).resolve().parent
data_dir = root_dir / 'data'
readme_path = root_dir / 'README.md'
init_path = root_dir / 'snat_sim' / '__init__.py'
pkg_requirements_path = root_dir / 'requirements.txt'
doc_requirements_path = root_dir / 'docs' / 'requirements.txt'


def parse_requirements(path: Path) -> list:
    """Return a list of package dependencies

    Args:
        path: The path to parse requirements from

    Returns:
        A list of Python requirements
    """

    with path.open() as req_file:
        return req_file.read().splitlines()


def get_extras(**paths: Union[list, Path]) -> dict:
    """Return a dictionary defining package installation extras

    Returns:
        A dictionary definition of extra dependencies
    """

    extras = dict()
    for extra_name, extra_definition in paths.items():
        if isinstance(extra_definition, Path):
            extras[extra_name] = parse_requirements(extra_definition)

        else:
            extras[extra_name] = extra_definition

    return extras


def get_data_files(*dirs: Path) -> list:
    """Create a list of data files to include with the package

    Args:
        *Directories to include as package data

    Returns:
         A list of string paths for all files in the given directories
    """

    file_list = []
    for src_dir in dirs:
        file_list.extend(map(str, src_dir.rglob('*')))

    return file_list


def get_meta(val_name: str) -> str:
    """Return package metadata as defined in the init file"""

    init_text = init_path.read_text()
    regex = re.compile(f"__{val_name}__ = '(.*?)'")
    return regex.findall(init_text)[0]


setup(
    # Package Meta Data
    name='snat_sim',
    version=get_meta('version'),
    author=get_meta('author'),
    license=get_meta('license'),
    keywords='Supernova Atmosphere Chromatic Effects DESC',
    description='Simulation tools for atmospheric effects in photometric supernova observations.',
    long_description=readme_path.read_text(),
    url='https://lsstdesc.org/SN-PWV/',

    # Package executable, source code and data
    packages=find_packages() + ['data'],
    data_files=get_data_files(data_dir),
    include_package_data=True,
    entry_points="""
        [console_scripts]
        snat-sim=cli:Application.execute
    """,

    # Installation requirements
    python_requires='>=3.9',
    install_requires=parse_requirements(pkg_requirements_path),
    extras_require=get_extras(tets=['coverage', 'sndata'], docs=doc_requirements_path)
)
