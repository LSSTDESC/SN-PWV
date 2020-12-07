Command Line Interface
======================

A command line interface is provided for running the fitting and simulation
pipeline. This is particularly useful when running multiple pipeline instances
for different configuration arguments. An outline of the command line arguments
and their default values is provided below.

.. argparse::
   :filename: ../scripts/fitting_cli.py
   :func: create_cli_parser
   :prog: fitting_cli.py
