"""The ``pipeline`` module defines the ``FittingPipeline`` class, which
is built to provide a parallelized approach to simulating and fitting
light-curves with atmospheric effects.

Usage Example
-------------

Instances of the ``FittingPipeline`` class can be run synchronously
(by calling ``FittingPipeline.run``) or asynchronously (with
``FittingPipeline.run_async``). A pipeline instance can be created as
follows:

.. doctest:: python

   >>> from snat_sim.models import SNModel
   >>> from snat_sim.pipeline import FittingPipeline

   >>> pipeline = FittingPipeline(
   ...     cadence='alt_sched',
   ...     sim_model=SNModel('salt2'),
   ...     fit_model=SNModel('salt2'),
   ...     vparams=['x0', 'x1', 'c'],
   ...     out_path='./demo_out_path.csv',
   ...     fitting_pool=6,
   ...     simulation_pool=3
   ... )

Module Docs
-----------
"""

from . import nodes
from .pipelines import FittingPipeline

FittingPipeline.__module__ = __name__
