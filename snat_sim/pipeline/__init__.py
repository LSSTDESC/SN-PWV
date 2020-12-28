"""The ``pipeline`` module defines the ``FittingPipeline`` class, which
is built to provide a parallelized approach to simulating and fitting
light-curves with atmospheric effects.

Usage Example
-------------

Instances of the ``FittingPipeline`` class can be run synchronously
(by calling ``FittingPipeline.run``) or asynchronously (with
``FittingPipeline.run_async``). Here we demonstrate running a pipeline
synchronously.

.. code-block:: python

   >>> from snat_sim.pipeline import FittingPipeline

   >>> print('Instantiating pipeline...')
   >>> pipeline = FittingPipeline(
   >>>     cadence='alt_sched',
   >>>     sim_model=sn_model_sim,
   >>>     fit_model=sn_model_fit,
   >>>     vparams=['x0', 'x1', 'c'],
   >>>     out_path='./demo_out_path.csv',
   >>>     pool_size=6
   >>> )

   >>> print('I/O Processes: 2')
   >>> print('Simulation Processes:', pipeline.simulation_pool)
   >>> print('Fitting Processes:', pipeline.fitting_pool)
   >>> pipeline.run()

Module Docs
-----------
"""

from pipeline import FittingPipeline
