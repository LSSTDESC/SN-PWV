from __future__ import annotations

from egon.connectors import Output
from egon.nodes import Source

from ..models import ObservedCadence
from ..plasticc import PLaSTICC


class LoadPlasticcSims(Source):
    """Pipeline node for loading PLaSTICC data from disk

    Connectors:
        output: Tuple with the simulation params (``dict``) and cadence (``ObservedCadence``)
    """

    def __init__(self, cadence: str, model: int = 11, iter_lim: int = float('inf'), num_processes: int = 1) -> None:
        """Source node for loading PLaSTICC light-curves from disk

        Args:
            cadence: Cadence to use when simulating light-curves
            model: The PLaSTICC supernova model to load simulation for (Default is model 11 - Normal SNe)
            iter_lim: Exit after loading the given number of light-curves
            num_processes: Number of processes to allocate to the node
        """

        self.cadence = PLaSTICC(cadence, model)
        self.iter_lim = iter_lim

        # Node connectors
        self.output = Output()
        super().__init__(num_processes)

    def action(self) -> None:
        """Load PLaSTICC light-curves from disk"""

        for light_curve in self.cadence.iter_lc(iter_lim=self.iter_lim):
            self.output.put(ObservedCadence.from_plasticc(light_curve))
