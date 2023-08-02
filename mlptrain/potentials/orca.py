import os
from mlptrain.potentials._base import MLPotential
from ase.calculators.orca import ORCA

class OrcaPotential(MLPotential):

    def __init__(self,
                 name: str,
                 path: str,
                 charge: int,
                 mult: int,
                 orcasimpleinput: str,
                 orcablocks: str,
                 system=None):

        super().__init__(name=name, system=system)

        self.charge = charge
        self.mult = mult

        # Values for the calculator
        self.orcasimpleinput = orcasimpleinput
        self.orcablocks = orcablocks

        # ORCA v5.0.3
        self.path = path
        os.environ['ASE_ORCA_COMMAND'] = f'{self.path} PREFIX.inp > PREFIX.out'

    @property
    def ase_calculator(self):

        return ORCA(charge=self.charge,
                    mult=self.mult,
                    orcasimpleinput=self.orcasimpleinput,
                    orcablocks=self.orcablocks)

    def _train(self) -> None:
        """ABC for MLPotential required but unused in OrcaPotential"""

    def requires_atomic_energies(self) -> None:
        """ABC for MLPotential required but unused in OrcaPotential"""

    def requires_non_zero_box_size(self) -> None:
        """ABC for MLPotential required but unused in OrcaPotential"""
        