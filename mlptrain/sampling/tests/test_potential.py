import numpy as np
from mlptrain.potentials._base import MLPotential
from ase.calculators.calculator import Calculator
from ase.calculators.lj import LennardJones


class HarmonicPotential(Calculator):

    __test__ = False

    def get_potential_energy(self, atoms):

        r = atoms.get_distance(0, 1)

        return (r - 1)**2

    def get_forces(self, atoms):

        derivative = np.zeros((len(atoms), 3))

        r = atoms.get_distance(0, 1)

        x_dist, y_dist, z_dist = [atoms[0].position[j] - atoms[1].position[j]
                                  for j in range(3)]

        x_i, y_i, z_i = (x_dist / r), (y_dist / r), (z_dist / r)

        derivative[0] = [x_i, y_i, z_i]
        derivative[1] = [-x_i, -y_i, -z_i]

        force = -2 * derivative * (r - 1)

        return force

class DoubleWellPotential(Calculator):

    __test__ = False

    def __init__(self, k, a1, a2):
        Calculator.__init__(self)
        self.k = k
        self.a1 = a1
        self.a2 = a2

    def get_potential_energy(self, atoms):

        r = atoms.get_distance(0, 1)

        return self.k * (r - self.a1)**2 * (r - self.a2)**2

    def get_forces(self, atoms):

        derivative = np.zeros((len(atoms), 3))
        k = self.k
        a1 = self.a1
        a2 = self.a2

        r = atoms.get_distance(0, 1)
        if r < 10**(-7):
            r = 10**(-7)

        x_dist, y_dist, z_dist = [atoms[0].position[j] - atoms[1].position[j]
                                  for j in range(3)]

        x_i, y_i, z_i = (x_dist / r), (y_dist / r), (z_dist / r)

        derivative[0] = [x_i, y_i, z_i]
        derivative[1] = [-x_i, -y_i, -z_i]

        force = -2 * k * derivative * (r - a1) * (r - a2) * (2*r - a1 - a2)

        return force


class TestPotential(MLPotential):

    __test__ = False

    def __init__(self,
                 name: str,
                 calculator='harmonic',
                 system=None,
                 **kwargs):

        super().__init__(name=name, system=system)
        self.calculator = calculator.lower()
        self.kwargs = kwargs

    @property
    def ase_calculator(self):

        if self.calculator == 'harmonic':
            return HarmonicPotential()

        if self.calculator == 'doublewell':
            return DoubleWellPotential(**self.kwargs)

        if self.calculator == 'lj':
            return LennardJones(rc=2.5, r0=3.0)

        else:
            raise NotImplementedError(f'{self.calculator} is not implemented '
                                      f'as a test potential')

    def _train(self) -> None:
        """ABC for MLPotential required but unused in TestPotential"""

    def requires_atomic_energies(self) -> None:
        """ABC for MLPotential required but unused in TestPotential"""

    def requires_non_zero_box_size(self) -> None:
        """ABC for MLPotential required but unused in TestPotential"""
