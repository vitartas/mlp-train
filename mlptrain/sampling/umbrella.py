import os
import re
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from scipy.stats import norm
from typing import Optional, Callable, List, Sequence, Tuple
from multiprocessing import Pool
from copy import deepcopy
from ase.io.trajectory import Trajectory as ASETrajectory
from ase.io import write as ase_write
from mlptrain.sampling.bias import Bias
from mlptrain.sampling.reaction_coord import DummyCoordinate
from mlptrain.configurations import ConfigurationSet
from mlptrain.sampling.md import run_mlp_md
from mlptrain.config import Config
from mlptrain.log import logger
from mlptrain.utils import (
    move_files,
    unique_name,
    convert_ase_energy,
    convert_exponents
)


class _Window:
    """Contains the attributes belonging to an US window used for WHAM or UI"""

    def __init__(self,
                 obs_zetas: np.ndarray,
                 bias: 'mlptrain.Bias'):
        """
        Umbrella Window

        -----------------------------------------------------------------------
        Arguments:

            obs_zetas: Values of the sampled (observed) reaction coordinate
                       ζ_i for this window (i)

            bias: Bias function, containing a reference value of ζ in this
                  window and its associated spring constant
        """
        self._bias = bias
        self._obs_zetas = obs_zetas

        self._gaussian_pdf:     Optional[_FittedGaussian] = None
        self._gaussian_plotted: Optional[_FittedGaussian] = None

        self.bin_edges:     Optional[np.ndarray] = None
        self.bias_energies: Optional[np.ndarray] = None
        self.hist:          Optional[np.ndarray] = None

        # Weight used in umbrella integration
        self.p_ui: Optional[float] = None

        self.free_energy = 0.0

    def bin(self) -> None:
        """Bin the observed reaction coordinates in this window into an a set
        of bins, defined by the array of bin centres"""

        if self.bin_centres is None:
            raise TypeError('Cannot bin with undefined bin centres')

        self.hist, _ = np.histogram(self._obs_zetas, bins=self.bin_edges)

        self.bias_energies = ((self._bias.kappa/2)
                              * (self.bin_centres - self._bias.ref)**2)

        return None

    def set_bin_edges(self,
                      outer_zeta_refs: Tuple[float, float],
                      n_bins:          int
                      ) -> None:
        """
        Compute and store an array with zeta values at bin edges

        -----------------------------------------------------------------------
        Arguments:

            outer_zeta_refs: (Tuple) Left-most and right-most reaction
                                     reaction coordinate values

            n_bins: (int) number of bins to use in the histogram
        """

        lmost_edge, rmost_edge = outer_zeta_refs
        _bin_edges = np.linspace(lmost_edge, rmost_edge, num=n_bins+1)

        self.bin_edges = _bin_edges
        return None

    @property
    def bin_centres(self) -> Optional[np.ndarray]:
        """Array of zeta values at bin centres"""

        if self.bin_edges is None:
            return None

        _edges = self.bin_edges

        return (_edges[1:] + _edges[:-1]) / 2

    @property
    def gaussian_pdf(self) -> '_FittedGaussian':
        """Fitted gaussian as a probability density function"""

        if self._gaussian_pdf is None:
            self._fit_gaussian(normalised=True)

        return self._gaussian_pdf

    @property
    def gaussian_plotted(self) -> '_FittedGaussian':
        """Gaussian which was plotted during umbrella sampling simulation"""

        if self._gaussian_plotted is None:
            raise TypeError('No plotted gaussian is stored in the window, '
                            'make sure to run umbrella sampling first')

        return self._gaussian_plotted

    @property
    def n(self) -> int:
        """Number of samples in this window"""
        if self.hist is None:
            raise ValueError('Cannot determine the number of samples - '
                             'window has not been binned')

        return int(np.sum(self.hist))

    def dA_dq(self,
              zetas: np.ndarray,
              beta:  float
              ) -> np.ndarray:
        """
        PMF from a single window

        -----------------------------------------------------------------------
        Arguments:

            zetas: (np.ndarray) Discretised reaction coordinate

            beta: (float) β = 1 / (k_B T)

        Returns:

            (np.ndarray): PMF from a single window
        """

        if self.gaussian_pdf is None:
            raise TypeError('Cannot estimate PMF if the window does not '
                            'contain a fitted probability density function')

        mean_zeta_b = self.gaussian_pdf.mean
        std_zeta_b = self.gaussian_pdf.std
        kappa = self._bias.kappa
        zeta_ref = self.zeta_ref

        # Equation 8.8.21 from Tuckerman, p. 344
        _dA_dq = ((1.0 / beta) * (zetas - mean_zeta_b) / (std_zeta_b**2)
                  - kappa * (zetas - zeta_ref))

        return _dA_dq

    def var_dA_dq(self,
                  zetas:     np.ndarray,
                  beta:      float,
                  blocksize: int
                  ) -> np.ndarray:
        """
        Variance of PMF from a single window [1]

        [1] Kastner, J., & Thiel, W. (2006). Journal of Chemical Physics,
            124(23), 234106. https://doi.org/10.1063/1.2206775

        -----------------------------------------------------------------------
        Arguments:

            zetas: (np.ndarray) Discretised reaction coordinate

            beta: (float) β = 1 / (k_B T)

            blocksize: (int) Block size used when computing the variance

        Returns:

            (np.ndarray): PMF from a single window
        """

        obs_zetas = self._obs_zetas
        mean_q = self.gaussian_pdf.mean
        std_q = self.gaussian_pdf.std

        n_blocks = len(obs_zetas) // blocksize
        block_means = []

        for block_idx in range(n_blocks):
            start_idx = blocksize * block_idx
            end_idx = start_idx + blocksize

            block_mean = np.mean(obs_zetas[start_idx:end_idx])
            block_means.append(block_mean)

        block_std_q = np.std(block_means, ddof=1)

        var_mean_q = (1 / n_blocks) * block_std_q**2
        var_var_q = (2 / n_blocks) * block_std_q**4

        # [1] Equation 5
        var_dA_dq = ((1 / (beta**2 * std_q**4))
                     * (var_mean_q
                        + ((zetas - mean_q)**2 * var_var_q) / std_q**4))

        return var_dA_dq

    @property
    def zeta_ref(self) -> float:
        """
        ζ_ref for this window

        -----------------------------------------------------------------------
        Returns:
            (float):
        """
        return self._bias.ref

    def block_analysis(self, label: Optional[str] = None) -> None:
        """
        Split the trajectory into blocks and compute the standard error of the
        mean zeta value over the blocks. Repeat for different block sizes and
        plot the results

        -----------------------------------------------------------------------
        Arguments:

            label: (str) String distinguishing a particular window, useful if
                         block analysis is performed to all windows at once
        """

        logger.info('Performing block analysis'
                    f'{f" for window {label}" if label is not None else ""}')

        min_n_blocks = 10
        min_blocksize = 10
        blocksize_interval = 5
        max_blocksize = len(self._obs_zetas) // min_n_blocks

        if max_blocksize < min_blocksize:
            raise ValueError('The simulation is too short to perform '
                             'block analysis')

        blocksizes = list(range(min_blocksize, max_blocksize + 1,
                                blocksize_interval))

        # Insert blocksize of 1
        blocksizes.insert(0, 1)

        std_errs = []
        for blocksize in blocksizes:
            n_blocks = len(self._obs_zetas) // blocksize
            block_means = []

            for block_idx in range(n_blocks):
                start_idx = blocksize * block_idx
                end_idx = blocksize * (block_idx + 1)
                block_mean = np.mean(self._obs_zetas[start_idx:end_idx])
                block_means.append(block_mean)

            std_err = (1 / np.sqrt(n_blocks)) * np.std(block_means, ddof=1)
            std_errs.append(std_err)

        self._plot_block_analysis(blocksizes=blocksizes,
                                  std_errs=std_errs,
                                  label=label)
        return None

    @staticmethod
    def _plot_block_analysis(blocksizes: List,
                             std_errs:   List,
                             label:      Optional[str] = None) -> None:
        """
        Plot block averaging analysis of the window

        -----------------------------------------------------------------------
        Arguments:

            label: (str) String distinguishing a particular window, useful if
                         block analysis is performed to all windows at once
        """

        fig, ax = plt.subplots()
        ax.plot(blocksizes, std_errs, color='k')

        ax.set_xlabel('Block size')
        ax.set_ylabel(r'$\sigma_{\mu_{\zeta}}$ / Å')

        fig.tight_layout()

        if label is None:
            figname = 'block_analysis_window.pdf'

        else:
            figname = f'block_analysis_window_{label}.pdf'

        fig.savefig(figname)
        plt.close(fig)

        return None

    @classmethod
    def from_file(cls, filename: str) -> '_Window':
        """
        Load a window from a saved file

        -----------------------------------------------------------------------
        Arguments:
            filename:

        Returns:
            (mlptrain.sampling.umbrella._Window):
        """
        file_lines = open(filename, 'r', errors='ignore').readlines()
        header_line = file_lines.pop(0)            # Pop the first line

        ref_zeta = float(header_line.split()[0])   # Å
        kappa = float(header_line.split()[1])      # eV / Å^2

        obs_zeta = [float(line.split()[0]) for line in file_lines
                    if len(line.split()) > 0]

        window = cls(obs_zetas=np.array(obs_zeta),
                     bias=Bias(zeta_func=DummyCoordinate(),
                               kappa=kappa,
                               reference=ref_zeta))

        return window

    def save(self, filename: str) -> None:
        """
        Save this window to a file

        -----------------------------------------------------------------------
        Arguments:
            filename:
        """
        with open(filename, 'w') as out_file:
            print(self._bias.ref, self._bias.kappa, file=out_file)

            for zeta in self._obs_zetas:
                print(zeta, file=out_file)

        return None

    def _fit_gaussian(self, normalised) -> None:
        """Fit a gaussian to a histogram of data"""

        gaussian = _FittedGaussian()

        a_0, mu_0, sigma_0 = (np.max(self.hist),
                              np.average(self._obs_zetas),
                              float(np.std(self._obs_zetas)))

        try:
            gaussian.params, _ = curve_fit(gaussian.value,
                                           self.bin_centres,
                                           self.hist,
                                           p0=[1.0, 1.0, 1.0],  # init guess
                                           maxfev=10000)

        except RuntimeError:
            logger.warning('Could not fit gaussian to a histogram, using '
                           'parameters obtained without fitting instead')

            gaussian.params = a_0, mu_0, sigma_0

        if normalised:
            gaussian.params = 1, *gaussian.params[1:]

        self._gaussian_pdf = gaussian
        return None

    def _plot_gaussian(self, hist, bin_centres) -> None:
        """Fit a Gaussian to a histogram of data and plot the result"""

        gaussian = _FittedGaussian()

        try:
            gaussian.params, _ = curve_fit(gaussian.value, bin_centres, hist,
                                           p0=[1.0, 1.0, 1.0],
                                           maxfev=10000)

            if np.min(np.abs(bin_centres - gaussian.mean)) > 1.0:
                raise RuntimeError('Gaussian mean was not within the 1 Å of '
                                   'the ζ range')

        except RuntimeError:
            logger.error('Failed to fit a gaussian to this data')
            return None

        # Plot the fitted line in the same color as the histogram
        color = plt.gca().lines[-1].get_color()
        zetas = np.linspace(min(bin_centres), max(bin_centres), num=500)

        plt.plot(zetas, gaussian(zetas), c=color)

        self._gaussian_plotted = gaussian
        return None

    def plot(self,
             min_zeta:      float,
             max_zeta:      float,
             plot_gaussian: bool = True) -> None:
        """
        Plot this window along with a fitted Gaussian function if possible

        -----------------------------------------------------------------------
        Arguments:
            min_zeta:

            max_zeta:

            plot_gaussian:
        """
        hist, bin_edges = np.histogram(self._obs_zetas,
                                       density=False,
                                       bins=np.linspace(min_zeta - 0.1*abs(min_zeta),
                                                        max_zeta + 0.1*abs(max_zeta),
                                                        num=400))

        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2
        plt.plot(bin_centres, hist, alpha=0.1)

        if plot_gaussian:
            self._plot_gaussian(hist, bin_centres)

        plt.xlabel('Reaction coordinate / Å')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('fitted_data.pdf')

        return None


class UmbrellaSampling:
    """
    Umbrella sampling class for generating pulling simulation, running
    umbrella sampling windows and running WHAM or umbrella integration.
    """

    def __init__(self,
                 zeta_func: 'mlptrain.sampling.reaction_coord.ReactionCoordinate',
                 kappa:      float,
                 temp:       Optional[float] = None):
        """
        Umbrella sampling to predict free energy using an mlp under a harmonic
        bias:

            ω = κ/2 (ζ(r) - ζ_ref)^2

        where ω is the bias in a particular window, ζ a function that takes in
        nuclear positions (r) and returns a scalar and ζ_ref the reference
        value of the reaction coordinate in that particular window.

        -----------------------------------------------------------------------
        Arguments:

            zeta_func: Reaction coordinate, as the function of atomic positions

            kappa: Value of the spring constant, κ, used in umbrella sampling
        """

        self.kappa:             float = kappa                        # eV Å^-2
        self.zeta_func:         Callable = zeta_func                 # ζ(r)
        self.temp:              Optional[float] = temp               # K

        self.windows:           List[_Window] = []

    @staticmethod
    def _best_init_frame(bias, traj):
        """Find the frames whose bias value is the lowest, i.e. has the
        closest reaction coordinate to the desired"""
        if len(traj) == 0:
            raise RuntimeError('Cannot determine the best frame from a '
                               'trajectory with length zero')

        min_e_idx = np.argmin([bias(frame.ase_atoms) for frame in traj])

        return traj[min_e_idx]

    def _reference_values(self, traj, num, init_ref, final_ref) -> np.ndarray:
        """Set the values of the reference for each window, if the
        initial and final reference values of the reaction coordinate are None
        then use the values in the start or end of the trajectory"""

        if init_ref is None:
            init_ref = self.zeta_func(traj[0])

        if final_ref is None:
            final_ref = self.zeta_func(traj[-1])

        return np.linspace(init_ref, final_ref, num)

    def _no_ok_frame_in(self, traj, ref) -> bool:
        """
        Does there exist a good reference structure in a trajectory?
        defined by the minimum absolute difference in the reaction coordinate
        (ζ) observed in the trajectory and the target value

        -----------------------------------------------------------------------
        Arguments:
            traj: A trajectory containing structures
            ref: ζ_ref

        Returns:
            (bool):
        """
        return np.min(np.abs(self.zeta_func(traj) - ref)) > 0.5

    def run_umbrella_sampling(self,
                              traj:    'mlptrain.ConfigurationSet',
                              mlp:     'mlptrain.potentials._base.MLPotential',
                              temp:        float,
                              interval:    int,
                              dt:          float,
                              init_ref:    Optional[float] = None,
                              final_ref:   Optional[float] = None,
                              n_windows:   int = 10,
                              save_sep:    bool = True,
                              all_to_xyz:  bool = False,
                              **kwargs
                              ) -> None:
        """
        Run umbrella sampling across n_windows, fitting Gaussians to the
        sampled values of the reaction coordinate.

        *NOTE* will leave a dangling plt.figure open

        -----------------------------------------------------------------------
        Arguments:
            traj: Trajectory from which to initialise the umbrella over, e.g.
                  a 'pulling' trajectory that has sufficient sampling of a
                  range f reaction coordinates

            mlp: Machine learnt potential

            temp: Temperature in K to initialise velocities and to run NVT MD.
                  Must be positive
            
            interval: (int) Interval between saving the geometry
            
            dt: (float) Time-step in fs
            
            init_ref: (float | None) Value of reaction coordinate in Å for
                                     first window
            
            final_ref: (float | None) Value of reaction coordinate in Å for
                                      first window
            
            n_windows: (int) Number of windows to run in the umbrella sampling

            save_sep: (bool) If True saves trajectories of each window
                             separately as .xyz files

            all_to_xyz: (bool) If True all .traj trajectory files are saved as
                              .xyz files (when using save_fs, save_ps, save_ns)

        -------------------
        Keyword Arguments:

            {fs, ps, ns}: Simulation time in some units

            {save_fs, save_ps, save_ns}: Trajectory saving interval
                                         in some units

            constraints: (List) List of ASE constraints to use in the dynamics
                                e.g. [ase.constraints.Hookean(a1, a2, k, rt)]
        """

        start_umbrella = time.perf_counter()

        if temp <= 0:
            raise ValueError('Temperature must be positive and non-zero for '
                             'umbrella sampling')

        self.temp = temp
        zeta_refs = self._reference_values(traj, n_windows, init_ref, final_ref)

        # window_process.get() --> window_traj
        window_processes, window_trajs, biases = [], [], []

        n_processes = min(n_windows, Config.n_cores)
        logger.info(f'Running Umbrella Sampling with {n_windows} window(s), '
                    f'{n_processes} window(s) are run in parallel')

        with Pool(processes=n_processes) as pool:

            for idx, ref in enumerate(zeta_refs):

                # Without copy kwargs is overwritten at every iteration
                kwargs_single = deepcopy(kwargs)
                kwargs_single['idx'] = idx + 1
                kwargs_single['ref'] = ref

                bias = Bias(self.zeta_func, kappa=self.kappa, reference=ref)

                if self._no_ok_frame_in(traj, ref):
                    # Takes the trajectory of the previous window, .get() blocks
                    # the main process until the previous window finishes
                    _traj = window_processes[idx-1].get()
                else:
                    _traj = traj

                init_frame = self._best_init_frame(bias, _traj)

                window_process = pool.apply_async(func=self._run_individual_window,
                                                  args=(init_frame,
                                                        mlp,
                                                        temp,
                                                        interval,
                                                        dt,
                                                        bias),
                                                  kwds=kwargs_single)
                window_processes.append(window_process)
                biases.append(bias)

            for window_process, bias in zip(window_processes, biases):

                window_traj = window_process.get()
                window = _Window(obs_zetas=self.zeta_func(window_traj),
                                 bias=bias)
                window.plot(min_zeta=min(zeta_refs),
                            max_zeta=max(zeta_refs),
                            plot_gaussian=True)

                self.windows.append(window)
                window_trajs.append(window_traj)

        # Move .traj files into 'trajectories' folder and compute .xyz files
        self._move_and_save_files(window_trajs, save_sep, all_to_xyz)

        finish_umbrella = time.perf_counter()
        logger.info('Umbrella sampling done in '
                    f'{(finish_umbrella - start_umbrella) / 60:.1f} m')

        return None

    def _run_individual_window(self, frame, mlp, temp, interval, dt, bias,
                               **kwargs):
        """Runs an individual umbrella sampling window"""

        logger.info(f'Running US window {kwargs["idx"]} with '
                    f'ζ_ref={kwargs["ref"]:.2f} Å '
                    f'and κ = {self.kappa:.3f} eV / Å^2')

        kwargs['n_cores'] = 1

        traj = run_mlp_md(configuration=frame,
                          mlp=mlp,
                          temp=temp,
                          dt=dt,
                          interval=interval,
                          bias=bias,
                          kept_substrings=['.traj'],
                          **kwargs)

        return traj

    @staticmethod
    def _move_and_save_files(window_trajs, save_sep, all_to_xyz) -> None:
        """Saves window trajectories, moves them into trajectories folder and
        computes .xyz files"""

        move_files([r'trajectory_\d+\.traj', r'trajectory_\d+_\w+\.traj'],
                   dst_folder='trajectories',
                   regex=True)

        os.chdir('trajectories')

        if save_sep:
            for idx, traj in enumerate(window_trajs, start=1):
                traj.save(filename=f'window_{idx}.xyz')

        else:
            combined_traj = ConfigurationSet()
            for window_traj in window_trajs:
                combined_traj += window_traj

            combined_traj.save(filename='combined_trajectory.xyz')

        if all_to_xyz:
            pattern = re.compile(r'trajectory_\d+_\w+\.traj')

            for filename in os.listdir():
                if re.search(pattern, filename) is not None:
                    basename = filename[:-5]
                    idx = basename.split('_')[1]
                    sim_time = basename.split('_')[2]

                    ase_traj = ASETrajectory(filename)
                    ase_write(f'window_{idx}_{sim_time}.xyz', ase_traj)

        os.chdir('..')

        return None

    def free_energies(self, prob_dist) -> np.ndarray:
        """
        Free energies at each point along the profile, eqn. 8.6.5 in Tuckerman

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray): A(ζ)
        """
        return - (1.0 / self.beta) * np.log(prob_dist)

    @property
    def zeta_refs(self) -> Optional[np.ndarray]:
        """
        Array of ζ_ref for each window

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray(float) | None):
        """
        if len(self.windows) == 0:
            return None

        return np.array([w_k.zeta_ref for w_k in self.windows])

    @property
    def beta(self) -> float:
        """
        β = 1 / (k_B T)

        -----------------------------------------------------------------------
        Returns:
            (float): β in units of eV^-1
        """
        if self.temp is None:
            raise ValueError('Cannot calculate β without a defined temperature'
                             ' please set .temp')

        k_b = 8.617333262E-5  # Boltzmann constant in eV / K
        return 1.0 / (k_b * self.temp)

    def wham(self,
             tol:                 float = 1E-3,
             max_iterations:      int = 100000,
             n_bins:              int = 100,
             units:               str = 'kcal mol-1',
             compute_uncertainty: bool = False,
             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct an unbiased distribution (on a grid) from a set of windows

        -----------------------------------------------------------------------
        Arguments:

            tol: Tolerance on the convergence

            max_iterations: Maximum number of WHAM iterations to perform

            n_bins: Number of bins to use in the histogram (minus one) and
                    the number of reaction coordinate values plotted and
                    returned

            units: (str) Energy units, available: eV, kcal mol-1, kj mol-1

            compute_uncertainty: (bool) If True compute free energy uncertainty
                                        using umbrella integration error
                                        propagation

        ---------------
        Keyword Arguments:

            blocksize: (int) Block size to use in uncertainty quantification.
                             If not supplied, a value of 1000 is used

        Returns:
            (np.ndarray, np.ndarray): Tuple containing the reaction coordinate
                                      and values of the free energy
        """
        beta = self.beta   # 1 / (k_B T)
        outer_zeta_refs = (self.zeta_refs[0], self.zeta_refs[-1])

        for window in self.windows:
            window.set_bin_edges(outer_zeta_refs, n_bins=n_bins)
            window.bin()

        # Discretised reaction coordinate
        zetas = np.linspace(self.zeta_refs[0], self.zeta_refs[-1], num=n_bins)

        p = np.ones_like(zetas) / len(zetas)  # P(ζ) uniform distribution
        p_prev = np.inf * np.ones_like(p)     # Start with P(ζ)_(-1) = ∞

        def converged():
            return np.max(np.abs(p_prev - p)) < tol

        for iteration in range(max_iterations):

            # Equation 8.8.18 from Tuckerman, p. 343
            p = (sum(w_k.hist for w_k in self.windows)
                 / sum(w_k.n * np.exp(beta * (w_k.free_energy - w_k.bias_energies))
                       for w_k in self.windows))

            for w_k in self.windows:
                # Equation 8.8.19 from Tuckerman, p. 343
                w_k.free_energy = (-(1.0/beta)
                                   * np.log(np.sum(p * np.exp(-w_k.bias_energies * beta))))

            if converged():
                logger.info(f'WHAM converged in {iteration} iterations')
                break

            p_prev = p

        if compute_uncertainty:
            self._attach_p_ui_values_to_windows(zetas=zetas)
            uncertainties = self._compute_ui_uncertainty(zetas=zetas, **kwargs)

        else:
            uncertainties = None

        self._save_free_energy(free_energies=self.free_energies(p),
                               zetas=zetas,
                               uncertainties=uncertainties,
                               units=units)
        self.plot_free_energy()

        return zetas, self.free_energies(p)

    def umbrella_integration(self,
                             n_bins:              int = 100,
                             units:               str = 'kcal mol-1',
                             compute_uncertainty: bool = False,
                             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform umbrella integration on the umbrella windows to un-bias the
        probability distribution. Such that the the PMF becomes

        .. math::
            dA/dq = Σ_i p_i(q) dA_i/ dq

        where the sum runs over the windows. Also plot and save the resulting
        free energy.

        -----------------------------------------------------------------------
        Arguments:

            n_bins: Number of bins to use in the histogram (minus one) and
                    the number of reaction coordinate values plotted and
                    returned

            units: (str) Energy units, available: eV, kcal mol-1, kj mol-1

            compute_uncertainty: (bool) If True compute free energy uncertainty
                                        using umbrella integration error
                                        propagation

        ---------------
        Keyword Arguments:

            blocksize: (int) Block size to use in uncertainty quantification.
                             If not supplied, a value of 1000 is used

        Returns:

            (np.ndarray, np.ndarray): Tuple containing the reaction coordinate
                                      and values of the free energy
        """
        beta = self.beta   # 1 / (k_B T)
        outer_zeta_refs = (self.zeta_refs[0], self.zeta_refs[-1])

        for window in self.windows:
            window.set_bin_edges(outer_zeta_refs, n_bins=n_bins)
            window.bin()

        # Discretised reaction coordinate
        zetas = np.linspace(self.zeta_refs[0], self.zeta_refs[-1], num=n_bins)
        zetas_spacing = zetas[1] - zetas[0]

        self._attach_p_ui_values_to_windows(zetas=zetas)

        dA_dq = np.zeros_like(zetas)
        for i, window in enumerate(self.windows):
            dA_dq += window.p_ui * window.dA_dq(zetas, beta=beta)

        free_energies = np.zeros_like(zetas)
        for i, _ in enumerate(zetas):
            if i == 0:
                free_energies[i] = 0.0
                continue

            free_energies[i] = simpson(dA_dq[:i],
                                       zetas[:i],
                                       dx=zetas_spacing)

        if compute_uncertainty:
            uncertainties = self._compute_ui_uncertainty(zetas=zetas, **kwargs)

        else:
            uncertainties = None

        self._save_free_energy(free_energies=free_energies,
                               zetas=zetas,
                               uncertainties=uncertainties,
                               units=units)
        self.plot_free_energy()

        return zetas, free_energies

    def _attach_p_ui_values_to_windows(self, zetas: np.ndarray) -> None:
        """
        Compute p_ui values for every window, (not to be confused with
        p values that appear in WHAM), which are used in umbrella integration
        and uncertainty quantification, and attach those values to the
        corresponding windows

        ----------------------------------------------------------------------
        Arguments:

            zetas: (np.ndarray) Discretised reaction coordinate
        """

        a_list = []
        for i, window in enumerate(self.windows):
            a_i = window.n * window.gaussian_pdf(zetas)
            a_list.append(a_i)

        sum_a = sum(a_list)
        p_ui_list = [a_i / sum_a for a_i in a_list]

        for i, (window, p_ui_i) in enumerate(zip(self.windows, p_ui_list)):
            window.p_ui = p_ui_i

        return None

    def _compute_ui_uncertainty(self,
                                zetas: np.ndarray,
                                **kwargs) -> Optional[np.ndarray]:
        """
        Compute free energy standard deviation using umbrella integration
        error propagation. It should mainly be used when windows are combined
        using umbrella integration, but in many cases the UI standard deviation
        is a good estimate for WHAM free energy too (care must be taken as the
        UI standard deviation cannot account for the growth of WHAM free energy
        statistical error when a large number of bins is used)[1]

        [1] Kastner, J., & Thiel, W. (2006). Journal of Chemical Physics,
            124(23), 234106. https://doi.org/10.1063/1.2206775

        -----------------------------------------------------------------------
        Arguments:

            zetas: (np.ndarray) Discretised reaction coordinate

        ---------------
        Keyword Arguments:

            blocksize: (int) Block size to use in uncertainty quantification.
                             If not supplied, a value of 1000 is used

        Returns:

            (np.ndarray): Free energy standard deviation with the same shape
                          as zetas
        """

        blocksize = kwargs.get('blocksize', 1000)
        min_n_blocks = 10
        n_obs_zetas = min(len(window._obs_zetas) for window in self.windows)
        max_blocksize = n_obs_zetas // min_n_blocks
        if blocksize > max_blocksize:
            logger.warning('Simulation is too short to get a good estimate '
                           'of the UI uncertainties. Either run a longer US '
                           'simulation, or change the default block size '
                           '(making sure the blocks are not correlated)')

            return None

        var_dA_dq = 0
        for i, window in enumerate(self.windows):
            var_dAi_dq = window.var_dA_dq(zetas=zetas,
                                          beta=self.beta,
                                          blocksize=blocksize)
            # [1] Equation 9
            var_dA_dq += window.p_ui**2 * var_dAi_dq

        var_A = np.zeros_like(zetas)
        for i, _ in enumerate(zetas):

            if i == 0:
                var_A[i] = 0.0
                continue

            lower_edge = zetas[0]
            upper_edge = zetas[i]
            average_std = self._compute_average_std_in_interval(lower_edge,
                                                                upper_edge)
            # [1] Equation 15
            var_A[i] = (np.mean(var_dA_dq[:i])
                        * (np.sqrt(2 * np.pi) * (upper_edge - lower_edge)
                           * average_std - 2 * average_std**2))

        return np.sqrt(np.abs(var_A))

    def truncate_window_trajectories(self,
                                     removed_fraction: float = 0.20
                                     ) -> None:
        """Remove not less then the fraction of the frames from the start of
        the window trajectories"""

        for window in self.windows:
            obs_zetas = window._obs_zetas
            n_removed = int(-(removed_fraction * len(obs_zetas) // -1))
            window._obs_zetas = obs_zetas[n_removed:]

        return None

    def _compute_average_std_in_interval(self,
                                         lower_edge: float,
                                         upper_edge: float
                                         ) -> float:
        """
        Compute average of standard deviations over the windows contributing
        to the interval, used in [1] Equation 15

        -----------------------------------------------------------------------
        Arguments:

            lower_edge: (float) Lowest value of the reaction coordinate, which
                                is the lower edge of the interval

            upper_edge: (float) Value of the reaction coordinate at which the
                                free energy is calculated, it is the upper edge
                                of the interval
        """

        integrals = np.zeros(len(self.windows))
        for i, window in enumerate(self.windows):
            distr = norm(loc=window.gaussian_pdf.mean,
                         scale=window.gaussian_pdf.std)
            integral = distr.cdf(upper_edge) - distr.cdf(lower_edge)
            integrals[i] = integral

        normalised_integrals = integrals / np.sum(integrals)

        average_std = 0
        for window, integral in zip(self.windows, normalised_integrals):
            average_std += integral * window.gaussian_pdf.std

        return average_std

    def window_block_analysis(self) -> None:
        """
        Perform block averaging analysis on the trajectories of each window and
        plot the results
        """

        with Pool(processes=Config.n_cores) as pool:

            for i, window in enumerate(self.windows, start=1):
                pool.apply_async(func=window.block_analysis, args=(i,))

            pool.close()
            pool.join()

        move_files(moved_substrings=[r'block_analysis_window_\d+\.pdf'],
                   dst_folder='window_block_analysis',
                   regex=True)

        return None

    @staticmethod
    def _save_free_energy(free_energies: np.ndarray,
                          zetas:         np.ndarray,
                          uncertainties: Optional[np.ndarray] = None,
                          units:         str = 'kcal mol-1'
                          ) -> None:
        """
        Save the free energy (and uncertainty) as a .txt file

        -----------------------------------------------------------------------
        Arguments:

            free_energies: (np.ndarray) Free energy values at every value of
                                        the reaction coordinate

            zetas: (np.ndarray) Values of the reaction coordinate

            uncertainties: (np.ndarray) Standard deviation of the free energy
                                    at every value of the reaction coordinate

            units: (str) Energy units, available: eV, kcal mol-1, kj mol-1
        """

        free_energies = convert_ase_energy(free_energies, units)
        rel_free_energies = free_energies - min(free_energies)

        filename = 'umbrella_free_energy.txt'
        if os.path.exists(filename):
            os.rename(filename, unique_name(filename))

        with open(filename, 'w') as outfile:
            print(f'# Units: {units.lower()}', file=outfile)

            if uncertainties is None:
                print('# Reaction_coordinate Free_energy',
                      file=outfile)

                data = zip(zetas, rel_free_energies)
                for zeta, free_energy in data:
                    print(zeta, free_energy, file=outfile)

            else:
                print('# Reaction_coordinate Free_energy Uncertainty',
                      file=outfile)

                uncertainties = convert_ase_energy(uncertainties, units)
                data = zip(zetas, rel_free_energies, uncertainties)
                for zeta, free_energy, uncertainty in data:
                    print(zeta, free_energy, uncertainty, file=outfile)

        return None

    @staticmethod
    def plot_free_energy(filename:         Optional[str] = None,
                         confidence_level: float = 0.95) -> None:
        """
        Plot the free energy against the reaction coordinate

        -----------------------------------------------------------------------
        Arguments:

            filename: (str) Name of the file containing reaction coordinate
                            values, free energies, and uncertainties

            confidence_level: (float) Specifies what confidence level to use
                                      in plots (probability for free energy
                                      to lie within the plotted range)
        """

        if filename is None:
            filename = 'umbrella_free_energy.txt'
            if not os.path.exists(filename):
                raise ValueError('File for plotting the free energy cannot be '
                                 'found, make sure to compute the free energy '
                                 'before running this method')

        logger.info(f'Plotting US free energy using {filename}')

        with open(filename, 'r') as f:
            # '# Units ...'
            first_line = f.readline()
            units = ' '.join(first_line.split()[2:])

            # '# Reaction_coordinate Free_energy Uncertainty'
            second_line = f.readline()
            uncertainty_present = second_line.split()[-1] == 'Uncertainty'

        zetas = np.loadtxt(filename, usecols=0)
        rel_free_energies = np.loadtxt(filename, usecols=1)

        fig, ax = plt.subplots()
        ax.plot(zetas, rel_free_energies, label='Free energy')

        if uncertainty_present:
            uncertainties = np.loadtxt(filename, usecols=2)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                conf_interval = norm.interval(confidence_level,
                                              loc=rel_free_energies,
                                              scale=uncertainties)

            lower_bound = conf_interval[0]
            upper_bound = conf_interval[1]

            ax.fill_between(zetas, lower_bound, upper_bound,
                            alpha=0.3,
                            label='Confidence interval')

        ax.legend()
        ax.set_xlabel('Reaction coordinate / Å')
        ax.set_ylabel(f'ΔG / {convert_exponents(units)}')

        fig.tight_layout()

        figname = 'umbrella_free_energy.pdf'
        if os.path.exists(figname):
            os.rename(figname, unique_name(figname))

        fig.savefig(figname)
        plt.close(fig)

        return None

    def save(self, folder_name: str = 'umbrella') -> None:
        """
        Save the windows in this US to a folder containing each window as .txt
        files within in
        """

        if len(self.windows) is None:
            logger.error(f'Cannot save US to {folder_name} - had no windows')
            return None

        os.mkdir(folder_name)
        for idx, window in enumerate(self.windows):
            window.save(filename=os.path.join(folder_name,
                                              f'window_{idx+1}.txt'))

        return None

    def load(self, folder_name: str) -> None:
        """Load data from a set of saved windows"""

        if not os.path.isdir(folder_name):
            raise ValueError(f'Loading from a folder was not possible as '
                             f'{folder_name} is not a valid folder')

        for filename in os.listdir(folder_name):

            if filename.startswith('window_') and filename.endswith('.txt'):
                window = _Window.from_file(os.path.join(folder_name, filename))
                self.windows.append(window)

        return None

    @classmethod
    def from_folder(cls,
                    folder_name: str,
                    temp: float) -> 'UmbrellaSampling':
        """
        Create an umbrella sampling instance from a folder containing the
        window data

        -----------------------------------------------------------------------
        Arguments:
            folder_name:

            temp: Temperature (K)

        Returns:
            (mlptrain.sampling.umbrella.UmbrellaSampling):
        """
        us = cls(zeta_func=DummyCoordinate(), kappa=0.0, temp=temp)
        us.load(folder_name=folder_name)
        us._order_windows_by_zeta_ref()

        return us

    @classmethod
    def from_folders(cls,
                     *args: str,
                     temp: float) -> 'UmbrellaSampling':
        """
        Load a set of individual umbrella sampling simulations in to a single
        one

        -----------------------------------------------------------------------
        Arguments:
            *args: Names of folders

            temp: Temperature (K)

        Returns:
            (mlptrain.sampling.umbrella.UmbrellaSampling):
        """
        us = cls(zeta_func=DummyCoordinate(), kappa=0.0, temp=temp)

        for folder_name in args:
            us.load(folder_name=folder_name)

        us._order_windows_by_zeta_ref()
        return us

    def _order_windows_by_zeta_ref(self) -> None:
        """Sort the windows in this umbrella by ζ_ref"""
        self.windows = sorted(self.windows, key=lambda window: window.zeta_ref)
        return None


class _FittedGaussian:

    def __init__(self,
                 a: float = 1.0,
                 b: float = 1.0,
                 c: float = 1.0):
        """
        Gaussian defined by three parameters:

        a * exp(-(x - b)^2 / (2 * c^2))
        """
        self.params = a, b, c

    def __call__(self, x):
        return self.value(x, *self.params)

    @staticmethod
    def value(x, a, b, c):
        return a * np.exp(-(x - b)**2 / (2. * c**2))

    @property
    def mean(self) -> float:
        """Mean of the Normal distribution, of which this is an approx."""
        return self.params[1]

    @property
    def std(self) -> float:
        """Standard deviation of the Normal distribution"""
        return np.abs(self.params[2])
