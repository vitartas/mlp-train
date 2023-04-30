import os
import ast
import time
# import shutil
import logging
import numpy as np
from typing import Optional, Dict, List
from ase.calculators.calculator import Calculator
from ase.data import chemical_symbols
from mlptrain.potentials._base import MLPotential
from mlptrain.config import Config
from mlptrain.box import Box
from mlptrain.log import logger

try:
    import torch
    import torch.nn.functional
    import mace
    from e3nn import o3
    from torch.optim.swa_utils import SWALR, AveragedModel
    from torch_ema import ExponentialMovingAverage
    from mace import data, modules, tools
    from mace.tools import torch_geometric, torch_tools, utils
    from mace.tools.scripts_utils import create_error_table, get_dataset_from_xyz
    # from mace.calculators import MACECalculator as ParentMACECalculator
    from mace.calculators import MACECalculator
except ModuleNotFoundError:
    pass


# class MACECalculator(ParentMACECalculator):
#
#     def __init__(self,
#                  model_path: str,
#                  # mace: 'mlptrain.potentials.MACE',
#                  device: str,
#                  energy_units_to_eV: float = 1.0,
#                  length_units_to_A: float = 1.0,
#                  default_dtype: str = 'float64',
#                  **kwargs):
#         """doc"""
#
#         Calculator.__init__(self, **kwargs)
#         self.results = {}
#
#         # model = getattr(modules, Config.mace_params['model'])
#         ## self.model = model()
#         # self.model = model.load_state_dict(torch.load(model_path))
#
#         self.mace = MACE(name='model_path', system=None)
#         self.state_dict = torch.load(f'{self.mace.name}_state_dict.model')
#         self.model = self._get_model()
#
#         self.r_max = self.state_dict['r_max']
#         self.device = torch_tools.init_device(device)
#         self.energy_units_to_eV = energy_units_to_eV
#         self.length_units_to_A = length_units_to_A
#         self.z_table = utils.AtomicNumberTable(
#             [int(z) for z in self.state_dict['atomic_numbers']])
#
#         torch_tools.set_default_dtype(default_dtype)
#
#         # self.model = torch.load(f=model_path, map_location=device)
#         # self.r_max = float(self.model.r_max)
#         # self.device = torch_tools.init_device(device)
#         # self.energy_units_to_eV = energy_units_to_eV
#         # self.length_units_to_A = length_units_to_A
#         # self.z_table = utils.AtomicNumberTable(
#         #     [int(z) for z in self.model.atomic_numbers])
#
#     @property
#     def args(self) -> '':
#         """doc"""
#         return self.mace.args
#
#     def _get_model(self) -> 'torch.nn.Module':
#         """Loads the torch model from the state dictionary"""
#
#         model_config = dict(
#             r_max=Config.mace_params['r_max'],
#             num_bessel=self.args.num_radial_basis,
#             num_polynomial_cutoff=self.args.num_cutoff_basis,
#             max_ell=self.args.max_ell,
#             interaction_cls=modules.interaction_classes[
#                 self.args.interaction],
#             num_interactions=self.args.num_interactions,
#             num_elements=len(self.z_table),
#             hidden_irreps=o3.Irreps(Config.mace_params['hidden_irreps']),
#             atomic_energies=self.atomic_energies_array,
#             avg_num_neighbors=self.avg_num_neighbors,
#             atomic_numbers=self.z_table.zs)
#
#         if Config.mace_params['model'] == "MACE":
#
#             self.model = modules.ScaleShiftMACE(
#                 **model_config,
#                 correlation=Config.mace_params['correlation'],
#                 gate=modules.gate_dict[self.args.gate],
#                 interaction_cls_first=modules.interaction_classes[
#                     "RealAgnosticInteractionBlock"],
#                 MLP_irreps=o3.Irreps(self.args.MLP_irreps),
#                 atomic_inter_scale=self.state_dict['scale_shift.scale'],
#                 atomic_inter_shift=0.0)
#
#         elif Config.mace_params['model'] == "ScaleShiftMACE":
#
#             self.model = modules.ScaleShiftMACE(
#                 **model_config,
#                 correlation=Config.mace_params['correlation'],
#                 gate=modules.gate_dict[self.args.gate],
#                 interaction_cls_first=modules.interaction_classes[
#                     self.args.interaction_first],
#                 MLP_irreps=o3.Irreps(self.args.MLP_irreps),
#                 atomic_inter_scale=self.state_dict['scale_shift.scale'],
#                 atomic_inter_shift=self.state_dict['scale_shift.shift'])
#
#         elif Config.mace_params['model'] == "ScaleShiftBOTNet":
#
#             self.model = modules.ScaleShiftBOTNet(
#                 **model_config,
#                 gate=modules.gate_dict[self.args.gate],
#                 interaction_cls_first=modules.interaction_classes[
#                     self.args.interaction_first],
#                 MLP_irreps=o3.Irreps(self.args.MLP_irreps),
#                 atomic_inter_scale=self.state_dict['scale_shift.scale'],
#                 atomic_inter_shift=self.state_dict['scale_shift.shift'])
#
#         elif Config.mace_params['model'] == "BOTNet":
#
#             self.model = modules.BOTNet(
#                 **model_config,
#                 gate=modules.gate_dict[self.args.gate],
#                 interaction_cls_first=modules.interaction_classes[
#                     self.args.interaction_first],
#                 MLP_irreps=o3.Irreps(self.args.MLP_irreps))
#
#         else:
#             raise RuntimeError(f'Unknown model: {Config.mace_params["model"]}')


class MACE(MLPotential):
    """@DynamicAttrs"""

    def __init__(self,
                 name:   str,
                 system: 'mlptrain.System',
                 ) -> None:
        """
        MACE machine learning potential

        -----------------------------------------------------------------------
        Arguments:

            name: (str) Name of the potential, used in naming output files

            system: (mlptrain.System) Object defining the system without
                                      specifying the coordinates
        """
        super().__init__(name=name, system=system)

        try:
            import mace
        except ModuleNotFoundError:
            raise ModuleNotFoundError('MACE install not found, install it '
                                      'here: https://github.com/ACEsuit/mace')

        logging.info(f"MACE version: {mace.__version__}")

        # Cache
        self._defaults = {'_train_configs': None,
                          '_valid_configs': None,
                          '_z_table': None,
                          '_loss_fn': None,
                          '_train_loader': None,
                          '_valid_loader': None,
                          '_model': None,
                          '_optimizer': None,
                          '_scheduler': None,
                          '_checkpoint_handler': None,
                          '_start_epoch': None,
                          '_swa': None,
                          '_ema': None}

        for var, default in self._defaults.items():
            setattr(self, var, default)

    def _train(self,
               n_cores: Optional[int] = None
               ) -> None:
        """
        Train MACE potential using the data as .xyz file and save the
        final potential as .model file

        -----------------------------------------------------------------------
        Arguments:

            n_cores: (int) Number of cores to use in training
        """

        n_cores = (n_cores if n_cores is not None else Config.n_cores)
        os.environ['OMP_NUM_THREADS'] = str(n_cores)
        logger.info('Training a MACE potential on '
                    f'*{len(self.training_data)}* training data, '
                    f'using {n_cores} in training')

        for config in self.training_data:
            if self.requires_non_zero_box_size and config.box is None:
                config.box = Box([100, 100, 100])

        self.training_data.save_xyz(filename=f'{self.name}_data.xyz')

        start_time = time.perf_counter()
        self._run_train()
        delta_time = time.perf_counter() - start_time

        logger.info(f'MACE training ran in {delta_time / 60:.1f} m')

        self._load_latest_epoch()
        self._show_error_table()
        self._save_model()

        os.remove(f'{self.name}_data.xyz')
        return None

    @property
    def requires_atomic_energies(self) -> bool:
        return True

    @property
    def requires_non_zero_box_size(self) -> bool:
        """ACE cannot use a zero size box"""
        return True

    @property
    def ase_calculator(self) -> 'ase.calculators.calculator.Calculator':
        """ASE calculator for MACE potential"""

        calculator = MACECalculator(model_path=self.filename,
                                    device='cpu',
                                    default_dtype="float64")
        return calculator

    # @property
    # def filename(self):
    #     """Filename of the saved potential"""
    #     return f'{self.name}.model'

    # TODO:
    # def pool(self, processes: int) -> 'torch.multiprocessing.Pool':
    #     """Multiprocessing pool of the potential"""
    #     return torch.multiprocessing.Pool(processes=processes)


    def _run_train(self) -> None:
        """
        Run MACE training.

        This code is adjusted from run_train.py in the MACE package.
        For more details, see
        https://github.com/ACEsuit/mace/tree/main/scripts
        """

        self._reset_train_objs()

        tools.setup_logger(level=self.args.log_level,
                           tag=self.name,
                           directory=self.args.log_dir)
        tools.set_seeds(self.args.seed)
        tools.set_default_dtype(self.args.default_dtype)

        logging.info(f'Using {self.args.train_file} as the training set')
        logging.info(f'Total number of configurations: '
                     f'valid={len(self.valid_configs)}, '
                     f'train={len(self.train_configs)}')
        logging.info(self.z_table)
        logging.info(f'Chemical symbols: {self.z_table_symbol}')
        logging.info(f'Atomic energies: {self.atomic_energies}')
        logging.info(f'Loss: {self.loss_fn}')
        logging.info(f'Selected the following outputs: {self.output_args}')

        if self.args.compute_avg_num_neighbors:
            logging.info(f'Average number of neighbors: '
                         f'{self.avg_num_neighbors:.3f}')

        self.model.to(Config.mace_params['device'])

        metrics_logger = tools.MetricsLogger(directory=self.args.results_dir,
                                             tag=f'{self.name}_train')

        logging.info(f'Model: {self.model}')
        logging.info(f'Number of parameters: '
                     f'{tools.count_parameters(self.model)}')
        logging.info(f'Optimizer: {self.optimizer}')

        tools.train(model=self.model,
                    loss_fn=self.loss_fn,
                    train_loader=self.train_loader,
                    valid_loader=self.valid_loader,
                    optimizer=self.optimizer,
                    lr_scheduler=self.scheduler,
                    checkpoint_handler=self.checkpoint_handler,
                    eval_interval=self.args.eval_interval,
                    start_epoch=self.start_epoch,
                    max_num_epochs=self.max_num_epochs,
                    logger=metrics_logger,
                    patience=self.args.patience,
                    output_args=self.output_args,
                    device=Config.mace_params['device'],
                    swa=self.swa,
                    ema=self.ema,
                    max_grad_norm=self.args.clip_grad,
                    log_errors=Config.mace_params['error_table'])

        return None

    def _load_latest_epoch(self) -> None:
        """Load the latest epoch of the training"""

        epoch = self.checkpoint_handler.load_latest(
            state=tools.CheckpointState(self.model,
                                        self.optimizer,
                                        self.scheduler),
            device=Config.mace_params['device'])

        logging.info(f"Loaded model from epoch {epoch}")

        return None

    def _show_error_table(self) -> None:
        """Generate an error table and show it in logs"""

        logging.info("Generating error table")

        all_collections = [("train", self.train_configs),
                           ("valid", self.valid_configs)]

        table = create_error_table(
            table_type=Config.mace_params['error_table'],
            all_collections=all_collections,
            z_table=self.z_table,
            r_max=Config.mace_params['r_max'],
            valid_batch_size=self.valid_batch_size,
            model=self.model,
            loss_fn=self.loss_fn,
            output_args=self.output_args,
            log_wandb=self.args.wandb,
            device=Config.mace_params['device'])

        logging.info("\n" + str(table))
        return None

    def _save_model(self) -> None:
        """Save the trained model"""

        model_filename = f'{self.name}.model'
        model_path = os.path.join(self.args.checkpoints_dir, model_filename)
        state_dict_filename = f'{self.name}_state_dict.model'

        if Config.mace_params['save_cpu']:
            self.model.to('cpu')

        logging.info(f'Saving the model {model_filename} '
                     f'to {self.args.checkpoints_dir} '
                     'and its state dict in the current directory')

        torch.save(self.model, model_path)
        # TODO:
        # shutil.copyfile(src=os.path.join(os.getcwd(), model_path),
        #                 dst=os.path.join(os.getcwd(), self.filename))
        torch.save(self.model.state_dict(), state_dict_filename)

        return None

    def _reset_train_objs(self) -> None:
        """Reset training objects to defaults, important during retraining"""
        for var, default in self._defaults.items():
            setattr(self, var, default)

        return None

    @property
    def max_num_epochs(self) -> int:
        """Maximum number of epochs to perform during training"""
        if self.n_train < 100:
            return 1000

        elif 100 <= self.n_train < 300:
            return 1200

        else:
            return 1500

    @property
    def args(self) -> 'argparse.Namespace':
        """Namespace containing mostly default MACE parameters"""
        args = mace.tools.build_default_arg_parser().parse_args([
            '--name', self.name,
            '--train_file', f'{self.name}_data.xyz',
            '--default_dtype', 'float64'])
        return args

    @property
    def device(self) -> 'torch.device':
        """Device to use for training"""
        return tools.init_device(device_str=Config.mace_params['device'])

    @property
    def config_type_weights(self) -> Dict:
        """Dictionary containing the weights for each configuration type"""
        config_type_weights = ast.literal_eval(
            Config.mace_params['config_type_weights'])

        if not isinstance(config_type_weights, dict):
            logging.warning('Config type weights not specified correctly, '
                            'using Default')
            config_type_weights = {"Default": 1.0}

        return config_type_weights

    @property
    def z_table(self) -> 'mace.tools.AtomicNumberTable':
        """Table containing atomic numbers of the system"""

        if self._z_table is None:
            self._z_table = tools.get_atomic_number_table_from_zs(
                z
                for configs in (self.train_configs, self.valid_configs)
                for config in configs
                for z in config.atomic_numbers)

        return self._z_table

    @property
    def z_table_symbol(self) -> List[str]:
        """List of chemical symbols of the system"""
        return [chemical_symbols[i] for i in self.z_table.zs]

    @property
    def atomic_energies_array(self) -> np.ndarray:
        """List of atomic energies of the system"""
        return np.array([self.atomic_energies[symbol]
                         for symbol in self.z_table_symbol])

    @property
    def train_configs(self) -> 'mace.data.Configurations':
        """Configurations in the training dataset"""

        if self._train_configs is None:
            _, self._train_configs = data.load_from_xyz(
                file_path=self.args.train_file,
                config_type_weights=self.config_type_weights,
                energy_key=self.args.energy_key,
                forces_key=self.args.forces_key,
                extract_atomic_energies=False)

        return self._train_configs

    @property
    def valid_configs(self) -> 'mace.data.Configurations':
        """Configurations in the validation dataset"""

        if self._valid_configs is None:

            if self.args.valid_file is not None:
                _, self._valid_configs = data.load_from_xyz(
                    file_path=self.args.valid_path,
                    config_type_weights=self.config_type_weights,
                    energy_key=self.args.energy_key,
                    forces_key=self.args.forces_key,
                    extract_atomic_energies=False)

                logging.info(f'Loaded {len(self._valid_configs)} validation'
                             f'configurations from "{self.args.valid_file}"')

            else:
                logging.info(f'Using {100 * self.args.valid_fraction}% of the '
                             'training set for validation')

                self._train_configs, self._valid_configs = data.random_train_valid_split(
                    self.train_configs,
                    self.args.valid_fraction,
                    self.args.seed)

        return self._valid_configs

    @property
    def loss_fn(self) -> 'torch.nn.Module':
        """Loss function to use in the training"""

        if self._loss_fn is None:

            if Config.mace_params['loss'] == "weighted":
                self._loss_fn = modules.WeightedEnergyForcesLoss(
                    energy_weight=1.0,
                    forces_weight=5.0)

            elif Config.mace_params['loss'] == "forces_only":
                self._loss_fn = modules.WeightedForcesLoss(forces_weight=5.0)

            else:
                logging.info(f'{Config.mace_params["loss"]} is not allowed in '
                             f'mlp-train, setting loss to EnergyForcesLoss')

                self._loss_fn = modules.EnergyForcesLoss(
                    energy_weight=Config.mace_params['energy_weight'],
                    forces_weight=Config.mace_params['forces_weight'])

        return self._loss_fn

    @property
    def train_batch_size(self) -> int:
        """Batch size of the training set"""
        if len(self.train_configs) < Config.mace_params['batch_size']:
            return len(self.train_configs)
        else:
            return Config.mace_params['batch_size']

    @property
    def valid_batch_size(self) -> int:
        """Batch size of the validation set"""
        if len(self.valid_configs) < Config.mace_params['batch_size']:
            return len(self.valid_configs)
        else:
            return Config.mace_params['batch_size']

    @property
    def train_loader(self) -> 'mace.tools.torch_geometric.dataloader.DataLoader':
        """Torch dataloader with training configurations"""

        if self._train_loader is None:

            self._train_loader = torch_geometric.dataloader.DataLoader(
                dataset=[data.AtomicData.from_config(
                    config,
                    z_table=self.z_table,
                    cutoff=Config.mace_params['r_max'])
                    for config in self.train_configs],
                batch_size=self.train_batch_size,
                shuffle=True,
                drop_last=True)

        return self._train_loader

    @property
    def valid_loader(self) -> 'mace.tools.torch_geometric.dataloader.DataLoader':
        """Torch dataloader with validation configurations"""

        if self._valid_loader is None:

            self._valid_loader = torch_geometric.dataloader.DataLoader(
                dataset=[data.AtomicData.from_config(
                    config,
                    z_table=self.z_table,
                    cutoff=Config.mace_params['r_max'])
                    for config in self.valid_configs],
                batch_size=self.valid_batch_size,
                shuffle=False,
                drop_last=False)

        return self._valid_loader

    @property
    def avg_num_neighbors(self) -> float:
        """Average number of neighbours in the training set"""
        if self.args.compute_avg_num_neighbors:
            return modules.compute_avg_num_neighbors(self.train_loader)
        else:
            return self.args.avg_num_neighbors

    @property
    def output_args(self) -> Dict:
        """Dictionary containing required outputs"""
        return {"energy": True,
                "forces": True,
                "virials": False,
                "stress": False,
                "dipoles": False}

    @property
    def model(self) -> torch.nn.Module:
        """Torch Module to use in training"""

        if self._model is None:

            logging.info("Building model")

            # TODO: maybe a function to build a model, could use in calculator
            model_config = dict(
                r_max=Config.mace_params['r_max'],  # state dict
                num_bessel=self.args.num_radial_basis,  # args (fine as cannot change)
                num_polynomial_cutoff=self.args.num_cutoff_basis,
                max_ell=self.args.max_ell,
                interaction_cls=modules.interaction_classes[
                    self.args.interaction],
                num_interactions=self.args.num_interactions,
                num_elements=len(self.z_table),  # state dict
                hidden_irreps=o3.Irreps(Config.mace_params['hidden_irreps']),  # How do you know what hidden irreps were used for a given save?
                atomic_energies=self.atomic_energies_array,  # How do you save atomic energies, should you recompute?
                avg_num_neighbors=self.avg_num_neighbors,  # also smth that's not saved
                atomic_numbers=self.z_table.zs)  # state dict

            if Config.mace_params['model'] == "MACE":

                if self.args.scaling == "no_scaling":
                    std = 1.0
                    logging.info("No scaling selected")
                else:
                    mean, std = modules.scaling_classes[self.args.scaling](
                        self.train_loader,
                        self.atomic_energies_array)

                self._model = modules.ScaleShiftMACE(
                    **model_config,
                    correlation=Config.mace_params['correlation'],
                    gate=modules.gate_dict[self.args.gate],
                    interaction_cls_first=modules.interaction_classes[
                        "RealAgnosticInteractionBlock"],
                    MLP_irreps=o3.Irreps(self.args.MLP_irreps),
                    atomic_inter_scale=std,
                    atomic_inter_shift=0.0)

            elif Config.mace_params['model'] == "ScaleShiftMACE":

                mean, std = modules.scaling_classes[self.args.scaling](
                    self.train_loader,
                    self.atomic_energies_array)

                self._model = modules.ScaleShiftMACE(
                    **model_config,
                    correlation=Config.mace_params['correlation'],
                    gate=modules.gate_dict[self.args.gate],
                    interaction_cls_first=modules.interaction_classes[
                        self.args.interaction_first],
                    MLP_irreps=o3.Irreps(self.args.MLP_irreps),
                    atomic_inter_scale=std,
                    atomic_inter_shift=mean)

            elif Config.mace_params['model'] == "ScaleShiftBOTNet":

                mean, std = modules.scaling_classes[self.args.scaling](
                    self.train_loader,
                    self.atomic_energies_array)

                self._model = modules.ScaleShiftBOTNet(
                    **model_config,
                    gate=modules.gate_dict[self.args.gate],
                    interaction_cls_first=modules.interaction_classes[
                        self.args.interaction_first],
                    MLP_irreps=o3.Irreps(self.args.MLP_irreps),
                    atomic_inter_scale=std,
                    atomic_inter_shift=mean)

            elif Config.mace_params['model'] == "BOTNet":

                self._model = modules.BOTNet(
                    **model_config,
                    gate=modules.gate_dict[self.args.gate],
                    interaction_cls_first=modules.interaction_classes[
                        self.args.interaction_first],
                    MLP_irreps=o3.Irreps(self.args.MLP_irreps))

            else:
                raise RuntimeError(f'{Config.mace_params["model"]} cannot be '
                                   'used in mlp-train, please specify a '
                                   'different model in Config.mace_params')

        return self._model

    @property
    def opt_param_options(self) -> Dict:
        """Dictionary with optimiser parameter options"""

        decay_interactions = {}
        no_decay_interactions = {}
        for name, param in self.model.interactions.named_parameters():
            if "linear.weight" in name or "skip_tp_full.weight" in name:
                decay_interactions[name] = param
            else:
                no_decay_interactions[name] = param

        param_options = dict(
            params=[
                {
                    "name": "embedding",
                    "params": self.model.node_embedding.parameters(),
                    "weight_decay": 0.0,
                },
                {
                    "name": "interactions_decay",
                    "params": list(decay_interactions.values()),
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "name": "interactions_no_decay",
                    "params": list(no_decay_interactions.values()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "products",
                    "params": self.model.products.parameters(),
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "name": "readouts",
                    "params": self.model.readouts.parameters(),
                    "weight_decay": 0.0
                },
            ],
            lr=self.args.lr,
            amsgrad=Config.mace_params['amsgrad'])

        return param_options

    @property
    def optimizer(self) -> 'torch.optim.Optimizer':
        """Optimiser to use in training"""

        if self._optimizer is None:

            if self.args.optimizer == "adamw":
                self._optimizer = torch.optim.AdamW(**self.opt_param_options)
            else:
                self._optimizer = torch.optim.Adam(**self.opt_param_options)

        return self._optimizer

    @property
    def scheduler(self) -> 'torch.optim.lr_scheduler':
        """Torch scheduler for training"""

        if self._scheduler is None:

            if self.args.scheduler == "ExponentialLR":
                self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=self.optimizer,
                    gamma=self.args.lr_scheduler_gamma)

            elif self.args.scheduler == "ReduceLROnPlateau":
                self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=self.optimizer,
                    factor=self.args.lr_factor,
                    patience=self.args.scheduler_patience)

            else:
                raise RuntimeError(f'Unknown scheduler: '
                                   f'{self.args.scheduler}')

        return self._scheduler

    @property
    def checkpoint_handler(self) -> 'mace.tools.CheckpointHandler':
        """Checkpoint handler to use in training"""

        if self._checkpoint_handler is None:
            self._checkpoint_handler = tools.CheckpointHandler(
                directory=self.args.checkpoints_dir,
                tag=self.name,
                keep=self.args.keep_checkpoints)

        return self._checkpoint_handler

    @property
    def start_epoch(self) -> int:
        """Start epoch of the training"""

        if self._start_epoch is None:

            self._start_epoch = 0
            if Config.mace_params['restart_latest']:
                opt_start_epoch = self.checkpoint_handler.load_latest(
                    state=tools.CheckpointState(self.model,
                                                self.optimizer,
                                                self.scheduler),
                    device=Config.mace_params['device'])

                if opt_start_epoch is not None:
                    self._start_epoch = opt_start_epoch

        return self._start_epoch

    @property
    def swa(self) -> Optional['mace.tools.SWAContainer']:
        """Object for stochastic weight averaging during training"""

        if self._swa is None:

            if Config.mace_params['swa']:

                if Config.mace_params['start_swa'] is None:
                    # if not set start swa at 75% of training
                    start_swa = self.max_num_epochs // 4 * 3
                else:
                    start_swa = Config.mace_params['start_swa']

                if Config.mace_params['loss'] == "forces_only":
                    logging.info("Can not select swa with forces only loss.")

                loss_fn_energy = modules.WeightedEnergyForcesLoss(
                    energy_weight=self.args.swa_energy_weight,
                    forces_weight=self.args.swa_forces_weight)

                self._swa = tools.SWAContainer(
                    model=AveragedModel(self.model),
                    scheduler=SWALR(
                        optimizer=self.optimizer,
                        swa_lr=self.args.swa_lr,
                        anneal_epochs=1,
                        anneal_strategy="linear"),
                    start=start_swa,
                    loss_fn=loss_fn_energy)

                logging.info(f'Using stochastic weight averaging '
                             f'(after {self._swa.start} epochs) with '
                             f'energy weight : {self.args.swa_energy_weight}, '
                             f'forces weight : {self.args.swa_forces_weight}, '
                             f'learning rate : {self.args.swa_lr}')

            else:
                self._swa = None

        return self._swa

    @property
    def ema(self) -> Optional['torch_ema.ExponentialMovingAverage']:
        """Object for exponantial moving average during training"""

        if self._ema is None:

            if Config.mace_params['ema']:
                self._ema = ExponentialMovingAverage(
                    self.model.parameters(),
                    decay=Config.mace_params['ema_decay'])

            else:
                self._ema = None

        return self._ema
