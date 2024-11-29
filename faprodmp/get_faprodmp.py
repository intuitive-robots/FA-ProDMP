import typing
import logging
import pandas as pd
import torch
import numpy as np
from copy import deepcopy

from utils import cholesky

from mp_pytorch.mp import MPFactory
from mp_pytorch.mp import ProDMP

########### MP Hyperparameter Defaults ###################
TAU = 1 # prerequisite: recorded data index has already been normalized in range [0, 1]
DT = 1e-3 # precision of the precomputation
AUTO_SCALE = False
REGULARIZATION_FACTOR = 1e-9
DTYPE = torch.float64
# unchanged defaults
LEARN_TAU = False
LEARN_DELAY = False
BASIS_BANDWIDTH_FACTOR = 2
NUM_BASIS_OUTSIDE = 0
ALPHA = 25
ALPHA_PHASE = 2 
####################################################################################################


def get_FAPoDMP(trajectories: typing.List[pd.DataFrame],
             pos_features: typing.List[str],
             force_features: typing.List[str],
             num_basis:int,
             resolution: int = 1000,
             logger: logging.Logger = logging.Logger('FA-ProDMP_helper')
             ) -> ProDMP:
    """creates the base FA-ProDMP based on the provided trajectories

    Args:
        trajectories (typing.List[pd.DataFrame]): recorded robot trajectories as pandas DataFrames
        pos_features (typing.List[str]): Column names of the position features
        force_features (typing.List[str]): column names of the force features
        num_basis (int): number of basis functions used in the MP
        resolution (int, optional): number of interpolation points used for the MP creation. Defaults to 1000.
        logger (logging.Logger, optional): logger to be used by the utility. Defaults to logging.Logger('FA-ProDMP_helper').

    Returns:
        ProDMP: the base FA-ProDMP
    """
    # MP configuration
    mp_config = dict()
    mp_config["mp_type"] = 'prodmp'
    mp_config["num_dof"] = len(pos_features) + len(force_features)
    mp_config["tau"] = TAU
    mp_config["dtype"] = DTYPE
    mp_config["mp_args"]["dt"] = DT
    mp_config["learn_tau"] = LEARN_TAU
    mp_config["learn_delay"] = LEARN_DELAY
    mp_config["mp_args"]["num_basis"] = num_basis
    mp_config["mp_args"]["basis_bandwidth_factor"] = BASIS_BANDWIDTH_FACTOR
    mp_config["mp_args"]["num_basis_outside"] = NUM_BASIS_OUTSIDE
    mp_config["mp_args"]["alpha"] = ALPHA
    mp_config["mp_args"]["alpha_phase"] = ALPHA_PHASE
    mp_config["mp_args"]["auto_scale_basis"] = AUTO_SCALE

    logger.info('Equalizing trajectory resolution (number of steps)...')
    equalized_trajectories = _equalize_resolution(trajectories, resolution)

    logger.info('Transforming trajectories to torch.Tensor...')
    times, trajectories_tensor = _df_trajectories_to_tensor(equalized_trajectories)        

    logger.info('Learning parameters...')
    prodmp = MPFactory.init_mp(**deepcopy(mp_config)) # create a dummy MP to learn the parameters
    param_dict = prodmp.learn_mp_params_from_trajs(times, trajectories_tensor, reg=REGULARIZATION_FACTOR)

    logger.info('Computing parameter distribution...')
    params = param_dict['params']
    mean_params, params_L = _get_param_distribution(params)

    logger.info('Creating generative FA-ProDMP...')
    faprodmp = MPFactory.init_mp(**deepcopy(mp_config))
    faprodmp.update_inputs(
        times = times.mean(dim=0),
        params = mean_params,
        params_L = params_L,
        init_time = param_dict['init_time'].mean(),
        init_vel = param_dict['init_vel'].mean(dim=0),
        init_pos = param_dict['init_pos'].mean(dim=0)
    )

    return faprodmp

def _equalize_resolution(trajectories: typing.List[pd.DataFrame], 
                         num_steps:int
                         ) -> typing.List[pd.DataFrame]:
    """equalizes the number of interpolation steps in the trajectories

    Args:
        trajectories (typing.List[pd.DataFrame]): recorded robot trajectories as pandas DataFrames
        num_steps (int): number of interpolation steps

    Returns:
        typing.List[pd.DataFrame]: the equalized trajectories
    """
    trajs = deepcopy(trajectories)
    for traj_idx in range(len(trajs)):
        min_idx = trajs[traj_idx].index.min()
        max_idx = trajs[traj_idx].index.max()
        new_index = np.linspace(min_idx, max_idx, num_steps)
        new_df = pd.DataFrame(index=new_index)
        for col in trajs[traj_idx].columns:
            new_df[col] = np.interp(new_index, trajs[traj_idx].index, trajs[traj_idx][col])
        trajs[traj_idx] = new_df
    return trajs

def _df_trajectories_to_tensor(trajectories: typing.List[pd.DataFrame]
                               ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """transforms the trajectories to torch.Tensor

    Args:
        trajectories (typing.List[pd.DataFrame]): trajectories as pandas DataFrames

    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]: time indices and trajectories as 3D tensors
    """
    times = torch.Tensor()
    trajs = torch.Tensor()
    for trajectory in trajectories:
        # get times
        t = trajectory.index.to_numpy()
        t = torch.tensor(t, dtype=torch.float64)
        t = t.unsqueeze(0) # convert to 2D tensor for concatenation
        times = torch.cat((times, t), dim=0)

        # get position trajectory
        traj = trajectory.to_numpy()
        traj = torch.tensor(traj, dtype=torch.float64)
        traj = traj.unsqueeze(0) # convert to 2D tensor for concatenation
        trajs = torch.cat((trajs,traj), dim=0)
    return times, trajs

def _get_param_distribution(params: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """calculates the distribution characteristics over the provided DMP parameters

    Args:
        params (torch.Tensor): the DMP parameters of the demonstrations

    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]: mean and Cholesky decomposition of the covariance matrix
    """
    if params.size(0) == 1:
        raise ValueError("Only one parameter set provided. Cannot compute distribution.")

    mean_params = params.mean(dim=0)

    centered_params = params - mean_params
    cov_matrix = (centered_params.T @ centered_params) / (params.size(0) - 1)
    params_L = cholesky(cov_matrix)
    return mean_params, params_L