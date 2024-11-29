from copy import deepcopy
import logging
import numpy as np
import pandas as pd
import sys
import torch
import typing

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

def condition_FAProDMP_on_force(famp: ProDMP, 
                            current_trajectory: torch.Tensor,
                            time_idx: int, 
                            scaled_measured_forces: torch.Tensor,
                            scaled_force_threshold: float,
                            # force_sum_threshold: float = 0.67,
                            force_sum_threshold: float = 0.5,
                            reg: float = 1e-20
                            ) -> ProDMP:
    """conditions a FA-ProDMP based on measured forces

    Args:
        famp (ProDMP): the base FA-ProDMP
        current_trajectory (torch.Tensor): current planned trajectory
        time_idx (int): the time index at which the conditioning takes place
        scaled_measured_forces (torch.Tensor): the measured forces, scaled in the FA-ProDMP force scale
        scaled_force_threshold (float): the threshold at which to condition, scaled in the FA-ProDMP force scale
        force_sum_threshold (float, optional): the ratio of the threshold the conditioning forces should reach. Defaults to 0.67.
        reg (float, optional): regularization term for numerical stability. Defaults to 1e-20.

    Returns:
        ProDMP: the conditioned FA-ProDMP
    """
    # initialize conditioned FA-ProDMP
    conditioned_famp = deepcopy(famp)
    times = conditioned_famp.times

    num_dof_force = scaled_measured_forces.shape[0]
    num_dof_pos = conditioned_famp.num_dof - num_dof_force
    weight_mean_old = conditioned_famp.params.unsqueeze(1)
    weight_cov_old = conditioned_famp.params_L @ conditioned_famp.params_L.T

    # initialize conditioning vector
    conditioning = torch.cat((
        torch.ones(num_dof_pos,1) * sys.maxsize, # position will not be conditioned
        scaled_measured_forces
        ), dim = 0)

    # get prediction
    pos_traj_old = conditioned_famp.get_traj_pos()
    prediction = pos_traj_old[time_idx].unsqueeze(1)

    # get noise needed for numerical stability
    noise = torch.eye(conditioning.shape[0])
    noise[-num_dof_force:, -num_dof_force:] *= reg

    force_indices = []
    abs_force_differences = torch.abs(conditioning[num_dof_pos:].squeeze() - current_trajectory[time_idx][num_dof_pos:])
    sum_abs_force_differences = torch.sum(abs_force_differences)
    if sum_abs_force_differences >= scaled_force_threshold:
        sorted_indices = torch.argsort(abs_force_differences, descending=True)
        force_sum = 0
        # select the forces with the highest deviation until the force_sum_threshold is satisfied
        for idx in sorted_indices.tolist():
            force_indices.append(idx + num_dof_pos)
            force_sum += abs_force_differences[idx].item()
            if force_sum >= force_sum_threshold * sum_abs_force_differences:
                break
    else: # no conditioning needed
        return conditioned_famp

    conditioned_famp.compute_intermediate_terms_multi_dof()
    pos_H = conditioned_famp.pos_H_multi * conditioned_famp.weights_goal_scale.repeat(conditioned_famp.num_dof)
    relevant_columns = list(range(time_idx, pos_H.shape[0], times.shape[0]))

    pos_H_t = torch.zeros_like(pos_H[relevant_columns])
    pos_H_t[force_indices] = pos_H[relevant_columns][force_indices]

    kalman_gain = weight_cov_old @ pos_H_t.T @ torch.linalg.inv(pos_H_t @ weight_cov_old @ pos_H_t.T + noise)

    weight_mean_new = weight_mean_old + kalman_gain @ (conditioning - prediction)
    weight_cov_new = weight_cov_old - kalman_gain @ pos_H_t @ weight_cov_old

    params_new = weight_mean_new.squeeze()
    params_L_new = _cholesky(weight_cov_new)

    conditioned_famp.update_inputs(
        times=times,
        params=params_new,
        params_L=params_L_new,
        init_time=conditioned_famp.init_time,
        init_pos=conditioned_famp.init_pos,
        init_vel=conditioned_famp.init_vel
    )

    return conditioned_famp

def blend_trajectories(traj_old: torch.Tensor,
                       traj_new: torch.Tensor,
                       blend_time_idx: int,
                       time_scale: float,
                       estimated_replanning_time: int = 100,#ms
                       blend_duration: int = 1000,#ms
                       sigmoid_steepness: int = 20
                       ) -> torch.Tensor:
    """blends from one trajectory to another

    Args:
        traj_old (torch.Tensor): the initial trajectory
        traj_new (torch.Tensor): the trajectory to blend to
        blend_time_idx (int): the time index at which to start blending
        time_scale (float): the time scale of the trajectories in ms (depends on desired execution speed)
        estimated_replanning_time (int, optional): time offset to compensate trajectory replanning (in ms). Defaults to 100.
        blend_duration (int, optional): desired duration for blending in ms. Defaults to 1000.
        sigmoid_steepness (int, optional): sigmoid parameter. Defaults to 20.

    Returns:
        torch.Tensor: trajectory after blending
    """
    # rescale blend_duration and estimated_replanning_time
    blend_duration = int(blend_duration / time_scale) 
    estimated_replanning_time = int(estimated_replanning_time / time_scale)
    # get start and end index
    blend_start_idx = blend_time_idx + estimated_replanning_time
    blend_end_idx = blend_start_idx + blend_duration
    # if blending is no longer possible just return the old trajectory
    if blend_end_idx > traj_new.shape[0]:
        return traj_old
    # get sigmoid
    x = torch.linspace(0,blend_duration -1, blend_duration)
    c = (blend_duration -1) / 2
    sigmoid = 1 / (1 + torch.exp(-sigmoid_steepness/blend_duration * (x - c))).unsqueeze(1)
    # get blend factos
    blend_factor_old = torch.cat((
        torch.ones(blend_start_idx,1),
        torch.ones_like(sigmoid) - sigmoid,
        torch.zeros(traj_old.shape[0] - blend_end_idx,1)
    ), dim = 0)
    blend_factor_new = torch.cat((
        torch.zeros(blend_start_idx,1),
        sigmoid,
        torch.ones(traj_new.shape[0] - blend_end_idx,1)
    ), dim = 0)
    # combined trajectory
    alpha_old = blend_factor_old.repeat(1,traj_old.shape[1])
    alpha_new = blend_factor_new.repeat(1,traj_new.shape[1])
    traj_combined = traj_old * alpha_old + traj_new * alpha_new

    return traj_combined

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
    params_L = _cholesky(cov_matrix)
    return mean_params, params_L

def _cholesky(matrix: torch.Tensor,
              init_jitter: float =  1e-12,
              max_iterations: int = 10
              )-> torch.Tensor:
    """calculates the Cholesky decomposition of a matrix.
    Positive-definiteness is enforced by adding jitter to the diagonal.

    Args:
        matrix (torch.Tensor): matrix to be decomposed
        init_jitter (float, optional): jitter to be added in the first iteration. Defaults to 1e-12.
        max_iterations (int, optional): max number of iterations trying to achieve positive-definiteness. Defaults to 10.

    Returns:
        torch.Tensor: the lower triangular Cholesky decomposition
    """
    jitter = init_jitter
    for _ in range(max_iterations):
        try:
            cholesky = torch.linalg.cholesky(matrix)
            return cholesky
        except:
            matrix += jitter * torch.eye(matrix.shape[0])
            jitter *= 10
    raise ValueError('cholesky could not be calculated')