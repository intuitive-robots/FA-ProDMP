import torch
from copy import deepcopy
import sys

from utils import cholesky

from mp_pytorch.mp import ProDMP

def force_condition(faprodmp: ProDMP, 
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
        faprodmp (ProDMP): the base FA-ProDMP
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
    conditioned_faprodmp = deepcopy(faprodmp)
    times = conditioned_faprodmp.times

    num_dof_force = scaled_measured_forces.shape[0]
    num_dof_pos = conditioned_faprodmp.num_dof - num_dof_force
    weight_mean_old = conditioned_faprodmp.params.unsqueeze(1)
    weight_cov_old = conditioned_faprodmp.params_L @ conditioned_faprodmp.params_L.T

    # initialize conditioning vector
    conditioning = torch.cat((
        torch.ones(num_dof_pos,1) * sys.maxsize, # position will not be conditioned
        scaled_measured_forces
        ), dim = 0)

    # get prediction
    pos_traj_old = conditioned_faprodmp.get_traj_pos()
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
        return conditioned_faprodmp

    conditioned_faprodmp.compute_intermediate_terms_multi_dof()
    pos_H = conditioned_faprodmp.pos_H_multi * conditioned_faprodmp.weights_goal_scale.repeat(conditioned_faprodmp.num_dof)
    relevant_columns = list(range(time_idx, pos_H.shape[0], times.shape[0]))

    pos_H_t = torch.zeros_like(pos_H[relevant_columns])
    pos_H_t[force_indices] = pos_H[relevant_columns][force_indices]

    kalman_gain = weight_cov_old @ pos_H_t.T @ torch.linalg.inv(pos_H_t @ weight_cov_old @ pos_H_t.T + noise)

    weight_mean_new = weight_mean_old + kalman_gain @ (conditioning - prediction)
    weight_cov_new = weight_cov_old - kalman_gain @ pos_H_t @ weight_cov_old

    params_new = weight_mean_new.squeeze()
    params_L_new = cholesky(weight_cov_new)

    conditioned_faprodmp.update_inputs(
        times=times,
        params=params_new,
        params_L=params_L_new,
        init_time=conditioned_faprodmp.init_time,
        init_pos=conditioned_faprodmp.init_pos,
        init_vel=conditioned_faprodmp.init_vel
    )

    return conditioned_faprodmp