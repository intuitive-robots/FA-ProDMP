import torch

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