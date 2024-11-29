# Force-Aware ProDMP
This repository provides a utility for fast implementation of the Force-Aware ProDMPs.
It provides the following 3 commands:
- `get_FAProDMP` to generate a FA-ProDMP from a given set of demonstrations
- `condition_FAProDMP_on_force` to condition the FA-ProDMP
- `blend_trajectories` to blend between 2 trajectories

## Expected Data Format
The utility expects each demonstration in the form of a Pandas DataFrame.
Each time step should contain positional and force information.
Additionally, we assume that the DataFrame is indexed on the time information.

## Dependency Requirements
This utility depends on the following packages:
- pandas
- numpy
- torch
- MP_PyTorch (preferably using the submodule provided)
    - matplotlib

It should be used with Python 3.9.0.
A conda configuration is provided in `conda_env.yml`.

## Citation
If you interest this project and use it in a scientific publication, we would appreciate citations to the following information:
```markdown
@misc{lödige2024useforcebot,
      title={Use the Force, Bot! -- Force-Aware ProDMP with Event-Based Replanning}, 
      author={Paul Werner Lödige and Maximilian Xiling Li and Rudolf Lioutikov},
      year={2024},
      eprint={2409.11144},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.11144}, 
}

```