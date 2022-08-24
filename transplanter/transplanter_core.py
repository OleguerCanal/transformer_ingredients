import copy

import torch

def copy_weights(teacher_weights: torch.Tensor,
                 student_weights: torch.Tensor,
                 initial_weights_std: float = 0.001,
                 scaling_factor: float = 1.0,
                 base_bias: float = 0.0):
    student_weights = torch.normal(mean=base_bias, std=initial_weights_std, size=student_weights.size())
    student_weights = _overwrite_tensor(big_tensor=student_weights,
                                        small_tensor=teacher_weights)
    return student_weights*scaling_factor

def _overwrite_tensor(big_tensor: torch.Tensor,
                      small_tensor: torch.Tensor):
    s = small_tensor.shape
    if len(s) == 1:
        big_tensor[0:s[0]] = copy.deepcopy(small_tensor)
    elif len(s) == 2:
        big_tensor[0:s[0], 0:s[1]] = copy.deepcopy(small_tensor)
    elif len(s) == 3:
        big_tensor[0:s[0], 0:s[1], 0:s[2]] = copy.deepcopy(small_tensor)
    elif len(s) == 4:
        big_tensor[0:s[0], 0:s[1], 0:s[2], 0:s[3]] = copy.deepcopy(small_tensor)
    elif len(s) == 5:
        big_tensor[0:s[0], 0:s[1], 0:s[2], 0:s[3], 0:s[4]] = copy.deepcopy(small_tensor)
    return big_tensor