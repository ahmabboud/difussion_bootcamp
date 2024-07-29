import random
from typing import Optional

import numpy as np
import torch


def set_all_random_seeds(seed: Optional[int] = 42) -> None:
    """Set seeds for python random, numpy random, and pytorch random.

    Will no-op if seed is `None`.

    Args:
        seed (int): The seed value to be used for random number generators. Default is 42.
    """
    if seed is None:
        print("No seed provided. Using random seed.")
    else:
        print(f"Setting seed to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def unset_all_random_seeds() -> None:
    """
    Set random seeds for Python random, NumPy, and PyTorch to None. Running this function would undo
    the effects of set_all_random_seeds.
    """
    print("Setting all random seeds to None.")
    random.seed(None)
    np.random.seed(None)
    torch.seed()