import os
from pathlib import Path

import numpy as np
import torch


def create_workdir(workdir, subdir) -> None:

    if isinstance(workdir, Path):
        workdir.joinpath(subdir).mkdir(parents=True, exist_ok=True)
    else:
        os.makedirs(os.path.join(workdir, subdir), exist_ok=True)


def set_seed(seed: int):
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)