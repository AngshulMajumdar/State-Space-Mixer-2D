import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic():
    """Every test sees the same random inputs on every run."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    yield
