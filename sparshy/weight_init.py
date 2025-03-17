from enum import Enum
from typing import Tuple
import math
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Dictionary mapping initialization strategies
initialization_methods = locals()

def random(dimensions: Tuple[int, int]) -> np.ndarray:
    return np.random.randn(*dimensions)

def xavier(shape: Tuple[int, int]) -> np.ndarray:
    input_size, output_size = shape
    scale_factor = math.sqrt(2.0 / (input_size + output_size))
    return scale_factor * np.random.randn(input_size, output_size)

def he(shape: Tuple[int, int]) -> np.ndarray:
    input_nodes = shape[0]
    std_dev = math.sqrt(2.0 / input_nodes)
    return std_dev * np.random.randn(*shape)

class WeightInit(str, Enum):
    random = "random"
    xavier = "xavier"
    he = "he"

    def __call__(self, *args):
        return initialization_methods[self.value](*args)
