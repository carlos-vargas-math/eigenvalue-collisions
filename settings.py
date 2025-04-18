from dataclasses import dataclass
from random_matrix_model import points_on_curve
from random_matrix_model import initial_matrix_writter
import numpy as np

@dataclass
class Settings:
    dim: int = 10
    distribution: object = initial_matrix_writter.Distribution.ELLIPTIC_LAW
    remove_trace: bool = False
    curve: object = points_on_curve.Curve.ELLIPSE
    seed: int = 2005
    seed_end: int = 2005
    grid_m: int = 6
    s_steps: int = 2000
    t_steps: int = 1000

settings = Settings()
