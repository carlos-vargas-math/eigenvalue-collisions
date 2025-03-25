from dataclasses import dataclass
from random_matrix_model import points_on_curve
from random_matrix_model import initial_matrix_writter

@dataclass
class Settings:
    dim: int = 10
    distribution: object = initial_matrix_writter.Distribution.COMPLEX_GAUSSIAN
    remove_trace: bool = True
    curve: object = points_on_curve.Curve.CIRCLE
    seed: int = 1000
    seed_end: int = 1000
    grid_m: int = 10
    s_steps: int = 1000
    t_steps: int = 1000

settings = Settings()
