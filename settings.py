from dataclasses import dataclass
from random_matrix_model import points_on_curve

@dataclass
class Settings:
    dim: int = 10
    distribution: str = 'complexGaussian'
    remove_trace: bool = True
    curve: object = points_on_curve.Curve.CIRCLE
    seed: int = 1000
    seed_end: int = 1000
    grid_m: int = 20
    s_steps: int = 1000
    t_steps: int = 1000

settings = Settings()
