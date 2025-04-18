from dataclasses import dataclass
from enum import Enum

class Curve(Enum):
    ZERO = 1
    CIRCLE = 2
    CIRCUIT = 3
    CROSSING = 4
    ELLIPSE = 5

class Distribution(Enum):
    BERNOULLI = "bernoulli"
    COMPLEX_GAUSSIAN = "complexGaussian"
    REAL_GAUSSIAN = "realGaussian"
    GINIBRE_MEANDER = "ginibreMeander"
    OPPOSING_SECTORS = "opposingSectors"
    ELLIPTIC_LAW = "elipticLaw"

@dataclass
class Settings:
    dim: int = 10
    distribution: object = Distribution.ELLIPTIC_LAW
    remove_trace: bool = False
    curve: object = Curve.ELLIPSE
    seed: int = 2006
    seed_end: int = 2006
    grid_m: int = 6
    s_steps: int = 2000
    t_steps: int = 1000
    a: float = 0.025

settings = Settings()

def generate_directory_name():
    dim = settings.dim
    distribution = settings.distribution
    remove_trace = settings.remove_trace
    curve = settings.curve
    curve_str = str(curve)
    if settings.curve == Curve.ELLIPSE:
        curve_str += "a=" + str(settings.a) 
    seed = settings.seed
    summary_name = "computed_examples/N=" + str(dim) + "&" + curve_str + "&Seed=" + str(seed) + "&" + str(distribution) + "&Traceless=" + str(remove_trace)
    return summary_name
