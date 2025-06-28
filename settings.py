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
    dim: int = 8
    distribution: object = Distribution.COMPLEX_GAUSSIAN
    remove_trace: bool = False
    curve: object = Curve.CIRCLE
    seed: int = 2008
    seed_end: int = 2008

    grid_m: int = 10
    s_steps: int = 2000
    t_steps: int = 1000

    # the parameter a in [0,1] interpolates from C to C*, as aC* + sqrt(1-a^2)C.
    # it only works for Curve.Ellipse and Distribution.ELLIPTIC_LAW
    # the hermitian case corresponds to a = 0.5 (the grid search does not work in Hermitian cases!)
    a: float = 0

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
