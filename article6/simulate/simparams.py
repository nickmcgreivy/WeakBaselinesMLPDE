from dataclasses import dataclass

from rungekutta import FUNCTION_MAP
from flux import Flux


@dataclass
class CoreParams:
    Lx: float
    fluxstr: str
    nu: float

    def __post_init__(self):
        self.flux = Flux(self.fluxstr)


@dataclass
class SimulationParams:
    name: str
    cfl_safety: float
    rk: str

    def __post_init__(self):
        self.rk_fn = FUNCTION_MAP[self.rk]
