from enum import Enum
import jax.numpy as np


class Flux(Enum):
    """
    Flux is a subclass of Enum, which determines the flux that is used to compute
    the time-derivative of the equation.
    """

    CENTERED = "centered"
    GODUNOV = "godunov"

    def __str__(self):
        return self.value