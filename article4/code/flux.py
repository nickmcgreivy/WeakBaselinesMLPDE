from enum import Enum
import jax.numpy as np


class Flux(Enum):
    """
    Flux is a subclass of Enum, which determines the flux that is used to compute
    the time-derivative of the equation.

    LEARNED is the data-driven discretization of the equation, called the "learned
    flux interpolation"
    """

    UPWIND = "upwind"
    CENTERED = "centered"
    LEARNED = "learned"
    VANLEER = "vanleer"
    CONSERVATION = "conservation"

    def __str__(self):
        return self.value