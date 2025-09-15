from .leggedrobot_terms import *
from .leggedrobot_terms import __all__ as all

__all__ = all + [
    "G1BaseHeightPenalty",
]


### ---- Reward Terms ---- ###
class G1BaseHeightPenalty(BaseHeightPenalty):
    target_height = 0.75
