from .leggedrobot_terms import (
    BaseHeightPenalty,
)


### ---- Reward Terms ---- ###
class G1BaseHeightPenalty(BaseHeightPenalty):
    target_height = 0.75
