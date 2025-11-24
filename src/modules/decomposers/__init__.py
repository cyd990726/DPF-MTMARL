REGISTRY = {}

from .sc2_decomposer import SC2Decomposer
from .mpe_decomposer import MPEDecomposer
REGISTRY["sc2_decomposer"] = SC2Decomposer
REGISTRY["mpe_decomposer"] = MPEDecomposer
