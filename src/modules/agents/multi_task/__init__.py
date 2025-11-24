from .ally_union_rnn_agent import AllyUnionRNNAgent
from .sotax_agent import SotaXAgent
from .sota_agent import SotaAgent
from .sota_mpe_agent import SotaMPEAgent
from .sotax_mpe_agent import SotaXMPEAgent
from .odis_agent import ODISAgent
from .updet_agent import UPDeTAgent

REGISTRY = {}

REGISTRY["mt_odis"] = ODISAgent
REGISTRY["mt_updet"] = UPDeTAgent
