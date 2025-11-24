REGISTRY = {}

# normal agents
from .rnn_agent import RNNAgent
from .sota_agent import SotaAgent
from .sotax_agent import SotaXAgent
from .sota_mpe_agent import SotaMPEAgent
from .sotax_mpe_agent import SotaXMPEAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["sota"] = SotaAgent
REGISTRY["sotax"] = SotaXAgent
REGISTRY["sota_mpe"] = SotaMPEAgent
REGISTRY["sotax_mpe"] = SotaXMPEAgent


# multi-task agents
from .multi_task import AllyUnionRNNAgent as MultiTaskAllyUnionRNNAgent
from .multi_task import SotaXAgent as MultiTaskSotaXAgent
from .multi_task import SotaAgent as MultiTaskSotaAgent
from .multi_task import SotaXMPEAgent as MultiTaskSotaXMPEAgent
from .multi_task import SotaMPEAgent as MultiTaskSotaMPEAgent
from .multi_task import ODISAgent
from .multi_task import UPDeTAgent


REGISTRY["mt_ally_union_rnn"] = MultiTaskAllyUnionRNNAgent
REGISTRY["mt_sotax"] = MultiTaskSotaXAgent
REGISTRY["mt_sota"] = MultiTaskSotaAgent
REGISTRY["mt_sotax_mpe"] = MultiTaskSotaXMPEAgent
REGISTRY["mt_sota_mpe"] = MultiTaskSotaMPEAgent
REGISTRY["mt_odis"] = ODISAgent
REGISTRY["mt_updet"] = UPDeTAgent