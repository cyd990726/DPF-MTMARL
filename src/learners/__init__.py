REGISTRY = {}

# normal learner
from .q_learner import QLearner
from .dc_learner import DCLearner
from .trans_learner import TransLearner
from .xtrans_learner import XTransLearner

REGISTRY["q_learner"] = QLearner
REGISTRY["dc_learner"] = DCLearner
REGISTRY["trans_learner"] = TransLearner
REGISTRY["xtrans_learner"] = XTransLearner


# some multi-task learner
from .multi_task import TransLearner as MultiTaskTransLearner
from .multi_task import XTransLearner as MultiTaskXTransLearner
from .multi_task import ODISLearner
from .multi_task import UPDeTLearner
from .multi_task import LatentQLearner
REGISTRY["mt_trans_learner"] = MultiTaskTransLearner
REGISTRY["mt_xtrans_learner"] = MultiTaskXTransLearner
REGISTRY["odis_learner"] = ODISLearner
REGISTRY["updet_learner"] = UPDeTLearner
REGISTRY["latent_q_learner"] = LatentQLearner

from .multi_task import XDistralLearner as MultiTaskXDistraLeaner
from .multi_task import OriginalXDistralLearner
from .multi_task import XDistralWLearner as MultiTaskXDistraWLeaner
REGISTRY["mt_xdistral_learner"] = MultiTaskXDistraLeaner
REGISTRY["original_xdistral_learner"] = OriginalXDistralLearner
REGISTRY["mt_xdistral_weight_learner"] = MultiTaskXDistraWLeaner