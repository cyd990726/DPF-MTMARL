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
from .multi_task import XTransLearner as MultiTaskXTransLearner
REGISTRY["mt_xtrans_learner"] = MultiTaskXTransLearner

from .multi_task import XDistralLearner as MultiTaskXDistraLeaner
REGISTRY["mt_xdistral_learner"] = MultiTaskXDistraLeaner