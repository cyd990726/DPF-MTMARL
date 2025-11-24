from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os

from .grid_mpe import GridMPEEnv
from .grid_mpe import EasyGridMPEEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["grid_mpe"] = partial(env_fn, env=GridMPEEnv)
REGISTRY["easy_grid_mpe"] = partial(env_fn, env=EasyGridMPEEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
