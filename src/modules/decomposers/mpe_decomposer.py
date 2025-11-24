from smac.env.multiagentenv import MultiAgentEnv
from smac.env.starcraft2.maps import get_map_params

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import enum
import numpy as np


actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

task_id2config = {
    "2": {
        "n_agents": 2,
        "field_size": [6,6],
        "sight": 5,
        "episode_limit": 50,
        "reach_range": 2,
    },
    "3": {
        "n_agents": 3,
        "field_size": [8,8],
        "sight": 7,
        "episode_limit": 50,
        "reach_range": 2,
    },
    "4": {
        "n_agents": 4,
        "field_size": [10,10],
        "sight": 9,
        "episode_limit": 50,
        "reach_range": 2,
    },
    "5": {
        "n_agents": 5,
        "field_size": [10,10],
        "sight": 9,
        "episode_limit": 50,
        "reach_range": 2,
    },
    "6": {
        "n_agents": 6,
        "field_size": [15,15],
        "sight": 14,
        "episode_limit": 50,
        "reach_range": 2,
    },
    "7": {
        "n_agents": 7,
        "field_size": [15,15],
        "sight": 14,
        "episode_limit": 70,
        "reach_range": 2,
    },
    "8": {
        "n_agents": 8,
        "field_size": [15,15],
        "sight": 14,
        "episode_limit": 70,
        "reach_range": 2,
    },
    "9": {
        "n_agents": 9,
        "field_size": [15,15],
        "sight": 14,
        "episode_limit": 70,
        "reach_range": 2,
    },
    "10": {
        "n_agents": 10,
        "field_size": [15,15],
        "sight": 14,
        "episode_limit": 70,
        "reach_range": 2,
    },
    "11": {
        "n_agents": 11,
        "field_size": [18, 18],
        "sight": 17,
        "episode_limit": 70,
        "reach_range": 2,
    },
    "12": {
        "n_agents": 12,
        "field_size": [18, 18],
        "sight": 17,
        "episode_limit": 70,
        "reach_range": 2,
    },
    "15": {
        "n_agents": 15,
        "field_size": [20,20],
        "sight": 19,
        "episode_limit": 80,
        "reach_range": 2,
    }
}


class MPEDecomposer:
    def __init__(self, args):
        # Load map params
        self.args = args
        self.task_id = args.env_args["task_id"]
        task_config = task_id2config[str(self.task_id)]
        self.n_agents = task_config["n_agents"]
        if args.env == "easy_grid_mpe":
            self.n_landmarks = 1    
        else:
            self.n_landmarks = self.n_agents
        self.episode_limit = task_config["episode_limit"]
        self.field_size = task_config["field_size"]
        self.n_actions = 5
        
        self.own_feats, self.ally_feats, self.enemy_feats, self.obs_nf_en, self.obs_nf_al = \
            self.get_obs_size()
        self.own_obs_dim = self.own_feats
        self.obs_dim = self.own_obs_dim + self.enemy_feats + self.ally_feats
        
        self.enemy_state_dim, self.ally_state_dim, self.state_nf_en, self.state_nf_al = \
            self.get_state_size()
        self.state_dim = self.enemy_state_dim + self.ally_state_dim

    def get_state_size(self):
        nf_al = nf_en = 2
        enemy_state = self.n_landmarks * nf_en
        ally_state = self.n_agents * nf_al
        
        return enemy_state, ally_state, nf_en, nf_al

    def get_obs_size(self):
        nf_al = nf_en = 2
        own_feats = 2
        enemy_feats = self.n_landmarks * nf_en
        ally_feats = (self.n_agents - 1) * nf_al

        return own_feats, ally_feats, enemy_feats, nf_en, nf_al

    def decompose_state(self, state_input):
        # state_input = [ally_state, enemy_state]
        # assume state_input.shape == [batch_size, seq_len, state]
        
        ally_states = [state_input[:, :, i * self.state_nf_al:(i + 1) * self.state_nf_al] for i in range(self.n_agents)]
        base = self.n_agents * self.state_nf_al
        enemy_states = [state_input[:, :, base + i * self.state_nf_en:base + (i + 1)*self.state_nf_en] for i in range(self.n_landmarks)]
        return ally_states, enemy_states

    def decompose_obs(self, obs_input):
        own_feats = obs_input[:, :self.own_feats]
        base = self.own_feats
        ally_feats = [obs_input[:, base + i * self.obs_nf_al:base + (i + 1) * self.obs_nf_al] for i in range(self.n_agents - 1)]
        base += self.obs_nf_al * (self.n_agents - 1)
        enemy_feats = [obs_input[:, base + i * self.obs_nf_en:base + (i + 1) * self.obs_nf_en] for i in range(self.n_landmarks)]
        return own_feats, enemy_feats, ally_feats