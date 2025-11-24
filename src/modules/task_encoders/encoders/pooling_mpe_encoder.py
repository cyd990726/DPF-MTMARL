from re import S
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import numpy as np
import torch as th

class PoolingMPEEncoder(nn.Module):
    """
        dynamic-dim inputs -->> fixed-dim outputs
    """

    def __init__(self, decomposer, args):
        super(PoolingMPEEncoder, self).__init__()
        self.args = args
        self.decomposer = decomposer
        self.task_repre_dim = args.task_repre_dim
        self.state_latent_dim = args.state_latent_dim

        #### part about obs
        ## fetch some obs shape info from decomposer
        self.entity_embed_dim = args.entity_embed_dim
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        self.n_actions = decomposer.n_actions
        
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + self.n_actions

        ## get agent num info
        self.n_agents = args.n_agents
        self.n_enemy = decomposer.n_landmarks
        self.n_ally = self.n_agents - 1
        self.n_entity = self.n_agents + self.n_enemy

        ## define obs-related network
        self.obs_own_embed = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)
        self.obs_ally_embed = nn.Linear(obs_al_dim, self.entity_embed_dim)
        self.obs_enemy_embed = nn.Linear(obs_en_dim, self.entity_embed_dim)
        
        #### part about state
        ## get state shape info from decomposer
        state_nf_al, state_nf_en = decomposer.state_nf_al, decomposer.state_nf_en

        ## define state-related network
        self.state_ally_embed = nn.Linear(state_nf_al, self.entity_embed_dim)
        self.state_enemy_embed = nn.Linear(state_nf_en, self.entity_embed_dim)

        self.hypernet = nn.Linear(self.task_repre_dim, self.entity_embed_dim * 2 * self.state_latent_dim)
        
    def forward(self, obs, state, actions, task_repre):
        #### check shape valid
        if len(obs.shape) != 2:
            assert len(obs.shape) == 3, f"Invalid obs shape {obs.shape}"
            bs = obs.shape[0]
            obs = obs.reshape(bs*obs.shape[1], obs.shape[2])
        else:
            bs = obs.shape[0]//self.n_agnets
        
        if len(actions.shape) != 2:
            assert len(actions.shape) == 3, f"Invalid actions shape {actions.shape}"
            actions = actions.reshape(bs*actions.shape[1], actions.shape[2])

        ## !!! suppose state.shape = [bs, state_dim]
        state = state.unsqueeze(1)  # [bs, state_dim] -> [bs, 1, state_dim]
        
        #### part about obs
        ## decompose observation input
        own_obs, enemy_feats, ally_feats = self.decomposer.decompose_obs(obs)    # own_obs: [bs*self.n_agents, own_obs_dim]

        ## incorporate compact_action_states into own_obs
        own_obs = th.cat([own_obs, actions], dim=-1)

        ## incorporate attack_action_info into enemy_feats
        enemy_feats = th.stack(enemy_feats, dim=0)
        ally_feats = th.stack(ally_feats, dim=0)

        ## through embed network
        obs_own_hidden = self.obs_own_embed(own_obs)
        obs_ally_hidden = self.obs_ally_embed(ally_feats)
        obs_enemy_hidden = self.obs_enemy_embed(enemy_feats)
        
        ## mean pooling
        # obs_hidden: [bs, n_agents, dim]
        obs_hidden = th.mean(th.cat([obs_own_hidden[None, ...], obs_ally_hidden, obs_enemy_hidden]), dim=0).reshape(bs, self.n_agents, self.entity_embed_dim)

        #### part about state
        ## decompose state input
        ally_states, enemy_states = self.decomposer.decompose_state(state)
        ally_states = th.stack(ally_states, dim=0).squeeze(-2)
        enemy_states = th.stack(enemy_states, dim=0).squeeze(-2) 
        
        ## through embed network
        state_ally_hidden = self.state_ally_embed(ally_states)
        state_enemy_hidden = self.state_enemy_embed(enemy_states)
        
        ## mean pooling
        state_hidden = th.mean(th.cat([state_ally_hidden, state_enemy_hidden]), dim=0) # [bs, entity_embed_dim]

        #### final output
        ## concat state_hidden and obs_hidden
        # tot_hidden: [bs, n_agents, entity_embed_dim * 2]
        tot_hidden = th.cat([state_hidden.unsqueeze(1).repeat(1, self.n_agents, 1), obs_hidden], dim=-1)
        ## hypernet operation
        # task_repre: [n_agnets, task_repre_dim]
        # transform_weight: [n_agents, entity_embed_dim * 2, state_latent_dim]
        transform_weight = self.hypernet(task_repre).reshape(self.n_agents, self.entity_embed_dim * 2, self.state_latent_dim)
        # encoded_latent: [bs, n_agents, state_latent_dim]
        encoded_latent = th.matmul(tot_hidden.unsqueeze(-2), transform_weight.unsqueeze(0)).squeeze(-2)     

        return encoded_latent, bs           

