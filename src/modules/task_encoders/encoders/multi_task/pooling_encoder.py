import torch.nn as nn

import numpy as np
import torch as th


class PoolingEncoder(nn.Module):
    """
        dynamic-dim inputs -->> fixed-dim outputs
    """

    def __init__(self, task2decomposer, task2n_agents, decomposer, args):
        super(PoolingEncoder, self).__init__()
        self.args = args
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.task_repre_dim = args.task_repre_dim
        self.state_latent_dim = args.state_latent_dim

        #### part about obs
        ## fetch some obs shape info from decomposer
        self.entity_embed_dim = args.entity_embed_dim
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        self.n_actions_no_attack = decomposer.n_actions_no_attack
        
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + self.n_actions_no_attack + 1
        obs_en_dim += 1

        ## define obs-related network
        self.obs_own_embed = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)
        self.obs_ally_embed = nn.Linear(obs_al_dim, self.entity_embed_dim)
        self.obs_enemy_embed = nn.Linear(obs_en_dim, self.entity_embed_dim)
        
        #### part about state
        ## get state shape info from decomposer
        state_nf_al, state_nf_en, timestep_state_dim = decomposer.state_nf_al, decomposer.state_nf_en, decomposer.timestep_number_state_dim
        self.state_last_action, self.state_timestep_number = decomposer.state_last_action, decomposer.state_timestep_number

        ## we suppose self.state_timestep_number is False

        ## define state-related network
        if self.state_last_action:
            self.state_ally_embed = nn.Linear(state_nf_al + self.n_actions_no_attack + 1, self.entity_embed_dim)
            self.state_enemy_embed = nn.Linear(state_nf_en, self.entity_embed_dim)
            state_nf_al += self.n_actions_no_attack + 1
        else:
            self.state_ally_embed = nn.Linear(state_nf_al, self.entity_embed_dim)
            self.state_enemy_embed = nn.Linear(state_nf_en, self.entity_embed_dim)

        self.hypernet = nn.Linear(self.task_repre_dim, self.entity_embed_dim * 2 * self.state_latent_dim)

    def forward(self, obs, state, actions, task, task_repre):
        #### check shape valid
        if len(obs.shape) != 2:
            assert len(obs.shape) == 3, f"Invalid obs shape {obs.shape}"
            bs = obs.shape[0]
            obs = obs.reshape(bs*obs.shape[1], obs.shape[2])
        else:
            bs = obs.shape[0]//self.task2n_agents[task]
        
        if len(actions.shape) != 2:
            assert len(actions.shape) == 3, f"Invalid actions shape {actions.shape}"
            actions = actions.reshape(bs*actions.shape[1], actions.shape[2])

        ## !!! suppose state.shape = [bs, state_dim]
        state = state.unsqueeze(1)  # [bs, state_dim] -> [bs, 1, state_dim]
        
        #### part about obs
        ## decompose observation input
        own_obs, enemy_feats, ally_feats = self.task2decomposer[task].decompose_obs(obs)    # own_obs: [bs*self.n_agents, own_obs_dim]

        ## decompose action
        _, attack_action_info, compact_action_states = self.task2decomposer[task].decompose_action_info(actions)

        ## incorporate compact_action_states into own_obs
        own_obs = th.cat([own_obs, compact_action_states], dim=-1)

        ## incorporate attack_action_info into enemy_feats
        attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
        enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1) 
        ally_feats = th.stack(ally_feats, dim=0)

        ## through embed network
        obs_own_hidden = self.obs_own_embed(own_obs)
        obs_ally_hidden = self.obs_ally_embed(ally_feats)
        obs_enemy_hidden = self.obs_enemy_embed(enemy_feats)
        
        ## mean pooling
        # obs_hidden: [bs, n_agents, dim]
        obs_hidden = th.mean(
            th.cat([obs_own_hidden[None, ...], obs_ally_hidden, obs_enemy_hidden]),
            dim=0
        ).reshape(bs, self.task2n_agents[task], self.entity_embed_dim)

        #### part about state
        ## decompose state input
        ally_states, enemy_states, last_action_states, timestep_number_state = self.task2decomposer[task].decompose_state(state)
        ally_states = th.stack(ally_states, dim=0).squeeze(-2)
        enemy_states = th.stack(enemy_states, dim=0).squeeze(-2)
        
        ## stack action informatioun
        if self.state_last_action:
            last_action_states = th.stack(last_action_states, dim=0).squeeze(-2)
            _, _, compact_action_states = self.task2decomposer[task].decompose_action_info(last_action_states)
            ally_states = th.cat([ally_states, compact_action_states], dim=-1)

        ## through embed network
        state_ally_hidden = self.state_ally_embed(ally_states)
        state_enemy_hidden = self.state_enemy_embed(enemy_states)
        
        ## mean pooling
        state_hidden = th.mean(th.cat([state_ally_hidden, state_enemy_hidden]), dim=0) # [bs, entity_embed_dim]

        #### final output
        ## concat state_hidden and obs_hidden
        tot_hidden = th.cat([state_hidden.unsqueeze(1).repeat(1, self.task2n_agents[task], 1), obs_hidden], dim=-1)
        transform_weight = self.hypernet(task_repre).reshape(self.task2n_agents[task], self.entity_embed_dim * 2, self.state_latent_dim)
        encoded_latent = th.matmul(tot_hidden.unsqueeze(-2), transform_weight.unsqueeze(0)).squeeze(-2)

        return encoded_latent, bs           
    
##### utils function about config

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


if __name__ == "__main__":
    ##### prepare args
    from copy import deepcopy
    import collections
    import yaml
    import os, sys

    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    ##### add fake args attributes
    # # fake scenario is 5m_vs_6m
    # config_dict['n_agents'] = 5
    # config_dict['n_actions'] = 6 + 6
    # config_dict['state_shape'] = 10
    # config_dict['obs_shape'] = 6
    
    sys.path.append("/home/chenfeng/chenf/transfer/src")
    from utils.dict2namedtuple import convert
    from types import SimpleNamespace as SN
    args = SN(**config_dict)

    # define decomposer
    from modules.decomposers import REGISTRY as decomposer_REGISTRY
    decomposer =  decomposer_REGISTRY["sc2_decomposer"](args)

    args.n_agents = decomposer.n_agents
    args.n_actions = decomposer.n_actions
    args.state_shape = decomposer.state_dim
    args.obs_shape = decomposer.obs_dim

    input_shape_info = {
        "input_shape": args.obs_shape + args.n_actions + args.n_agents,
        "last_action_shape": args.n_actions,
        "agent_id_shape": args.n_agents,
    }    

    ##### test encoder

    # define model
    forward_model = PoolingEncoder(decomposer, args)
    
    # fake input
    bs, max_seq_len = 4, 10
    obs, state, actions = \
        th.as_tensor(np.random.randn(bs, args.n_agents, args.obs_shape)).float(), \
        th.as_tensor(np.random.randn(bs, args.state_shape)).float(), \
        th.as_tensor(np.random.randn(bs, args.n_agents, args.n_actions)).float()
    
    # task_repre to learn
    mu, sigma = th.zeros(args.n_agents, args.task_repre_dim, requires_grad=True), th.ones(args.n_agents, args.task_repre_dim, requires_grad=True)
    
    # forward
    out = forward_model(obs, state, actions)

    # compute loss
    print(out)