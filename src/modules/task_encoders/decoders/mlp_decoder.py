import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import numpy as np
import torch as th


class MLPDecoder(nn.Module):
    def __init__(self, args):
        super(MLPDecoder, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.state_dim = int(np.prod(args.state_shape))
        self.obs_dim = int(np.prod(args.obs_shape))
        self.reward_dim = 1

        self.obs_decoder = nn.Sequential(
            nn.Linear(args.state_latent_dim, args.state_latent_dim),
            nn.ReLU(),
            nn.Linear(args.state_latent_dim, self.obs_dim),
        )
        self.state_decoder = nn.Sequential(
            nn.Linear(args.state_latent_dim, args.state_latent_dim),
            nn.ReLU(),
            nn.Linear(args.state_latent_dim, self.state_dim),
        )
        self.reward_decoder = nn.Sequential(
            nn.Linear(args.state_latent_dim, args.state_latent_dim),
            nn.ReLU(),
            nn.Linear(args.state_latent_dim, self.reward_dim),
        )
        # define surrogate encoder        
        # self.surrogate_encoder = nn.Sequential(
        #     nn.Linear(self.obs_dim + self.state_dim + self.reward_dim, args.task_repre_dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(args.task_repre_dim * 2, args.task_repre_dim * 2),
        # )

    def forward(self, encoded_latent, bs):
        # notice: we get encoded_latent outside

        # get decoder inputs
        # dist = D.Normal(task_repre_mu, task_repre_sigma)
        # task_repres_input = dist.rsample((bs,)).to(encoded_latent.device)
        # decoder_inputs = th.cat([encoded_latent, task_repres_input], dim=-1)

        # forward obs_decoder, state_decoder, reward_decoder
        next_obs = self.obs_decoder(encoded_latent)
        next_state = self.state_decoder(encoded_latent)
        reward = self.reward_decoder(encoded_latent)

        return next_obs, next_state, reward        

    # def compute_mi_loss(self, next_obs, next_state, reward, task_repre_mu, task_repre_sigma):
    #     """
    #         This function compute the lower bound of MI, and return the opposite number of lower bound as loss
    #     """
    #     # get surrogate gaussian distribution
    #     surrogate_input = th.cat([next_obs, next_state[:, None, :].repeat(1, self.args.n_agents, 1), reward[:, None, :].repeat(1, self.args.n_agents, 1)], dim=-1)
    #     surrogate_output = self.surrogate_encoder(surrogate_input)
    #     surrogate_mu, surrogate_sigma = surrogate_output[:, :, :self.args.task_repre_dim], th.exp(surrogate_output[:, :, self.args.task_repre_dim:] * 0.5)
        
    #     # compute KL divergence
    #     dist_p = D.MultivariateNormal(task_repre_mu.to(surrogate_mu.device), th.diag_embed(task_repre_sigma).to(surrogate_sigma.device))
    #     dist_q = D.MultivariateNormal(surrogate_mu, th.diag_embed(surrogate_sigma))
    #     kl_losses = D.kl.kl_divergence(dist_p, dist_q)
    #     return kl_losses

##### utils function about config

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
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
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "default.yaml"), "r") as f:
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
    config_dict['n_agents'] = 3
    config_dict['n_actions'] = 6 + 3
    config_dict['state_shape'] = 10
    config_dict['obs_shape'] = 6
    
    sys.path.append("/home/chenfeng/chenf/transfer/src")
    from utils.dict2namedtuple import convert
    args = convert(config_dict)

    ##### test forwardmodel
    
    # define model
    forward_model = ForwardModelEncoder(args)
    
    # fake input
    bs, max_seq_len = 4, 10
    obs, state, actions = \
        th.as_tensor(np.random.randn(bs, args.n_agents, args.obs_shape)).float(), \
        th.as_tensor(np.random.randn(bs, args.state_shape)).float(), \
        th.as_tensor(np.random.randn(bs, args.n_agents, args.n_actions)).float()
    
    # task_repre to learn
    mu, sigma = th.zeros(args.n_agents, args.task_repre_dim, requires_grad=True), th.ones(args.n_agents, args.task_repre_dim, requires_grad=True)
    
    # forward
    out = forward_model(obs, state, actions, mu, sigma)

    # compute loss
    print(out)