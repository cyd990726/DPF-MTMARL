import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils.embed import polynomial_embed, binary_embed

class SotaMPEAgent(nn.Module):
    def __init__(self, input_shape_info, decomposer, args):
        super(SotaMPEAgent, self).__init__()
        self.last_action_shape = input_shape_info["last_action_shape"]
        self.agent_id_shape = input_shape_info["agent_id_shape"]
        self.args = args
    
        #### define various dimension information
        ## set attributes
        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        ## get obs shape information
        self.decomposer = decomposer
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al        
        n_actions = decomposer.n_actions
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions
        ## get agent num information
        self.n_agents = args.n_agents
        self.n_enemy = self.decomposer.n_landmarks
        self.n_ally = args.n_agents - 1
        self.n_entity = self.n_agents + self.n_enemy

        #### define various networks
        ## networks for attention
        self.query = nn.Linear(wrapped_obs_own_dim, self.attn_embed_dim)
        self.ally_key = nn.Linear(obs_al_dim, self.attn_embed_dim)
        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim)
        self.enemy_key = nn.Linear(obs_en_dim, self.attn_embed_dim)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)

        ## network for computing action Q value
        self.rnn = nn.GRUCell(self.entity_embed_dim * 3, args.rnn_hidden_dim)
        self.action_layer = nn.Linear(args.rnn_hidden_dim, n_actions)

    def init_hidden(self):
        # make hidden states on the same device as model
        return self.action_layer.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # decompose inputs into observation inputs, last_action_info, agent_id_info
        obs_dim = self.decomposer.obs_dim
        obs_inputs, last_action_inputs, agent_id_inputs = inputs[:, :obs_dim], \
            inputs[:, obs_dim:obs_dim+self.last_action_shape], inputs[:, obs_dim+self.last_action_shape:]

        # decompose observation input
        own_obs, enemy_feats, ally_feats = self.decomposer.decompose_obs(obs_inputs)    # own_obs: [bs*self.n_agents, own_obs_dim]
        bs = int(own_obs.shape[0]/self.n_agents)

        # embed agent_id inputs and decompose last_action_inputs
        agent_id_inputs = [th.as_tensor(binary_embed(i + 1, self.args.id_length, self.args.max_agent), dtype=own_obs.dtype) for i in range(self.n_agents)]
        agent_id_inputs = th.stack(agent_id_inputs, dim=0).repeat(bs, 1).to(own_obs.device)

        # incorporate agent_id embed and compact_action_states
        own_obs = th.cat([own_obs, agent_id_inputs, last_action_inputs], dim=-1)

        enemy_feats = th.stack(enemy_feats, dim=0)
        ally_feats = th.stack(ally_feats, dim=0)

        # compute key, query and value for attention
        own_hidden = self.own_value(own_obs)
        query = self.query(own_obs)
        ally_keys = self.ally_key(ally_feats).permute(1, 2, 0)
        enemy_keys = self.enemy_key(enemy_feats).permute(1, 2, 0)
        ally_values = self.ally_value(ally_feats).permute(1, 0, 2)
        enemy_values = self.enemy_value(enemy_feats).permute(1, 0, 2)

        # do attention
        ally_hidden = self.attention(query, ally_keys, ally_values, self.attn_embed_dim)
        enemy_hidden = self.attention(query, enemy_keys, enemy_values, self.attn_embed_dim)        
        tot_hidden = th.cat([own_hidden, ally_hidden, enemy_hidden], dim=-1)

        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(tot_hidden, h_in)

        # compute action_q
        q = self.action_layer(h)
        
        return q, h

    def attention(self, q, k, v, attn_dim):
        """
            q: [bs*n_agents, attn_dim]
            k: [bs*n_agents, attn_dim, n_entity]
            v: [bs*n_agents, n_entity, value_dim]
        """
        energy = th.bmm(q.unsqueeze(1)/(attn_dim ** (1/2)), k)
        attn_score = F.softmax(energy, dim=-1) 
        attn_out = th.bmm(attn_score, v).squeeze(1)
        return attn_out
