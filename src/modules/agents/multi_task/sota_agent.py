import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils.embed import polynomial_embed, binary_embed


class SotaAgent(nn.Module):
    """  sota agent for multi-task learning """
    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(SotaAgent, self).__init__()
        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in task2input_shape_info}
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args

        #### define various dimension information
        ## set attributes
        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        ## get obs shape information
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al        
        n_actions_no_attack = decomposer.n_actions_no_attack
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
        ## enemy_obs ought to add attack_action_info
        obs_en_dim += 1

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
        self.wo_action_layer = nn.Linear(args.rnn_hidden_dim, n_actions_no_attack)
        ## attack action networks
        self.enemy_embed = nn.Sequential(
            nn.Linear(obs_en_dim, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        )
        self.attack_action_layer = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim + args.rnn_hidden_dim, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, 1)
        )

    def init_hidden(self):
        # make hidden states on the same device as model
        return self.wo_action_layer.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, task):
        # get decomposer, last_action_shape and n_agents of this specific task
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]    

        # decompose inputs into observation inputs, last_action_info, agent_id_info
        obs_dim = task_decomposer.obs_dim
        obs_inputs, last_action_inputs, agent_id_inputs = inputs[:, :obs_dim], \
            inputs[:, obs_dim:obs_dim+last_action_shape], inputs[:, obs_dim+last_action_shape:]

        # decompose observation input
        own_obs, enemy_feats, ally_feats = task_decomposer.decompose_obs(obs_inputs)    # own_obs: [bs*self.n_agents, own_obs_dim]
        bs = int(own_obs.shape[0]/task_n_agents)

        # embed agent_id inputs and decompose last_action_inputs
        agent_id_inputs = [th.as_tensor(binary_embed(i + 1, self.args.id_length, self.args.max_agent), dtype=own_obs.dtype) for i in range(task_n_agents)]
        agent_id_inputs = th.stack(agent_id_inputs, dim=0).repeat(bs, 1).to(own_obs.device)
        _, attack_action_info, compact_action_states = task_decomposer.decompose_action_info(last_action_inputs)

        # incorporate agent_id embed and compact_action_states
        own_obs = th.cat([own_obs, agent_id_inputs, compact_action_states], dim=-1)

        # incorporate attack_action_info into enemy_feats
        attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
        enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1) 
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

        # compute wo_action_q
        wo_action_q = self.wo_action_layer(h)
        
        # compute action_q
        enemy_hidden = self.enemy_embed(enemy_feats)
        attack_action_input = th.cat([enemy_hidden, h.unsqueeze(0).repeat(enemy_feats.size(0), 1, 1)], dim=-1)
        attack_action_q = self.attack_action_layer(attack_action_input).squeeze(-1).transpose(0, 1)
        
        q = th.cat([wo_action_q, attack_action_q], dim=-1)

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
