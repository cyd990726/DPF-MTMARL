import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils.embed import polynomial_embed, binary_embed


class AllyUnionRNNAgent(nn.Module):
    """ trans agent for multi-task learning """
    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(AllyUnionRNNAgent, self).__init__()
        self.tasks = [task for task in task2input_shape_info]
        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in self.tasks}
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args
                
        # get obs shape information
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al        
        n_actions_no_attack = decomposer.n_actions_no_attack
        task_repre_dim = args.task_repre_dim
        # get wrapped obs_own_dim
        # Notice: we consider agent_id and last_action as obs_input by default
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack
        # enemy_obs ought to add attack_action_info
        obs_en_dim += 1

        # define pairwise_net
        self.pairwise_embed_1 = nn.Sequential(
            nn.Linear(wrapped_obs_own_dim + obs_en_dim + obs_al_dim + task_repre_dim, args.pairwise_embed_dim),
            nn.ReLU(),
        )
        self.pairwise_rnn_1 = nn.GRUCell(args.pairwise_embed_dim, args.pairwise_embed_dim)
        self.pairwise_action_layer_1 = nn.Linear(args.pairwise_embed_dim, n_actions_no_attack)
        
        self.pairwise_embed_2 = nn.Sequential(
            nn.Linear(wrapped_obs_own_dim + obs_en_dim + obs_al_dim + task_repre_dim, args.pairwise_embed_dim),
            nn.ReLU(),
        )
        self.pairwise_rnn_2 = nn.GRUCell(args.pairwise_embed_dim, args.pairwise_embed_dim)
        self.pairwise_action_layer_2 = nn.Linear(args.pairwise_embed_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        attack_hidden = self.pairwise_action_layer_1.weight.new(1, self.args.pairwise_embed_dim).zero_()
        no_attack_hidden = self.pairwise_action_layer_2.weight.new(1, self.args.pairwise_embed_dim).zero_()
        return attack_hidden, no_attack_hidden

    def forward(self, inputs, attack_hidden, no_attack_hidden, task_repre_mu, task):
        # get args, decomposer and last_action_shape of this specific task        
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]
        
        # decompose inputs into observation inputs, last_action_info, agent_id_info
        obs_dim = task_decomposer.obs_dim
        obs_inputs, last_action_inputs, agent_id_inputs = inputs[:, :obs_dim], \
            inputs[:, obs_dim:obs_dim+last_action_shape], inputs[:, obs_dim+last_action_shape:]

        # decompose observation input
        own_obs, enemy_feats, ally_feats = task_decomposer.decompose_obs(obs_inputs)
        bs, n_agents, n_enemy, n_ally = int(own_obs.shape[0]/task_n_agents), task_n_agents, len(enemy_feats), len(ally_feats)
        
        # embed agent_id inputs
        agent_id_inputs = []
        for i in range(n_agents):
            agent_id_embed = th.as_tensor(binary_embed(i + 1, self.args.id_length, self.args.max_agent), dtype=own_obs.dtype)
            agent_id_inputs.append(agent_id_embed)
        agent_id_inputs = th.stack(agent_id_inputs, dim=0).unsqueeze(1).repeat(bs, n_ally*n_enemy, 1).to(own_obs.device)

        # decompose last_action_inputs
        no_attack_action_info, attack_action_info, _ = task_decomposer.decompose_action_info(last_action_inputs)
        no_attack_action_inputs = no_attack_action_info.unsqueeze(1).repeat(1, n_ally*n_enemy, 1)
        
        # incorporate attack_action_info into enemy_feats
        attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
        enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1)    
    
        # concat/combine different parts of obs
        enemy_feats = th.cat([item.unsqueeze(1) for item in enemy_feats], dim=1).repeat(1, n_ally, 1)
        ally_feats = th.cat([item.unsqueeze(1).repeat(1, n_enemy, 1) for item in ally_feats], dim=1)
        own_obs = own_obs.unsqueeze(1).repeat(1, n_ally*n_enemy, 1)
        
        # concat agent_id embed
        own_obs = th.cat([own_obs, agent_id_inputs, no_attack_action_inputs], dim=-1)

        # aggregate observation information
        obs_input = th.cat([own_obs, ally_feats, enemy_feats], dim=-1)

        # concat task repre mu
        obs_input = th.cat([obs_input, task_repre_mu[:, None, :].repeat(bs, n_ally*n_enemy, 1).to(obs_input.device)], dim=-1)

        # calculate no_attack action q values
        no_attack_embed = self.pairwise_embed_1(obs_input).view(-1, self.args.pairwise_embed_dim)
        no_attack_h_in = no_attack_hidden.view(-1, self.args.pairwise_embed_dim).repeat(n_ally*n_enemy, 1)
        no_attack_h = self.pairwise_rnn_1(no_attack_embed, no_attack_h_in).view(bs*n_agents, n_ally*n_enemy, -1)
        no_attack_q = self.pairwise_action_layer_1(no_attack_h).mean(dim=1)

        # calculate attack action q values
        attack_embed = self.pairwise_embed_2(obs_input).view(-1, self.args.pairwise_embed_dim)
        attack_h_in = attack_hidden.view(-1, self.args.pairwise_embed_dim).repeat(n_ally*n_enemy, 1)
        attack_h = self.pairwise_rnn_2(attack_embed, attack_h_in).view(bs*n_agents, n_ally*n_enemy, -1)
        attack_q = self.pairwise_action_layer_2(attack_h).view(bs*n_agents, n_ally, n_enemy).mean(dim=1)
        
        # calculate q values
        q = th.cat([no_attack_q, attack_q], dim=-1)

        return q, no_attack_h.mean(dim=1), attack_h.mean(dim=1)
        
