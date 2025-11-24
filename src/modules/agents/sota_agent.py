import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils.embed import polynomial_embed, binary_embed


class SotaAgent(nn.Module):
    def __init__(self, input_shape_info, decomposer, args):
        """
        初始化 SotaAgent 类的实例。

        参数:
        input_shape_info: 一个字典，包含了输入的形状信息。包括 "last_action_shape" 和 "agent_id_shape"。
        decomposer: 一个分解器对象，用于分解观察输入。
        args: 一个参数对象，包含了模型需要的各种参数。

        在这个函数中，我们首先定义了各种维度信息，包括实体嵌入维度、注意力嵌入维度等。
        然后，我们获取了观察输入的形状信息，包括自身观察的维度、敌方特征的维度、盟友特征的维度等。
        接着，我们定义了各种网络，包括用于注意力的网络、用于计算动作 Q 值的网络等。
        """
        super(SotaAgent, self).__init__()
        self.last_action_shape = input_shape_info["last_action_shape"]
        self.agent_id_shape = input_shape_info["agent_id_shape"]
        self.args = args

        # 定义各种维度信息
        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim

        # 获取观察输入的形状信息
        self.decomposer = decomposer
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        n_actions_no_attack = decomposer.n_actions_no_attack

        # get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
        # enemy_obs ought to add attack_action_info
        obs_en_dim += 1

        # 获取智能体数量信息
        self.n_agents = args.n_agents
        self.n_enemy = self.decomposer.n_enemies
        self.n_ally = args.n_agents - 1
        self.n_entity = self.n_agents + self.n_enemy

        # 定义各种网络
        self.query = nn.Linear(wrapped_obs_own_dim, self.attn_embed_dim)  #
        self.ally_key = nn.Linear(obs_al_dim, self.attn_embed_dim)
        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim)
        self.enemy_key = nn.Linear(obs_en_dim, self.attn_embed_dim)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)

        self.rnn = nn.GRUCell(self.entity_embed_dim * 3, args.rnn_hidden_dim)
        self.wo_action_layer = nn.Linear(args.rnn_hidden_dim, n_actions_no_attack)
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

    def forward(self, inputs, hidden_state):
        """
        SotaAgent 类的前向传播函数。

        参数:
        inputs: 输入张量，包含了观察输入、上一步动作信息和智能体ID信息。
        hidden_state: 上一步的隐藏状态。

        在这个函数中，我们首先将输入分解为观察输入、上一步动作信息和智能体ID信息。
        然后，我们将观察输入进一步分解为自身观察、敌方特征和盟友特征。
        接着，我们计算了用于注意力的查询向量、键向量和值向量。
        然后，我们使用注意力机制计算了盟友隐藏状态和敌方隐藏状态，并将它们与自身隐藏状态合并。
        接着，我们使用 GRU 单元更新隐藏状态。
        最后，我们计算了无攻击动作的 Q 值和攻击动作的 Q 值，并将它们合并。

        返回:
        q: 动作 Q 值，形状为 [bs*n_agents, n_actions]，其中 bs 是批次大小，n_agents 是智能体的数量，n_actions 是动作的数量。
        h: 新的隐藏状态，形状为 [bs*n_agents, rnn_hidden_dim]，其中 rnn_hidden_dim 是 RNN 隐藏状态的维度。
        """
        # decompose inputs into observation inputs, last_action_info, agent_id_info
        obs_dim = self.decomposer.obs_dim
        "1.将输入分解为观察输入、上一步动作信息和智能体ID信息。"
        obs_inputs, last_action_inputs, agent_id_inputs = inputs[:, :obs_dim], \
            inputs[:, obs_dim:obs_dim + self.last_action_shape], inputs[:, obs_dim + self.last_action_shape:]

        # decompose observation input
        "2.将观察输入进一步分解为自身观察、敌方特征和盟友特征。"
        own_obs, enemy_feats, ally_feats = self.decomposer.decompose_obs(
            obs_inputs)  # own_obs: [bs*self.n_agents, own_obs_dim]
        bs = int(own_obs.shape[0] / self.n_agents)

        # embed agent_id inputs and decompose last_action_inputs
        agent_id_inputs = [
            th.as_tensor(binary_embed(i + 1, self.args.id_length, self.args.max_agent), dtype=own_obs.dtype) for i in
            range(self.n_agents)]
        agent_id_inputs = th.stack(agent_id_inputs, dim=0).repeat(bs, 1).to(
            own_obs.device)  # [n_agents, id_length]; repeat-> [bs*n_agents, id_length]
        _, attack_action_info, compact_action_states = self.decomposer.decompose_action_info(last_action_inputs)

        # incorporate agent_id embed and compact_action_states
        own_obs = th.cat([own_obs, agent_id_inputs, compact_action_states], dim=-1)

        # incorporate attack_action_info into enemy_feats
        attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
        enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info],
                             dim=-1)  # [n_enemy_agents, bs*n_agents, obs_en_dim]
        ally_feats = th.stack(ally_feats, dim=0)  # [n_ally_agents, bs*n_agents, obs_al_dim]

        # compute key, query and value for attention
        "3.计算用于注意力的查询向量、键向量和值向量。"
        own_hidden = self.own_value(own_obs)  # [bs*n_agents, entity_embed_dim]
        query = self.query(own_obs)  # [bs*n_agents, attn_embed_dim]
        ally_keys = self.ally_key(ally_feats).permute(1, 2,
                                                      0)  # [n_ally_agents, bs*n_agents, attn_embed_dim] -> [bs*n_agents,  attn_embed_dim,  n_ally_agents]
        enemy_keys = self.enemy_key(enemy_feats).permute(1, 2, 0)
        ally_values = self.ally_value(ally_feats).permute(1, 0,
                                                          2)  # [n_ally_agents, bs*n_agents, entity_embed_dim] -> [bs*n_agents, n_ally_agents, entity_embed_dim ]
        enemy_values = self.enemy_value(enemy_feats).permute(1, 0, 2)

        # do attention
        "4.使用注意力机制计算了盟友隐藏状态和敌方隐藏状态，并将它们与自身隐藏状态合并得到tot_hidden"
        ally_hidden = self.attention(query, ally_keys, ally_values,
                                     self.attn_embed_dim)  # [bs*n_agents, entity_embed_dim]
        enemy_hidden = self.attention(query, enemy_keys, enemy_values,
                                      self.attn_embed_dim)  # [bs*n_agents, entity_embed_dim]
        tot_hidden = th.cat([own_hidden, ally_hidden, enemy_hidden], dim=-1)  # [bs*n_agents, 3*entity_embed_dim]  

        "5.使用 GRU 单元更新隐藏状态。"
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(tot_hidden, h_in)  # [bs*n_agents, rnn_hidden_dim]

        """6.计算了无攻击动作的 Q 值和攻击动作的 Q 值，并将它们合并。"""
        # compute wo_action_q
        wo_action_q = self.wo_action_layer(h)

        # compute action_q
        enemy_hidden = self.enemy_embed(enemy_feats)  # [n_enemy_agents, bs*n_agents, rnn_hidden_dim]
        attack_action_input = th.cat([enemy_hidden, h.unsqueeze(0).repeat(enemy_feats.size(0), 1, 1)],
                                     dim=-1)  # unsqueeze-> [1, bs*n_agents, rnn_hidden_dim]; repeat->[n_enemy_agents, bs*n_agents, rnn_hidden_dim]; cat->[n_enemy_agents, bs*n_agents, rnn_hidden_dim*2]
        attack_action_q = self.attack_action_layer(attack_action_input).squeeze(-1).transpose(0,
                                                                                              1)  # layer-> [n_enemy_agents, bs*n_agents,1]; squeeze->[n_enemy_agents, bs*n_agents]; trans->[bs*n_agents, n_enemy_agents]

        q = th.cat([wo_action_q, attack_action_q], dim=-1)

        return q, h

    # 对应论文中Figure 2. Population-invariant network structure for policy learning.的(d) Attention scheme
    def attention(self, q, k, v, attn_dim):
        """
        实现注意力机制的函数。

        参数:
        q: 查询向量，形状为 [bs*n_agents, attn_dim]，其中 bs 是批次大小，n_agents 是智能体的数量。
        k: 键向量，形状为 [bs*n_agents, attn_dim, n_entity]，其中 n_entity 是实体的数量。
        v: 值向量，形状为 [bs*n_agents, n_entity, value_dim]，其中 value_dim 是值向量的维度。
        attn_dim: 注意力维度，用于调整查询向量和键向量的乘积。

        返回:
        注意力输出，形状为 [bs*n_agents, value_dim]。
        """
        # 计算查询向量和键向量的乘积，得到能量矩阵
        energy = th.bmm(q.unsqueeze(1) / (attn_dim ** (1 / 2)),
                        k)  # q-> [bs*n_agents, 1, attn_dim]; q \bmm k -> [bs*n_agents, 1, n_entity]

        # 对能量矩阵进行 softmax 操作，得到注意力分数
        attn_score = F.softmax(energy, dim=-1)

        # 将注意力分数和值向量进行乘积，得到注意力输出
        attn_out = th.bmm(attn_score, v).squeeze(1)  # [bs*n_agents,1, value_dim] -> [bs*n_agents, value_dim]

        return attn_out
