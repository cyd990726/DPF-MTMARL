import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    """
    初始化 QMixer 类的实例。

    参数:
    decomposer: 一个分解器对象，用于分解状态和动作信息。
    args: 一个参数对象，包含了模型需要的各种参数。

    在这个函数中，我们首先获取了智能体数量、嵌入维度、注意力嵌入维度和实体嵌入维度等信息。
    然后，我们获取了状态形状信息，包括敌方数量、盟友状态特征维度、敌方状态特征维度和时间步状态维度等。
    接着，我们获取了动作维度信息，包括无攻击动作的数量。
    然后，我们定义了状态信息处理器，用于将状态信息编码为嵌入向量。
    接着，我们定义了用于注意力的查询向量和键向量的线性变换。
    最后，我们定义了用于计算 Q 值的网络，包括用于计算隐藏层权重的超网络、用于计算隐藏层偏置的线性变换和用于计算 V(s) 的网络。
    """
    def __init__(self, decomposer, args):
        super(QMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        self.entity_embed_dim = args.entity_embed_dim

        # get detailed state shape informatin
        self.decomposer = decomposer
        self.n_enemies, state_nf_al, state_nf_en, timestep_state_dim = \
            decomposer.n_enemies, decomposer.state_nf_al, decomposer.state_nf_en, decomposer.timestep_number_state_dim
        self.state_last_action, self.state_timestep_number = decomposer.state_last_action, decomposer.state_timestep_number
        self.n_entities = self.n_agents + self.n_enemies

        # get action dimension information
        self.n_actions_no_attack = decomposer.n_actions_no_attack

        # define state information processor
        if self.state_last_action:
            self.ally_encoder = nn.Linear(state_nf_al + self.n_actions_no_attack + 1, self.entity_embed_dim)
            self.enemy_encoder = nn.Linear(state_nf_en, self.entity_embed_dim)
            state_nf_al += self.n_actions_no_attack + 1
        else:
            self.ally_encoder = nn.Linear(state_nf_al, self.entity_embed_dim)
            self.enemy_encoder = nn.Linear(state_nf_en, self.entity_embed_dim)

        # we ought to do attention
        self.query = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)
        self.key = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)

        if self.state_timestep_number:
            mixing_input_dim = self.entity_embed_dim + timestep_state_dim
            entity_mixing_input_dim = self.entity_embed_dim + self.entity_embed_dim + timestep_state_dim
        else:
            mixing_input_dim = self.entity_embed_dim
            entity_mixing_input_dim = self.entity_embed_dim + self.entity_embed_dim

        # task_repre dependent weights for hidden layer
        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(entity_mixing_input_dim, self.embed_dim)
            self.hyper_w_final = nn.Linear(mixing_input_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(entity_mixing_input_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(mixing_input_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(mixing_input_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(mixing_input_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))


    def forward(self, agent_qs, states):
        """
        QMixer 类的前向传播函数。

        参数:
        agent_qs: 智能体的 Q 值，形状为 [batch_size, seq_len, n_agents]，其中 batch_size 是批次大小，seq_len 是序列长度，n_agents 是智能体的数量。
        states: 状态，形状为 [batch_size, seq_len, state_dim]，其中 state_dim 是状态的维度。

        在这个函数中，我们首先获取了分解的状态信息，包括盟友状态、敌方状态、上一步动作状态和时间步状态。
        然后，我们将这些状态信息进行编码，得到了实体嵌入。
        接着，我们使用了自注意力机制，得到了注意力输出。
        然后，我们构建了混合输入。
        接着，我们定义了第一层的权重和偏置，并计算了隐藏层的输出。
        然后，我们定义了第二层的权重和 V(s)，并计算了最终的输出。
        最后，我们将最终的输出重塑并返回。

        返回:
        q_tot: 总的 Q 值，形状为 [bs, -1, 1]，其中 bs 是批次大小。
        """
        # agent_qs: [batch_size, seq_len, n_agents]
        # states: [batch_size, seq_len, state_dim]
        bs, seq_len = agent_qs.size(0), agent_qs.size(1)

        # get decomposed state information
        "1.首先获取了分解的状态信息，包括盟友状态、敌方状态、上一步动作状态和时间步状态。"
        ally_states, enemy_states, last_action_states, timestep_number_state = self.decomposer.decompose_state(states)
        ally_states = th.stack(ally_states, dim=0)  # [n_agents, bs, seq_len, state_nf_al]
        enemy_states = th.stack(enemy_states, dim=0)    # [n_enemies, bs, seq_len, state_nf_en]

        # stack action information
        if self.state_last_action:
            last_action_states = th.stack(last_action_states, dim=0)
            _, _, compact_action_states = self.decomposer.decompose_action_info(last_action_states)
            ally_states = th.cat([ally_states, compact_action_states], dim=-1)

        # do inference and get entity_embed
        "2.然后，我们将这些状态信息进行编码，得到了实体嵌入。"
        ally_embed = self.ally_encoder(ally_states) # [n_agents, bs, seq_len, entity_embed_dim]
        enemy_embed = self.enemy_encoder(enemy_states) # [n_enemies, bs, seq_len, entity_embed_dim]

        # we ought to do self-attention
        entity_embed = th.cat([ally_embed, enemy_embed], dim=0) # [n_agents+n_enemies, bs, seq_len, entity_embed_dim]
        # n_entities = n_agents+n_enemies
        # do attention
        "3.接着，我们使用了自注意力机制，得到了注意力输出。"
        proj_query = self.query(entity_embed).permute(1, 2, 0, 3).reshape(bs*seq_len, self.n_entities, self.attn_embed_dim) # query linear-> [n_entities, bs, seq_len, attn_embed_dim]; permute-> [bs,seq_len, n_entities, attn_embed_dim], reshape->[bs*seq_len, n_entities, attn_embed_dim]
        proj_key = self.key(entity_embed).permute(1, 2, 3, 0).reshape(bs*seq_len, self.attn_embed_dim, self.n_entities) # [bs*seq_len, attn_embed_dim, n_entities]
        energy = th.bmm(proj_query/(self.attn_embed_dim ** (1/2)), proj_key) # [bs*seq_len, n_entities, n_entities]
        attn_score = F.softmax(energy, dim=1)  # [bs*seq_len, n_entities, n_entities]
        proj_value = entity_embed.permute(1, 2, 3, 0).reshape(bs*seq_len, self.entity_embed_dim, self.n_entities) # [bs*seq_len, entity_embed_dim, n_entities]
        attn_out = th.bmm(proj_value, attn_score).mean(dim=-1).reshape(bs, seq_len, self.entity_embed_dim) #bmm->[bs*seq_len,entity_embed_dim,n_entities]; mean(pooling)->[bs*seq_len,entity_embed_dim,1];[bs, seq_len, entity_embed_dim]

        # concat timestep information
        if self.state_timestep_number:
            raise Exception(f"Not Implemented")
        else:
            pass
        
        # build mixing input
        "4.然后，我们构建了混合输入。"
        entity_mixing_input = th.cat([
            attn_out[:, :, None, :].repeat(1, 1, self.n_agents, 1), #repeat->  [bs, seq_len, n_agents, entity_embed_dim]
            ally_embed.permute(1, 2, 0, 3), #permute->[bs, seq_len, n_agents, entity_embed_dim]
        ], dim=-1) #cat->[bs, seq_len, n_agents, entity_embed_dim*2]
        mixing_input = th.cat([
            attn_out,
        ], dim=-1) #[bs, seq_len, entity_embed_dim]
        
        # First layer
        "5.接着，我们定义了第一层的权重和偏置，并计算了隐藏层的输出。"
        w1 = th.abs(self.hyper_w_1(entity_mixing_input)) #[bs, seq_len, n_agents, embed_dim]
        b1 = self.hyper_b_1(mixing_input) #[bs, seq_len, embed_dim]
        w1 = w1.view(-1, self.n_agents, self.embed_dim) #[bs*seq_len, n_agents, embed_dim]
        b1 = b1.view(-1, 1, self.embed_dim) #[bs*seq_len, 1, embed_dim]
        agent_qs = agent_qs.view(-1, 1, self.n_agents)#[bs*seq_len, 1, n_agents]
        hidden = F.elu(th.bmm(agent_qs, w1) + b1) #[bs*seq_len, 1, embed_dim]

        # Second layer
        "6.然后，我们定义了第二层的权重和 V(s)，并计算了最终的输出。"
        w_final = th.abs(self.hyper_w_final(mixing_input)).view(-1, self.embed_dim, 1)
        v = self.V(mixing_input).view(-1, 1, 1)

        # Compute final output
        y = th.bmm(hidden, w_final) + v

        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        
        return q_tot
