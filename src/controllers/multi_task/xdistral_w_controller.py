from modules.agents import REGISTRY as agent_REGISTRY
from modules.decomposers import REGISTRY as decomposer_REGISTRY
from modules.task_encoders import ENC_REGISTRY as encoder_REGISTRY
from modules.task_encoders import DEC_REGISTRY as decoder_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.distributions as D
import numpy as np
from sympy.matrices import Matrix, GramSchmidt
import torch.nn.functional as F
from torch.distributions import Categorical
import copy

# This multi-agent controller shares parameters between agents
class XDistralMAC:
    def __init__(self, train_tasks, task2scheme, task2args, main_args):
        self.pi_0_mixer = None
        self.pi_i_mixer = None
        
        # set some task-specific attributes
        self.train_tasks = train_tasks
        self.task2scheme = task2scheme
        self.task2args = task2args
        self.task2n_agents = {task: self.task2args[task].n_agents for task in train_tasks}
        self.main_args = main_args

        # set some common attributes
        self.agent_output_type = main_args.agent_output_type
        self.action_selector = action_REGISTRY[main_args.action_selector](main_args)
        
        # get decomposer for each task
        env2decomposer = {
            "sc2": "sc2_decomposer",
            "grid_mpe": "mpe_decomposer",
            "easy_grid_mpe": "mpe_decomposer",
        }
        self.task2decomposer, self.task2dynamic_decoder = {}, {}
        self.surrogate_decomposer = None
        for task in train_tasks:
            task_args = self.task2args[task]
            if task_args.env in ["sc2", "grid_mpe", "easy_grid_mpe"]:
                task_decomposer = decomposer_REGISTRY[env2decomposer[task_args.env]](task_args)
                self.task2decomposer[task] = task_decomposer
                if not self.surrogate_decomposer:
                    self.surrogate_decomposer = task_decomposer
            else:
                raise NotImplementedError(f"Unsupported env decomposer {task_args.env}")
            # set obs_shape
            task_args.obs_shape = task_decomposer.obs_dim
            # 每个任务的decoder都不一样
            task_decoder = decoder_REGISTRY[task_args.dynamic_decoder](task_args)
            self.task2dynamic_decoder[task] = task_decoder

        # 所有任务共用一个encoder
        self.dynamic_encoder = encoder_REGISTRY[self.main_args.dynamic_encoder](self.task2decomposer, self.task2n_agents, self.surrogate_decomposer, self.main_args)
        
        # build agents
        task2input_shape_info = self._get_input_shape()
        self._build_agents(task2input_shape_info)

        self.hidden_states = None
        '修改开始：cyd'
        # 给每一个pi_i都定义一个隐藏状态
        self.hidden_states_is = {}
        for task_name in self.train_tasks:
            self.hidden_states_is[task_name] = None
        '修改结束：cyd'

        # 初始化任务表征
        if getattr(self.main_args, "random_init_src_repres", False):
            # init task_repres randomly
            self.random_init_task_repres()
        else:
            # init task_repres as orthogo vectors
            self.init_task_repres()
    
    
    # pi_0+pi_i结合进行动作选择
    '修改开始：cyd'
    # 保留混合网络权重参数版本的select_action
    # def select_actions(self, ep_batch, t_ep, t_env, task, bs=slice(None), test_mode=False):

    
    #     avail_actions = ep_batch["avail_actions"][:, t_ep]
    #     agent_outputs = self.forward(ep_batch, t_ep, task, test_mode=test_mode)  # 确保 requires_grad=True
    #     agent_outputs = agent_outputs[bs]
        
    #     avail_actions = avail_actions[bs]
    #     pi0 = th.nn.functional.softmax(agent_outputs, dim=-1)  # tensor(2, 12, 20)

    #     Q = self.forward_pi_i(ep_batch, t_ep, task, test_mode=test_mode)  # 确保 requires_grad=True
    #     Q = Q[bs]

    #     task_repre = self.get_task_repres(task, require_grad=False)
    #     task_repre = task_repre.repeat(agent_outputs.shape[0], 1, 1)
        
    #     # 假设 agent_outputs 和 Q 是包含梯度的张量
    #     agent_outputs = agent_outputs.detach().requires_grad_(True)  # 确保 requires_grad=True
    #     Q = Q.detach().requires_grad_(True)  # 确保 requires_grad=True
        
    #     pi_0_test_q_vals = agent_outputs.max(dim=-1, keepdim=True)[0]  # 保留维度
    #     pi_i_test_q_vals = Q.max(dim=-1, keepdim=True)[0]  # 保留维度
        
    #     # 保留梯度
    #     pi_0_test_q_vals.retain_grad()
    #     pi_i_test_q_vals.retain_grad()
        
    #     state = ep_batch["state"][:, t_ep]  # (1, 150)
    #     self.pi_0_mixer.cuda()
    #     self.pi_i_mixer[task].cuda()
    #     tot_Q_0 = self.pi_0_mixer(pi_0_test_q_vals.unsqueeze(1), state.unsqueeze(1), task_repre.unsqueeze(1), self.task2decomposer[task])
    #     tot_Q_i = self.pi_i_mixer[task](pi_i_test_q_vals.unsqueeze(1), state.unsqueeze(1))
        
    #     tot_Q_0.backward()
    #     tot_Q_i.backward()
        
    #     # 计算权重
    #     pi_0_weight = pi_0_test_q_vals.grad
    #     pi_i_weight = pi_i_test_q_vals.grad
        
    #     # print("pi_0_weight:", pi_0_weight)
    #     # print("pi_i_weight:", pi_i_weight)
    #     # print(pi_0_weight.shape)
    #     pi_i = th.pow(th.pow(pi0, pi_0_weight), self.main_args.alpha)* th.exp(self.main_args.beta*Q*pi_i_weight)
        
    #     chosen_actions = self.action_selector.select_action(pi_i, avail_actions, t_env, test_mode=test_mode)

    #     # # 返回选择的动作
    #     return chosen_actions
    # 保留混合网络权重参数版本的select_action
    def select_actions(self, ep_batch, t_ep, t_env, task, bs=slice(None), test_mode=False):

    
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, task, test_mode=test_mode)  # 确保 requires_grad=True
        agent_outputs = agent_outputs[bs]
        
        avail_actions = avail_actions[bs]
        pi0 = th.nn.functional.softmax(agent_outputs, dim=-1)  # tensor(2, 12, 20)

        Q = self.forward_pi_i(ep_batch, t_ep, task, test_mode=test_mode)  # 确保 requires_grad=True
        Q = Q[bs]

        task_repre = self.get_task_repres(task, require_grad=False)
        task_repre = task_repre.repeat(agent_outputs.shape[0], 1, 1)
        
        # 假设 agent_outputs 和 Q 是包含梯度的张量
        agent_outputs = agent_outputs.detach().requires_grad_(True)  # 确保 requires_grad=True
        Q = Q.detach().requires_grad_(True)  # 确保 requires_grad=True
        
        pi_0_test_q_vals = agent_outputs.max(dim=-1, keepdim=True)[0]  # 保留维度
        pi_i_test_q_vals = Q.max(dim=-1, keepdim=True)[0]  # 保留维度
        
        # 保留梯度
        pi_0_test_q_vals.retain_grad()
        pi_i_test_q_vals.retain_grad()
        
        state = ep_batch["state"][:, t_ep]  # (1, 150)
        self.pi_0_mixer.cuda()
        self.pi_i_mixer[task].cuda()
        tot_Q_0 = self.pi_0_mixer(pi_0_test_q_vals.unsqueeze(1), state.unsqueeze(1), task_repre.unsqueeze(1), self.task2decomposer[task])
        tot_Q_i = self.pi_i_mixer[task](pi_i_test_q_vals.unsqueeze(1), state.unsqueeze(1))
        
        tot_Q_0.backward()
        tot_Q_i.backward()
        
        # 计算权重
        pi_0_weight = pi_0_test_q_vals.grad
        pi_i_weight = pi_i_test_q_vals.grad
        
        # print("pi_0_weight:", pi_0_weight)
        # print("pi_i_weight:", pi_i_weight)
        # print(pi_0_weight.shape)
        pi_i = th.pow(th.pow(pi0, pi_0_weight), self.main_args.alpha)* th.exp(self.main_args.beta*Q*pi_i_weight)
        
        chosen_actions = self.action_selector.select_action(pi_i, avail_actions, t_env, test_mode=test_mode)

        # # 返回选择的动作
        return chosen_actions
    
    # pi_i前向运算
    # 这里应该默认的输出类型为Q，所以这样写应该是没什么问题的
    def forward_pi_i(self, ep_batch, t, task, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t, task)
        agent_outs, self.hidden_states_is[task] = self.agent_is[task](agent_inputs, self.hidden_states_is[task])# (320, 186), (32, 10, 54)  (320, 32)
        return agent_outs.view(ep_batch.batch_size, self.task2n_agents[task], -1)

    '修改结束：cyd'
    
    # pi_0前向运算
    def forward(self, ep_batch, t, task, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t, task)  # 调用_build_inputs方法来构建代理的输入。这个方法会根据当前的环境状态（ep_batch）和时间步（t）来构建输入。
        avail_actions = ep_batch["avail_actions"][:, t]  # 获取当前可用的动作（avail_actions）。
        
        bs = agent_inputs.shape[0]//self.task2n_agents[task]  # 计算批处理大小（bs），这是通过将代理输入的形状除以当前任务的代理数量来得到的。
        # 获取当前任务的表示（task_repre），并将其重复bs次。
        task_repre = self.get_task_repres(task, require_grad=False)
        task_repre = task_repre.repeat(bs, 1)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, task_repre, task)

        # Softmax the agent outputs if they're policy logits
        # 如果代理输出类型为"pi_logits"，则对代理输出进行softmax操作。这是因为在多分类问题中，我们通常使用softmax函数将输出转化为概率分布。
        # 在这个过程中，如果设置了"mask_before_softmax"，则会对不可用的动作进行掩盖，使其在softmax操作后的概率接近于0。
        if self.agent_output_type == "pi_logits":
            if getattr(self.main_args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.task2n_agents[task], -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.main_args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                                + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.main_args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.task2n_agents[task], -1)  # 将代理输出的形状重塑为(batch_size, n_agents, -1)，并返回
        
        
    def init_hidden(self, batch_size, task):
        # we always know we are in which task when do init_hidden
        n_agents = self.task2n_agents[task]
        hidden_states = self.agent.init_hidden()
        self.hidden_states = hidden_states.unsqueeze(0).expand(batch_size, n_agents, -1)
        '修改开始：cyd'
        # 顺便把对应任务的pi_i的隐藏状态也给初始化了
        hidden_states_s = self.agent_is[task].init_hidden()
        self.hidden_states_is[task] =  hidden_states_s.unsqueeze(0).expand(batch_size, n_agents, -1)
        '修改结束：cyd'
        
    def parameters(self):
        return self.agent.parameters()
    
    '修改开始：cyd'
    # 返回指定的pi_i的参数
    def parameters_pi_i(self, task_name):
        return self.agent_is[task_name].parameters()   
    '修改结束：cyd'
    

    def load_state(self, other_mac):
        """ we don't load the state of task dynamic decoder """
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.dynamic_encoder.load_state_dict(other_mac.dynamic_encoder.state_dict())

    '修改开始：cyd'
    def load_state_pi_i(self, other_mac):
        for task in self.train_tasks:
            self.agent_is[task].load_state_dict(other_mac.agent_is[task].state_dict())
    '修改结束'

    def cuda(self):
        self.agent.cuda()
        self.dynamic_encoder.cuda()
        for task in self.train_tasks:
            self.task2dynamic_decoder[task].cuda()
            '修改开始：cyd'
            self.agent_is[task].cuda()
            '修改结束：cyd'

    def save_models(self, path):
        """ we don't save the state of task dynamic decoder """
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.dynamic_encoder.state_dict(), "{}/dynamic_encoder.th".format(path))
        
        '修改开始：cyd'
        # 保存pi_i
        for task_name in self.train_tasks:
            th.save(self.agent_is[task_name].state_dict(), "{}/{}_agent.th".format(path, task_name))
        '修改结束：cyd'

    def load_models(self, path):
        """ we don't load the state of task_encoder """
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.dynmaic_encoder.load_state_dict(th.load("{}/dynamic_encoder.th".format(path), map_location=lambda storage, loc: storage))
        '修改开始：cyd'
        # 保存pi_i
        for task_name in self.train_tasks:
            self.agent_is[task_name].load_state_dict(th.load("{}/{}_agent.th".format(path, task_name), map_location=lambda storage, loc: storage))
        '修改结束：cyd'
        
    def _build_agents(self, task2input_shape_info):
        # 这个是pi_0
        self.agent = agent_REGISTRY[self.main_args.agent](task2input_shape_info,
                                                        self.task2decomposer, self.task2n_agents,
                                                       self.surrogate_decomposer, self.main_args)
        '修改开始：cyd'
        
        self.agent_is = {}
        for task_name in self.train_tasks:
            # pi_i的智能体网络用单任务版本的
            args = copy.deepcopy(self.task2args[task_name])
            self.agent_is[task_name] = agent_REGISTRY["sota"](task2input_shape_info[task_name],
                                                        self.task2decomposer[task_name], args)
        '修改结束：cyd'

    def _build_inputs(self, batch, t, task):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])
        # get args, n_agents for this specific task
        task_args, n_agents = self.task2args[task], self.task2n_agents[task]
        if task_args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if task_args.obs_agent_id:
            inputs.append(th.eye(n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        
        inputs = th.cat([x.reshape(bs*n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self):
        task2input_shape_info = {}
        for task in self.train_tasks:
            task_scheme = self.task2scheme[task]
            input_shape = task_scheme["obs"]["vshape"]
            last_action_shape, agent_id_shape = 0, 0
            if self.task2args[task].obs_last_action:
                input_shape += task_scheme["actions_onehot"]["vshape"][0]
                last_action_shape = task_scheme["actions_onehot"]["vshape"][0]
            if self.task2args[task].obs_agent_id:
                input_shape += self.task2n_agents[task]
                agent_id_shape = self.task2n_agents[task]
            task2input_shape_info[task] = {
                "input_shape": input_shape,
                "last_action_shape": last_action_shape,
                "agent_id_shape": agent_id_shape,
            }
        return task2input_shape_info

    # ---------- some methods related with task representation ----------

    def init_task_repres(self):
        """ init task repres """
        def orthogo_tensor(x):
            m, n = x.size()
            x_np = x.t().numpy()
            matrix = [Matrix(col) for col in x_np.T]
            gram = GramSchmidt(matrix)
            ort_list = []
            for i in range(m):
                vector = []
                for j in range(n):
                    vector.append(float(gram[i][j]))
                ort_list.append(vector)
            ort_list = np.mat(ort_list)
            ort_list = th.from_numpy(ort_list)
            ort_list = F.normalize(ort_list,dim=1)
            return ort_list
        task_repres = th.rand((len(self.train_tasks), self.main_args.task_repre_dim))
        task_repres = orthogo_tensor(task_repres)
        self.task2repre = {}
        for i, task in enumerate(self.train_tasks):
            self.task2repre[task] = task_repres[i].to(self.main_args.device).float()

    def random_init_task_repres(self):
        """ init task repres randomly """
        task_repres = th.rand((len(self.train_tasks), self.main_args.task_repre_dim))
        def normalize_tensor(x):
            # x.shape: [m, n]
            x_norm = th.norm(x, dim=-1).unsqueeze(-1)
            normalized_x = x/x_norm
            return normalized_x
        if getattr(self.main_args, "normalize_repres", False):
            # normalize tensors
            task_repres = normalize_tensor(task_repres)
        self.task2repre = {}
        for i, task in enumerate(self.train_tasks):
            self.task2repre[task] = task_repres[i].to(self.main_args.device).float()           

    def get_task_repres(self, task, require_grad=False):
        assert not require_grad, "Not train task repre in mt training phase!"
        return self.task2repre[task].unsqueeze(0).repeat(self.task2n_agents[task], 1)


    def dynamic_encoder_parameters(self):
        return list(self.dynamic_encoder.parameters())

    def dynamic_decoder_parameters(self, task):
        return list(self.task2dynamic_decoder[task].parameters())

    def task_encoder_parameters(self, task):
        # should not add task_repre_parameter
        return list(self.dynamic_encoder.parameters()) + list(self.task2dynamic_decoder[task].parameters())

    def task_encoder_forward(self, batch, t, task):
        """
        #### shape information
        # obs: [bs, max_seq_len, n_agents, obs_dim]
        # state: [bs, max_seq_len, state_dim]
        # actions: [bs, max_seq_len, n_agents, action_dim]
        """
        obs, state, actions = batch["obs"][:, t], batch["state"][:, t], batch["actions_onehot"][:, t]
        # Should require_grad=False here !!!
        task_repre = self.get_task_repres(task, require_grad=False)
        encoded_latent, bs = self.dynamic_encoder(obs, state, actions, task, task_repre)
        next_obs, next_state, reward = self.task2dynamic_decoder[task](encoded_latent, bs)
        return next_obs, next_state, reward



    def save_task_repres(self, path, task):
        """ save task representations """
        task_repre = self.task2repre[task].cpu().detach().numpy()
        np.save(path, task_repre)

        