from modules.agents import REGISTRY as agent_REGISTRY
from modules.decomposers import REGISTRY as decomposer_REGISTRY
from modules.task_encoders import WHOLE_REGISTRY as task_encoder_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.distributions as D
import numpy as np


# This multi-agent controller shares parameters between agents
class TransMAC:
    def __init__(self, train_tasks, task2scheme, task2args, main_args):
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
        }
        self.task2decomposer, self.task2encoder = {}, {}
        self.surrogate_decomposer = None
        for task in train_tasks:
            task_args = self.task2args[task]
            if task_args.env == "sc2":
                task_decomposer = decomposer_REGISTRY[env2decomposer[task_args.env]](task_args)
                self.task2decomposer[task] = task_decomposer
                if not self.surrogate_decomposer:
                    self.surrogate_decomposer = task_decomposer
            else:
                raise NotImplementedError(f"Unsupported env decomposer {task_args.env}")
            # set obs_shape
            task_args.obs_shape = task_decomposer.obs_dim
            # define task encoder
            task_encoder = task_encoder_REGISTRY[task_args.task_encoder](task_args)
            self.task2encoder[task] = task_encoder

        task2input_shape_info = self._get_input_shape()
        self._build_agents(task2input_shape_info)

        self.hidden_states = None

        # define task_repre_mu, task_repre_sigma here
        self.task2repre_mu, self.task2repre_sigma = {}, {}
        for task in self.train_tasks:
            self.task2repre_mu[task] = th.zeros(task2args[task].n_agents, task2args[task].task_repre_dim, requires_grad=True)
            self.task2repre_sigma[task] = th.ones(task2args[task].n_agents, task2args[task].task_repre_dim, requires_grad=True)

    def select_actions(self, ep_batch, t_ep, t_env, task, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, task, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, task, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t, task)
        avail_actions = ep_batch["avail_actions"][:, t]
        
        bs = agent_inputs.shape[0]//self.task2n_agents[task]
        if test_mode:
            task_repre, _ = self.get_task_repres(task, require_grad=False)
            task_repre = task_repre.to(agent_inputs.device).repeat(bs, 1)
        else:
            task_repre = self.sample_task_repres(task, require_grad=False, shape=(bs,)).reshape(bs*self.task2n_agents[task], -1).to(agent_inputs.device)
        
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, task_repre, task)

        # Softmax the agent outputs if they're policy logits
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

        return agent_outs.view(ep_batch.batch_size, self.task2n_agents[task], -1)

    def init_hidden(self, batch_size, task):
        # we always know we are in which task when do init_hidden
        n_agents = self.task2n_agents[task]
        hidden_states = self.agent.init_hidden()
        self.hidden_states = hidden_states.unsqueeze(0).expand(batch_size, n_agents, -1)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        """ we don't load the state of task_encoder """
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()
        for task in self.train_tasks:
            # let all task_encoders to be on cuda
            self.task2encoder[task].cuda()

    def save_models(self, path):
        """ we don't save the state of task_encoder """
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        """ we don't load the state of task_encoder """
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, task2input_shape_info):
        self.agent = agent_REGISTRY[self.main_args.agent](task2input_shape_info,
                                                        self.task2decomposer, self.task2n_agents,
                                                        self.surrogate_decomposer, self.main_args)

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

    def get_task_repres(self, task, require_grad=True):
        if require_grad:
            return self.task2repre_mu[task], self.task2repre_sigma[task]
        else:
            # I don't do .clone()
            return self.task2repre_mu[task].detach(), self.task2repre_sigma[task].detach()

    def sample_task_repres(self, task, require_grad=True, shape=None):
        """ sample representations from distribution """
        dist = D.Normal(self.task2repre_mu[task], self.task2repre_sigma[task])
        if require_grad:
            task_repres = dist.rsample(shape) if shape is not None else dist.rsample()
        else:
            task_repres = dist.sample(shape) if shape is not None else dist.sample()
        return task_repres

    def task_encoder_parameters(self, task):
        return list(self.task2encoder[task].parameters()) + [self.task2repre_mu[task], self.task2repre_sigma[task]]

    def task_encoder_forward(self, batch, t, task):
        """
        #### shape information
        # obs: [bs, max_seq_len, n_agents, obs_dim]
        # state: [bs, max_seq_len, state_dim]
        # actions: [bs, max_seq_len, n_agents, action_dim]
        """
        obs, state, actions = batch["obs"][:, t], batch["state"][:, t], batch["actions_onehot"][:, t]
        task_repre_mu, task_repre_sigma = self.get_task_repres(task, require_grad=True)
        next_obs, next_state, reward = self.task2encoder[task](obs, state, actions, task_repre_mu, task_repre_sigma)
        return next_obs, next_state, reward

    def compute_mi_loss(self, batch, t, task):
        """ compute mi loss """
        next_obs, next_state, reward = batch["obs"][:, t+1], batch["state"][:, t+1], batch["reward"][:, t+1]
        task_repre_mu, task_repre_sigma = self.get_task_repres(task, require_grad=True)
        mi_loss = self.task2encoder[task].compute_mi_loss(next_obs, next_state, reward, task_repre_mu, task_repre_sigma)
        return mi_loss

    def save_task_repres(self, path, task):
        """save task representations"""
        task_repre_mu, task_repre_sigma = self.get_task_repres(task, require_grad=False)
        mu_array, sigma_array = task_repre_mu.cpu().numpy(), task_repre_sigma.cpu().numpy()
        np.save(path, np.stack([mu_array, sigma_array]))

    # ---------- util functions for debugging ----------

    def get_task_repres_grad(self):
        """
        if self.task_repre_mu.grad is None:
            raise Exception(f"try to access one tensor's grad which is NoneType")
        else:
            grad1 = self.task_repre_mu.grad.detach().mean().item()
            grad2 = self.task_repre_sigma.grad.detach().mean().item()
            return grad1, grad2
        """
        raise NotImplementedError
        