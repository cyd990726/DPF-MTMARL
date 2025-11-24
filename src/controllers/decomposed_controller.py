from modules.agents import REGISTRY as agent_REGISTRY
from modules.decomposers import REGISTRY as decomposer_REGISTRY
from modules.task_encoders import WHOLE_REGISTRY as task_encoder_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


"""
TODO
- check whether there are bugs in save_models and load_models
"""

# This multi-agent controller shares parameters between agents
class DecomposedMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        
        env2decomposer = {
            "sc2": "sc2_decomposer",
        }
        if args.env == "sc2":
            self.decomposer = decomposer_REGISTRY[env2decomposer[args.env]](args)
        else:
            raise NotImplementedError(f"Unsupported env decomposer {args.env}")
        args.obs_shape = self.decomposer.obs_dim

        self.task_encoder = task_encoder_REGISTRY[args.task_encoder](args)

        input_shape_info = self._get_input_shape(scheme)
        self._build_agents(input_shape_info)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs = self.agent(agent_inputs)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        pass

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.task_encoder.load_state_dict(other_mac.task_encoder.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.task_encoder.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.task_encoder.state_dict(), "{}/task_encoder.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.task_encoder.load_state_dict(th.load("{}/task_encoder.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape_info):
        self.agent = agent_REGISTRY[self.args.agent](input_shape_info, self.decomposer, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        last_action_shape, agent_id_shape = 0, 0
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
            last_action_shape = scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
            agent_id_shape = self.n_agents

        return {
            "input_shape": input_shape,
            "last_action_shape": last_action_shape,
            "agent_id_shape": agent_id_shape,
        }

    def get_task_repres(self, require_grad=True):
        if require_grad:
            return self.agent.task_repre_mu
        else:
            return self.agent.task_repre_mu.detach().clone()

    def task_encoder_parameters(self):
        return list(self.task_encoder.parameters())

    def task_encoder_forward(self, batch, t):
        # obs: [bs, max_seq_len, n_agents, obs_dim]
        # state: [bs, max_seq_len, state_dim]
        # actions: [bs, max_seq_len, n_agents, action_dim]
        obs, state, actions = batch["obs"][:, t], batch["state"][:, t], batch["actions_onehot"][:, t]
        task_repres = self.get_task_repres(require_grad=True)
        next_obs, next_state, reward = self.task_encoder(obs, state, actions, task_repres)
        
        return next_obs, next_state, reward