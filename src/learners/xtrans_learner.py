import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.attn_x import QMixer as AttnXQMixer
from modules.mixers.attn2_x import QMixer as Attn2XQMixer
from modules.mixers.attn2_h import QMixer as Attn2HiddenQMixer
from modules.mixers.attn2_hx import QMixer as Attn2HiddenXQMixer
from modules.mixers.attn2_hx_mpe import QMixer as Attn2HiddenXMPEQMixer
import torch as th
from torch.optim import RMSprop
import numpy as np

import os

class XTransLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents

        self.params = list(mac.parameters()) # only contains agent parameters

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "attn_x":
                self.mixer = AttnXQMixer(self.mac.decomposer, args)
            elif args.mixer == "attn2_x":
                self.mixer = Attn2XQMixer(self.mac.decomposer, args)
            elif args.mixer == "attn2_h":
                self.mixer = Attn2HiddenQMixer(self.mac.decomposer, args)
            elif args.mixer == "attn2_hx":
                self.mixer = Attn2HiddenXQMixer(self.mac.decomposer, args)
            elif args.mixer == "attn2_hx_mpe":
                self.mixer = Attn2HiddenXMPEQMixer(self.mac.decomposer, args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)    # optimize agent and mixer

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        # task encoder
        self.to_do_dynamic_learning = getattr(args, "pretrain", False)
        if getattr(args, "meta_test", False):
            self.task_encoder_params = self.mac.dynamic_decoder_parameters() + self.mac.task_repres_parameters()
        else:
            self.task_encoder_params = self.mac.task_encoder_parameters() # task_parameters contrain three parts/components
        self.task_encoder_optimiser = RMSprop(params=self.task_encoder_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        if self.args.save_repre:
            self.task_repre_dir = os.path.join(self.args.output_dir, "task_repre")
            self.save_repre_t = -self.args.save_repre_interval - 1
            os.makedirs(self.task_repre_dir, exist_ok=True)

        # used when in rl training, since in rl-training phase, task_repre is statistic
        self.task_repre = None
        
    def dynamic_train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # Calculate prediction loss
        obs_pred, state_pred, reward_pred = [], [], []
        for t in range(batch.max_seq_length):
            obs_preds, state_preds, reward_preds = self.mac.task_encoder_forward(batch, t=t)
            obs_pred.append(obs_preds)
            state_pred.append(state_preds)
            reward_pred.append(reward_preds)
        obs_pred = th.stack(obs_pred, dim=1)[:, :-1]
        state_pred = th.stack(state_pred, dim=1)[:, :-1]
        reward_pred = th.stack(reward_pred, dim=1)[:, :-1]
        # get target labels
        obs = batch["obs"][:, 1:].detach().clone()
        state = batch["state"][:, 1:].detach().clone().unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        repeated_rewards = batch["reward"][:, :-1].detach().clone().unsqueeze(2).repeat(1, 1, self.n_agents, 1)

        # calculate prediction loss
        pred_obs_loss = th.sqrt(((obs_pred - obs) ** 2).sum(dim=-1))
        pred_state_loss = th.sqrt(((state_pred - state) ** 2).sum(dim=-1))
        pred_reward_loss = ((reward_pred - repeated_rewards) ** 2).squeeze(dim=-1)
        
        mask = mask.expand_as(pred_reward_loss)
        
        # do loss mask
        pred_obs_loss = (pred_obs_loss * mask).mean(dim=-1).sum() / mask.sum()
        pred_state_loss = (pred_state_loss * mask).mean(dim=-1).sum() / mask.sum()
        pred_reward_loss = (pred_reward_loss * mask).mean(dim=-1).sum() / mask.sum()

        # sparse loss
        sparse_loss = self.mac.compute_sparse_loss()

        task_repre_loss = pred_obs_loss + pred_state_loss + 10 * pred_reward_loss + 0.1 * sparse_loss
 
        self.task_encoder_optimiser.zero_grad()
        task_repre_loss.backward()
        pred_grad_norm = th.nn.utils.clip_grad_norm_(self.task_encoder_params, self.args.grad_norm_clip)
        self.task_encoder_optimiser.step()
        
        # get grad norm scalar for tensorboard recording
        try:
            pred_grad_norm = pred_grad_norm.item()
        except:
            pass

        if self.args.save_repre and t_env - self.save_repre_t >= self.args.save_repre_interval:
            self.mac.save_task_repres(os.path.join(self.task_repre_dir, f"{t_env}.npy"))
            self.save_repre_t = t_env

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # Dynamic learning phase
            self.logger.log_stat("pred_obs_loss", pred_obs_loss.item(), t_env)
            self.logger.log_stat("pred_state_loss", pred_state_loss.item(), t_env)
            self.logger.log_stat("pred_reward_loss", pred_reward_loss.item(), t_env)
            self.logger.log_stat("task_encoder_grad_norm", pred_grad_norm, t_env)
            self.logger.log_stat("sparse_loss", sparse_loss.item(), t_env)
            self.log_stats_t = t_env
        
        if t_env > self.args.dynamic_learning_end:
            # set do_dynamic_learning False and reset some values
            self.to_do_dynamic_learning = False
            self.last_target_update_episode = 0
            self.log_stats_t = -self.args.learner_log_interval - 1
            # if to save_repre, save the final representation
            if self.args.save_repre:
                self.mac.save_task_repres(os.path.join(self.task_repre_dir, f"{t_env}.npy"))
            return True

    def rl_train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.task_repre is None:    
            self.task_repre = self.mac.get_task_repres(require_grad=False) # .to(chosen_action_qvals.device)
        bs, seq_len = chosen_action_qvals.size(0), chosen_action_qvals.size(1)
        task_repre = self.task_repre[None, None, ...].repeat(bs, seq_len, 1, 1)
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], task_repre)
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], task_repre)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Do RL Learning
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        # Prepare scalar for tensorboard logging
        try:
            grad_norm = grad_norm.item()
        except:
            pass
    
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        if self.to_do_dynamic_learning:
            terminated = self.dynamic_train(batch, t_env, episode_num)
            if terminated:
                return True
        else:
            self.rl_train(batch, t_env, episode_num)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
