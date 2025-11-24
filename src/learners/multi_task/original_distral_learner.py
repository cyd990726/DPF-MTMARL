import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.attn_x import QMixer as AttnXQMixer
from modules.mixers.multi_task.attn_x import QMixer as MultiTaskAttnXQMixer
from modules.mixers.multi_task.attn2_hx import QMixer as MultiTaskAttn2HiddenXQMixer
from modules.mixers.multi_task.attn2_hx_mpe import QMixer as MultiTaskAttn2HiddenXMPEQMixer
from modules.mixers.attn2_h import QMixer as AttnHQMixer

import torch as th
from torch.optim import RMSprop

import os
import numpy as np
import torch.nn.functional as F

class XDistralLearner:
    '修改开始：cyd'
    def __init__(self, mac, logger, main_args):
        '修改结束：cyd'
        # # 保存传入的参数
        self.main_args = main_args
        self.mac = mac
        self.logger = logger

        # get some attributes from mac
        self.task2args = mac.task2args
        self.task2n_agents = mac.task2n_agents
        self.surrogate_decomposer = mac.surrogate_decomposer
        self.task2decomposer = mac.task2decomposer

        self.params = list(mac.parameters())  # 获取mac的参数

        self.last_target_update_episode = 0  # 初始化最后一次目标更新的环境步数

        # 根据main_args.mixer的值来选择混合器（mixer）
        self.mixer = None
        
        '修改开始：cyd'
        # 为Πi定义一组mixer
        # main_args_copy = copy.deepcopy(main_args)
        self.pi_mixers = {}
        for task_name in self.task2args:
                # main_args_copy.n_agents = self.task2n_agents[task_name]
                args = copy.deepcopy(self.task2args[task_name])
                self.pi_mixers[task_name] = AttnHQMixer(self.task2decomposer[task_name], args)

        if main_args.mixer is not None:
            if main_args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif main_args.mixer == "qmix":
                self.mixer = QMixer(main_args)
            elif main_args.mixer == "mt_attn_x":
                self.mixer = MultiTaskAttnXQMixer(self.surrogate_decomposer, main_args)
            elif main_args.mixer == "mt_attn2_hx":
                self.mixer = MultiTaskAttn2HiddenXQMixer(self.surrogate_decomposer, main_args)
            elif main_args.mixer == "mt_attn2_hx_mpe":
                self.mixer = MultiTaskAttn2HiddenXMPEQMixer(self.surrogate_decomposer, main_args)
                
            else:
                raise ValueError(f"Mixer {main_args.mixer} not recognised.")
            self.params += list(self.mixer.parameters())  # 将混合器的参数加入到params中
            self.target_mixer = copy.deepcopy(self.mixer)  # 深拷贝混合器


            '修改开始：cyd'
            # 定义一组目标混合网络
            self.target_pi_mixers = {}
            for task_name in self.task2args:
                self.target_pi_mixers[task_name] = copy.deepcopy(self.pi_mixers[task_name])
            '修改结束：cyd'
                

        self.optimiser = RMSprop(params=self.params, lr=main_args.lr, alpha=main_args.optim_alpha,
                                 eps=main_args.optim_eps)
        
        '修改开始：cyd'
        # 定义一个字典存储pi_i的参数, 把混合器的参数加入params中
        self.param_pi_is = {}
        for task_name in self.task2args:
            self.param_pi_is[task_name] = list(mac.parameters_pi_i(task_name))+list(self.pi_mixers[task_name].parameters())
        
        #定义一组优化器
        self.optimiser_pi_is = {}
        for task_name in self.task2args:
            self.optimiser_pi_is[task_name] = RMSprop(params=self.param_pi_is[task_name], lr=main_args.lr, alpha=main_args.optim_alpha, eps=main_args.optim_eps)
        
        '修改结束：cyd'
        self.batchs = {}


        # 定义target_mac
        self.target_mac = copy.deepcopy(mac)  # 深拷贝mac
        # 为每一个任务定义一些属性
        self.task2train_info, self.task2encoder_params, self.task2encoder_optimiser = {}, {}, {}
        self.task2repre_dir = {}
        for task in self.task2args:
            task_args = self.task2args[task]
            self.task2train_info[task] = {}
            self.task2train_info[task]["log_stats_t"] = -task_args.learner_log_interval - 1
            # define task_encoder optimiser for this task
            self.task2train_info[task]["to_do_dynamic_learning"] = getattr(task_args, "pretrain", False)
            self.task2encoder_params[task] = list(self.mac.task_encoder_parameters(task))  # no repre parameters
            self.task2encoder_optimiser[task] = RMSprop(params=self.task2encoder_params[task], lr=task_args.lr,
                                                        alpha=task_args.optim_alpha, eps=task_args.optim_eps)
            # 初始化任务表示保存目录
            if self.main_args.save_repre:
                self.task2repre_dir[task] = os.path.join(self.main_args.output_dir, "task_repre", task)
                self.task2train_info[task]["repre_saved"] = False
                os.makedirs(self.task2repre_dir[task], exist_ok=True)
  
    
    def dynamic_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        # Get the relevant quantities
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # Calculate prediction loss
        # 对应论文中Figure 1. Transfer learning scheme of our method的左上角的部分
        obs_pred, state_pred, reward_pred = [], [], []
        for t in range(batch.max_seq_length):
            obs_preds, state_preds, reward_preds = self.mac.task_encoder_forward(batch, t=t, task=task)
            obs_pred.append(obs_preds)
            state_pred.append(state_preds)
            reward_pred.append(reward_preds)
        obs_pred = th.stack(obs_pred, dim=1)[:, :-1]
        state_pred = th.stack(state_pred, dim=1)[:, :-1]
        reward_pred = th.stack(reward_pred, dim=1)[:, :-1]
        # get target labels
        obs = batch["obs"][:, 1:].detach().clone()
        state = batch["state"][:, 1:].detach().clone().unsqueeze(2).repeat(1, 1, self.task2n_agents[task], 1)
        repeated_rewards = batch["reward"][:, :-1].detach().clone().unsqueeze(2).repeat(1, 1, self.task2n_agents[task],
                                                                                        1)

        # calculate prediction loss
        pred_obs_loss = th.sqrt(((obs_pred - obs) ** 2).sum(dim=-1))
        pred_state_loss = th.sqrt(((state_pred - state) ** 2).sum(dim=-1))
        pred_reward_loss = ((reward_pred - repeated_rewards) ** 2).squeeze(dim=-1)

        mask = mask.expand_as(pred_reward_loss)

        # do loss mask
        pred_obs_loss = (pred_obs_loss * mask).mean(dim=-1).sum() / mask.sum()
        pred_state_loss = (pred_state_loss * mask).mean(dim=-1).sum() / mask.sum()
        pred_reward_loss = (pred_reward_loss * mask).mean(dim=-1).sum() / mask.sum()

        task_repre_loss = pred_obs_loss + pred_state_loss + 10 * pred_reward_loss

        self.task2encoder_optimiser[task].zero_grad()
        task_repre_loss.backward()
        pred_grad_norm = th.nn.utils.clip_grad_norm_(self.task2encoder_params[task],
                                                     self.task2args[task].grad_norm_clip)
        self.task2encoder_optimiser[task].step()

        # get grad norm scalar for tensorboard recording
        try:
            pred_grad_norm = pred_grad_norm.item()
        except:
            pass
        # 首先，如果设置了保存任务表示（save_repre），并且当前任务的任务表示还没有被保存过，那么就调用mac.save_task_repres方法将任务表示保存到指定的文件中，
        # 并将self.task2train_info[task]["repre_saved"]设置为True，表示该任务的任务表示已经被保存过了。
        if self.main_args.save_repre and not self.task2train_info[task]["repre_saved"]:
            self.mac.save_task_repres(os.path.join(self.task2repre_dir[task], f"{t_env}.npy"), task)
            self.task2train_info[task]["repre_saved"] = True

        # 然后，如果当前环境步数（t_env）减去上一次记录训练信息的环境步数（self.task2train_info[task]["log_stats_t"]）大于或等于任务的学习日志间隔（self.task2args[task].learner_log_interval），
        # 那么就记录一些训练信息，包括预测观察损失、预测状态损失、预测奖励损失和任务编码器梯度范数，并更新self.task2train_info[task]["log_stats_t"]为当前环境步数。
        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            # Dynamic learning phase
            self.logger.log_stat(f"{task}/pred_obs_loss", pred_obs_loss.item(), t_env)
            self.logger.log_stat(f"{task}/pred_state_loss", pred_state_loss.item(), t_env)
            self.logger.log_stat(f"{task}/pred_reward_loss", pred_reward_loss.item(), t_env)
            self.logger.log_stat(f"{task}/task_encoder_grad_norm", pred_grad_norm, t_env)
            self.task2train_info[task]["log_stats_t"] = t_env

        # 如果当前环境步数（t_env）大于动态学习结束的步数（self.task2args[task].dynamic_learning_end），那么就停止动态学习
        if t_env > self.task2args[task].dynamic_learning_end:
            self.task2train_info[task][
                "to_do_dynamic_learning"] = False  # 将self.task2train_info[task]["to_do_dynamic_learning"]设置为False，
            self.task2train_info[task]["log_stats_t"] = -self.task2args[task].learner_log_interval - 1
            if self.main_args.save_repre:  # 如果设置了保存任务表示（save_repre），那么就再次保存任务表示
                self.mac.save_task_repres(os.path.join(self.task2repre_dir[task], f"{t_env}.npy"), task)
            return True  # 返回True表示动态学习已经结束。


    
    # 用distral论文中的方法，最小化log（pi_0）
    def optimize_pi_0(self, c_batch: EpisodeBatch, t_env: int, episode_num: int, c_task: str, cal = False):
        if not cal:
            self.batchs[c_task] = copy.deepcopy(c_batch)
            return
        
        pi_0_loss = 0
        
        for task in self.batchs:
            batch = self.batchs[task]
            # Get the relevant quantities
            rewards = batch["reward"][:, :-1]
            actions = batch["actions"][:, :-1]
            terminated = batch["terminated"][:, :-1].float()
            mask = batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            avail_actions = batch["avail_actions"]

            # 计算qval和target qval
            pi_0_mac_out = []
            self.mac.init_hidden(batch.batch_size, task)

            for t in range(batch.max_seq_length):
                pi_0_agent_outs = self.mac.forward(batch, t=t, task=task)
                pi_0_mac_out.append(pi_0_agent_outs)

            pi_0_mac_out = th.stack(pi_0_mac_out, dim=1)  # Concat over time
            # 进行log_softmax
            pi_0_mac_out = th.log_softmax(pi_0_mac_out, dim=-1)

            # Pick the Q-Values for the actions taken by each agent
            pi_0_chosen_action_qvals = th.gather(pi_0_mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

            seq_len = pi_0_chosen_action_qvals.shape[1]
            
            for t_inx  in range(seq_len):
                pi_0_chosen_action_qvals[:, t_inx, :] = pi_0_chosen_action_qvals[:, t_inx, :]*pow(self.main_args.gamma, t)
            
            mask = mask.expand_as(pi_0_chosen_action_qvals)

            pi_0_loss += (mask*pi_0_chosen_action_qvals).sum()/mask.sum()
        
        if pi_0_loss == 0:
            return
        
        self.optimiser.zero_grad()
        pi_0_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.main_args.grad_norm_clip)
        self.optimiser.step()

        try:
            grad_norm = grad_norm.item()
        except:
            pass

        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            self.logger.log_stat(f"{task}/pi_0_loss", pi_0_loss.item(), t_env)
            self.logger.log_stat(f"{task}/grad_norm", grad_norm, t_env)
        
        # 最后把batchs清空
        self.batchs.clear()
        

        
    # 固定pi_0优化每个环境的单独的pi_i
    # 优化pi_i的过程
    def optimize_pi_i(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        # Get the relevant quantities
        # reward直接除以智能体数目，后面计算next_qvals时会进行广播
        rewards = batch["reward"][:, :-1] # tensor(32, 65, 1)
        actions = batch["actions"][:, :-1]  # tensor(32, 65, 10, 1)
        terminated = batch["terminated"][:, :-1].float()  # tensor(32, 65, 1)
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]  # tensor(32, 66, 10, 18)

        
        # 1. 计算当前状态的Q(mixer级别)
        ## 计算当前状态pi_i的Q值
        mac_out = []
        self.mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward_pi_i(batch, t=t, task=task)  # tensor(32, 10, 18)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time tensor:(32, 68 , 10, 18)      
        
        ## 根据动作选出每个智能体的Q值，注意这里是去掉序列最后一个，算target_val的时候是去掉的第一个，因为计算target_val的时候用的是next_state
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # tensor(32, 67, 10)
        ## 获取batch_size和seq_len
        bs, seq_len = chosen_action_qvals.size(0), chosen_action_qvals.size(1)
        ## 获取任务表征
        task_repre = self.mac.get_task_repres(task, require_grad=False)[None, None, ...].repeat(bs, seq_len, 1, 1)
        
        if self.pi_mixers[task] is not None:
            # 经过混合网络得到Q_tot
            chosen_action_qvals = self.pi_mixers[task](chosen_action_qvals, batch["state"][:, :-1]) #tensor(32, 69, 1)
        
        # 2. 计算next_state_val(在target_mac上)
        ## 先计算pi_0（agent级别），pi_0还是在mac上计算
        p_mac_out = []
        self.mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t, task=task)  # tensor(32, 10, 18)
            p_mac_out.append(agent_outs)
        
        '''改一下，改成1：'''
        p_mac_out = th.stack(p_mac_out[1:], dim=1)  # Concat over time tensor:(32, 68 , 10, 18)

        
        ## 在计算next_qval，pi在target_mac上
        next_mac_out = []
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            agent_outs = self.target_mac.forward_pi_i(batch, t=t, task=task)  # tensor(32, 10, 18)
            next_mac_out.append(agent_outs)
            
        next_mac_out = th.stack(next_mac_out[1:], dim=1)  # tensor（32，69，10，18）
        next_mac_out[avail_actions[:, 1:] == 0] = -9999999
        
        ## 通过max算子得到下一个状态的最大Q值动作
        next_state_maxq_action = next_mac_out.max(dim=3)[1] #tensor(32, 69, 10)

        # 动作数和智能体数目
        action_num = next_mac_out.shape[-1] #18
        agent_num = next_mac_out.shape[2] # 10
        
        # 构建近似联合动作空间
        # 采样数变为原来的1/4
        
        action_space = []
        for agent in range(agent_num):
            for t in range(0, action_num, 4):
                if t==0:
                    if agent==0:
                        a = copy.deepcopy(next_state_maxq_action) 
                        action_space.append(a)
                else:
                    a = copy.deepcopy(next_state_maxq_action) 
                    a[:, :, agent] += t
                    a[:, :, agent] %= action_num
                    action_space.append(a)
        
        # 遍历近似动作空间中的每一个动作计算pi_0
        pi_0_s = []
        for action in action_space:
            action = action.unsqueeze(-1)
            pi_0 = p_mac_out.gather(dim=-1, index=action) # tensor(32, 69, 10, 1)
            
            bs, seq_len = pi_0.size(0), pi_0.size(1)
            task_repre = self.mac.get_task_repres(task, require_grad=False)[None, None, ...].repeat(bs, seq_len, 1, 1)
            if self.mixer is not None:
                '''改一下，改成1：'''
                pi_0 = self.mixer(pi_0, batch["state"][:, 1:], task_repre, self.task2decomposer[task])
                
            pi_0_s.append(pi_0)
        pi_0_s = th.stack(tensors=pi_0_s, dim=2).squeeze(-1) # tensor(batch_size, seq_len, pi_0_num)    
        
        # 进行softmax
        pi_0_s = th.softmax(pi_0_s, dim=-1)
        
        V_i = []
        for action_inx in range(len(action_space)):
            action = action_space[action_inx].unsqueeze(-1)
            
            # 从pi_0_s中根据action的索引得到pi_0
            pi_0 = pi_0_s[:,:,action_inx].unsqueeze(-1) # tensor(32, 69, 1)
            
            Q_i = next_mac_out.gather(dim=-1, index=action)
            # Q_i要经过mixer得到Qtot
            bs, seq_len = Q_i.size(0), Q_i.size(1)
            task_repre = self.target_mac.get_task_repres(task, require_grad=False)[None, None, ...].repeat(bs, seq_len, 1, 1)
            if self.target_pi_mixers[task] is not None:
                '''这里也改成1：'''
                '''之前忘记改了，现在改一下也不晚，跑两个sz和marine试试'''
                Q_i = self.target_pi_mixers[task](Q_i, batch["state"][:, 1:])
            
            # 根据公式计算
            v= th.pow(pi_0.detach(), self.main_args.alpha) * th.exp(self.main_args.beta * Q_i)
            V_i.append(v)
            
        V_i = th.log(sum(V_i))/self.main_args.beta
        expected_action_val = rewards + self.main_args.gamma * (1 - terminated) * V_i
        # 计算得到损失
        loss = chosen_action_qvals-expected_action_val.detach()
        mask = mask.expand_as(loss)
        mask_loss = loss*mask
        loss = (mask_loss**2).sum()/mask.sum()
        
        # # 还得计算一下正则项
        # re_loss = self.cal_res_loss(batch, chosen_action_qvals)
        
        # loss += re_loss*self.main_args.lamda
         
        # 优化
        self.optimiser_pi_is[task].zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.param_pi_is[task], self.main_args.grad_norm_clip)
        self.optimiser_pi_is[task].step()
        
        # 打印在tensorboard上
        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            self.logger.log_stat(f"{task}/optimize_model_loss", loss.item(), t_env)
            self.logger.log_stat(f"{task}/optimize_model_grad_norm", grad_norm, t_env)
            self.task2train_info[task]["log_stats_t"] = t_env
            
            # 记录胜率
            self.logger.stats[task] = self.logger.stats[task + "/test_battle_won_mean"][-1][-1]
    
    def rl_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str, cal = False):
        self.optimize_pi_i(batch, t_env, episode_num, task)
        self.optimize_pi_0(batch, t_env, episode_num, task, cal)

        # 更新self.task2train_info[task]["log_stats_t"]
        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            self.task2train_info[task]["log_stats_t"] = t_env

        # 更新target
        if (episode_num - self.last_target_update_episode) / self.main_args.target_update_interval >= 1.0:
            self._update_targets()
            self._update_targets_pi_i()
            self.last_target_update_episode = episode_num

    '修改开始：cyd'
    # 更新目标网络
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network pi_0")

    def _update_targets_pi_i(self):
        self.target_mac.load_state_pi_i(self.mac)
        for task_name in self.task2args:
            if self.pi_mixers[task_name] is not None:
                self.target_pi_mixers[task_name].load_state_dict(self.pi_mixers[task_name].state_dict())
        self.logger.console_logger.info("Updated target network pi_is")
                
    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        # 将pi_mixer和target_pi_mixer放到cuda上
        for task_name in self.task2args:
            if self.pi_mixers[task_name] is not None:
                self.pi_mixers[task_name].cuda()
            if self.target_pi_mixers[task_name] is not None:
                self.target_pi_mixers[task_name].cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        
        # 保存pi_mixer
        for task_name in self.task2args:
            if self.pi_mixers[task_name] is not None:
                th.save(self.pi_mixers[task_name].state_dict(), "{}/{}_mixer.th".format(path, task_name))
        
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        # 保存新定义的一组优化器
        for task_name in self.task2args:
            th.save(self.optimiser_pi_is[task_name].state_dict(), "{}/{}_opt.th".format(path, task_name))

    # 只需要加载pi_0即可
    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        # 加载pi_mixer
        for task_name in self.task2args:
            if self.pi_mixers[task_name] is not None:
                self.pi_mixers[task_name].load_state_dict(th.load("{}/{}_mixer.th".format(path, task_name), map_location=lambda storage, loc: storage))
        
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        # 加载各个任务的优化器
        for task_name in self.task2args:
            self.optimiser_pi_is[task_name].load_state_dict(th.load("{}/{}_opt.th".format(path, task_name), map_location=lambda storage, loc: storage))


    def kl_divergence_loss(self, q_t, q_s, temperature, alpha=0.5):
        # 将教师和学生的动作价值估计除以温度参数，让分布更平滑
        q_t = q_t / temperature
        q_s = q_s / temperature

        # 使用softmax函数计算概率分布
        p_t = th.nn.functional.softmax(q_t, dim=-1)
        p_s = th.nn.functional.softmax(q_s, dim=-1)

        # 计算教师和学生概率分布的对数
        log_p_t = th.nn.functional.log_softmax(q_t, dim=-1)
        log_p_s = th.nn.functional.log_softmax(q_s, dim=-1)

        # 计算教师和学生概率分布之间的KL散度
        kl_div_ts = th.sum(p_t * (log_p_t - log_p_s), dim=-1)
        kl_div_st = th.sum(p_s * (log_p_s - log_p_t), dim=-1)

        # 计算Jeffrey's散度损失
        loss = (alpha * kl_div_ts) + ((1.0 - alpha) * kl_div_st)

        return loss

    # 计算正则项损失
    def cal_res_loss(self, batch, q_vals):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # 还得计算一下正则项
        base_line = copy.deepcopy(rewards)
        seq_len = base_line.shape[1]
        for t_inx  in range(seq_len):
            for i_inx in range(t_inx, seq_len):
                base_line[:, t_inx, :]=base_line[:, t_inx, :]+base_line[:, i_inx, :]*pow(self.main_args.gamma, i_inx-t_inx)
        
        mask_re_loss = (q_vals-base_line)*mask
        re_loss = (mask_re_loss**2).sum()/mask.sum()
        
        return re_loss*self.main_args.lamda
    
    def bolz_policy(self, pi_0_mac_out, pi_i_mac_out):
         # 导出增强策略
        pi0 = th.nn.functional.softmax(pi_0_mac_out[:, :-1], dim=-1) # tensor(2, 12, 20)
        pi0 = pi0.detach()
        Q = pi_i_mac_out.detach()
        
        V = th.log((th.pow(pi0, self.main_args.alpha)*th.exp(self.main_args.beta*Q)).sum(-1))/self.main_args.beta # tensor(2, 12)
        
        #对V进行扩展让V最后一个维度和Q相同
        vsz = V.size()
        V = V.unsqueeze(-1)
        V = V.expand(*vsz, Q.shape[-1])
        
        # 导出增强策略
        bolz_policy = th.pow(pi0, self.main_args.alpha)* th.exp(self.main_args.beta*(Q-V))

        # bolz_policy_chose_maxq_action = bolz_policy.max(dim=3)[1] #tensor(32, 69, 10)
        return bolz_policy
