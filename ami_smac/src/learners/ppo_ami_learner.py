import copy
from turtle import position
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from modules.critics import REGISTRY as critic_resigtry
from components.standarize_stream import RunningMeanStd
from components.action_selectors import REGISTRY as action_REGISTRY
import numpy as np


class PPOAMILearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        device = "cuda" if args.use_cuda else "cpu"
        self.device = device
        if self.args.mode == "adv_policy" or self.args.mode == "adv_state":
            self.n_agents = args.n_adv_agents
        else:
            self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.prev_mac = copy.deepcopy(mac)
        self.cal_mac = copy.deepcopy(mac)

        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)
        self.maa_params = list(mac.maa_parameters())
        self.maa_optimiser = Adam(params=self.maa_params, lr=args.lr)
        
        self.oracle_params = list(mac.oracle_parameters())
        self.oracle_optimiser = Adam(params=self.oracle_params, lr=args.lr)
        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.oracleCritic = critic_resigtry[args.oracle_use_criticType](scheme, args).to(self.device)
        self.target_oracleCritic = copy.deepcopy(self.oracleCritic)

        self.oracleCritic_params = list(self.oracleCritic.parameters())
        self.oracleCritic_optimiser = Adam(params=self.oracleCritic_params, lr=args.lr)

        self.critic = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = - self.args.learner_log_interval - 1

        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)
            self.rew_clean_ms = RunningMeanStd(shape=(1,), device=device)

        self.adv_agent_ids = []

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.get_adv_agent_id(batch)
        rewards = batch["reward"][:, :-1]
        sensitivity_reward = th.zeros_like(rewards)
        minority_influence, majority_influence = 0, 0

        log_oracle_out = self.get_oracle_out(batch, t_env, episode_num)
        oracle_out = th.nn.functional.softmax(log_oracle_out, dim=-1)
        oracle_out_selected = self.action_selector.select_action(oracle_out, batch["avail_actions_all"], 0, False)
        oracle_out = th.nn.functional.one_hot(oracle_out_selected, num_classes=self.args.n_actions)

        if self.args.use_target_sensitivity_reward:
            sensitivity_reward = self.get_target_sensitivity_reward(batch, t_env, episode_num, oracle_out).detach()
            rewards_clean = rewards.clone()
            if self.args.standardise_rewards:
                self.rew_clean_ms.update(rewards_clean)
                rewards_clean = (rewards_clean - self.rew_clean_ms.mean) / th.sqrt(self.rew_clean_ms.var)
            
            rewards = rewards - self.args.target_sensitivity_reward_param * sensitivity_reward

        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if mask.sum() == 0:
            self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
            self.logger.console_logger.error(
                "Actor Critic Learner: mask.sum() == 0 at t_env {}".format(t_env))
            return

        mask_pred = mask.repeat(1, 1, 1)
        mask = mask.repeat(1, 1, self.n_agents)
        critic_mask = mask.clone()

        self.train_maa(batch, t_env, episode_num)

        old_mac_out = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.old_mac.forward(batch, t=t)
            old_mac_out.append(agent_outs)
        old_mac_out = th.stack(old_mac_out, dim=1)  # Concat over time
        old_pi = old_mac_out
        old_pi[mask == 0] = 1.0
        old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3) 
        old_log_pi_taken = th.log(old_pi_taken + 1e-10)

        log_old_oracle_out = self.get_old_oracle_out(batch, t_env, episode_num)

        for k in range(self.args.epochs):
            advantages, critic_train_stats = self.train_critic_sequential(self.critic, self.target_critic, batch,
                                                                          rewards,
                                                                          critic_mask)
            advantages = advantages.detach()
            log_oracle_out = self.get_oracle_out(batch, t_env, episode_num)
            oracle_out = th.nn.functional.softmax(log_oracle_out, dim=-1)
            
            oracle_advantages, oracleCritic_train_stats = self.train_oracleCritic_sequential(self.oracleCritic, self.target_oracleCritic, batch, rewards, critic_mask)
            oracle_advantages = oracle_advantages.detach()
            oracle_advantages = th.cat([oracle_advantages[:, 1:, :], th.zeros_like(oracle_advantages[:, :1, :]).cuda()], dim=1)
            
            log_oracle_out_taken = th.gather(log_oracle_out, dim=3, index=oracle_out_selected.unsqueeze(3)).squeeze(3)
            log_old_oracle_out_taken = th.gather(log_old_oracle_out, dim=3, index=oracle_out_selected.unsqueeze(3)).squeeze(3)
            oracle_ratios = th.exp(log_oracle_out_taken - log_old_oracle_out_taken.detach())
            
            oracle_surr1 = oracle_ratios * oracle_advantages
            oracle_surr2 = th.clamp(oracle_ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * oracle_advantages
            oracle_entropy = 0
            oracle_pg_loss = -((th.min(oracle_surr1, oracle_surr2) +
                                   self.args.entropy_coef * oracle_entropy) * mask_pred).sum() / mask_pred.sum()

            self.oracle_optimiser.zero_grad()
            oracle_pg_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.oracle_params, self.args.grad_norm_clip)
            self.oracle_optimiser.step()
            mac_out_no_softmax = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward_no_softmax(batch, t=t)
                mac_out_no_softmax.append(agent_outs)

            mac_out_no_softmax = th.stack(mac_out_no_softmax, dim=1)  # Concat over time

            mac_out = th.nn.functional.softmax(mac_out_no_softmax, dim=-1)
            pi = mac_out.clone()

            pi[mask == 0] = 1.0

            pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
            log_pi_taken = th.log(pi_taken + 1e-10)

            ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratios * advantages
            surr2 = th.clamp(ratios, 1 - self.args.eps_clip,1 + self.args.eps_clip) * advantages

            entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
            pg_loss = -((th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

            self.agent_optimiser.zero_grad()
            pg_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()

        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (
                self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
            self.prev_mac.load_state(self.mac)
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat("learner/" + key, sum(critic_train_stats[key]) / ts_logged, t_env)

            self.logger.log_stat(
                "learner/advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("learner/pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("learner/sensitivity_reward", th.sum(sensitivity_reward), t_env)
            self.logger.log_stat("learner/minority_influence", minority_influence, t_env)
            self.logger.log_stat("learner/majority_influence", majority_influence, t_env)
            self.logger.log_stat("learner/total_loss",
                                 pg_loss.item(), t_env)
            self.logger.log_stat("learner/agent_grad_norm",
                                 grad_norm.item(), t_env)
            self.logger.log_stat("learner/pi_max", (pi.max(dim=-1)
                                 [0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def train_maa(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        actions_all = batch["actions_all"][:, 1:]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions_all = actions_all.unsqueeze(2).expand(-1, -1, 1, -1, -1)

        if mask.sum() == 0:
            self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
            self.logger.console_logger.error(
                "Actor Critic Learner: mask.sum() == 0 at t_env {}".format(t_env))
            return

        maa_out = []
        self.mac.init_hidden_maa(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            maa_outs = self.mac.forward_maa(batch, t=t)
            maa_out.append(maa_outs)
        maa_out = th.stack(maa_out, dim=1)
        maa_mask = th.logical_not(batch['adv_agent_mask'][:, 1:, :, :]).int()
        actions_all = th.einsum('ijklm, ijlm -> ijklm', actions_all, maa_mask)
        maa_out = th.einsum('ijklm, ijlm -> ijklm', maa_out, maa_mask)
        maa_out = maa_out.reshape(-1, self.n_actions)

        ts = actions_all.shape[1]
        reshaped_actions_all = actions_all.reshape(-1)

        pred = maa_out.argmax(dim=1)
        correct = (pred == reshaped_actions_all).reshape(batch.batch_size, ts, -1)
        correct = correct.float().mean(dim=2, keepdim=True)
        acc = (correct * mask).sum() / mask.sum()

        for k in range(self.args.epochs):
            maa_out = []
            self.mac.init_hidden_maa(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                maa_outs = self.mac.forward_maa(batch, t=t)
                maa_out.append(maa_outs)
            maa_out = th.stack(maa_out, dim=1)
            maa_out = th.einsum('ijklm, ijlm -> ijklm', maa_out, maa_mask)
            maa_out = maa_out.reshape(-1, self.n_actions)
            maa_out_entropy = - th.sum(th.einsum('ij, ij -> ij', maa_out, th.log(maa_out + 1e-10)))

            maa_loss = cross_entropy(maa_out, reshaped_actions_all, reduction="none")
            maa_loss = maa_loss.reshape(batch.batch_size, ts, -1).mean(dim=2, keepdim=True)
            maa_loss = (maa_loss * mask).sum() / mask.sum()
            maa_loss = maa_loss + self.args.entropy_reg_param * maa_out_entropy

            self.maa_optimiser.zero_grad()
            maa_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.maa_params, self.args.grad_norm_clip)
            self.maa_optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("learner/maa_loss", maa_loss.item(), t_env)
            self.logger.log_stat("learner/maa_train_acc", acc.item(), t_env)
            self.logger.log_stat("learner/maa_grad_norm", grad_norm.item(), t_env)
        

    def get_oracle_out(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.mac.init_hidden_oracle(batch.batch_size)
        oracle_out = []
        for t in range(batch.max_seq_length - 1):
            oracle_outs = self.mac.forward_oracle_logsoftmax(batch, t=t)
            oracle_out.append(oracle_outs)
        oracle_out = th.stack(oracle_out, dim=1)
        return oracle_out

    def get_old_oracle_out(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.old_mac.init_hidden_oracle(batch.batch_size)
        oracle_out = []
        for t in range(batch.max_seq_length - 1):
            oracle_outs = self.old_mac.forward_oracle_logsoftmax(batch, t=t)
            oracle_out.append(oracle_outs)
        oracle_out = th.stack(oracle_out, dim=1)
        return oracle_out

    def cal_position_mask(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        obs = batch["obs"]
        if self.args.env == "sc2":
            start = self.args.position_start
            end = (self.args.n_agents - 1) * self.args.position_pace + start
            position_mask = obs[:, :, :, start:end:self.args.position_pace]
            
            position_new = []
            for i in range(self.args.batch_size):
                pos_new_tmp = []
                for j in range(self.args.n_adv_agents):
                    pos_new_tmp.append(th.cat([position_mask[i, :, j, :self.adv_agent_ids[i, j]], th.zeros_like(position_mask[i, :, j, 0]).unsqueeze(-1), position_mask[i, :, j, self.adv_agent_ids[i, j]:]], dim=-1))
                pos_new_tmp = th.stack(pos_new_tmp, dim=-2)
                position_new.append(pos_new_tmp)
            position_new = th.stack(position_new, dim=0)

            position_final = th.zeros_like(position_new[:, :, 0, :])
            for i in range(self.args.n_adv_agents):
                position_final = th.logical_or(position_final, position_new[:, :, i, :])
            position_final = position_final.unsqueeze(2)
        else:
            position_final = th.ones_like(obs)[:, :, :, 0].unsqueeze(2)
        
        if not self.args.use_position_mask:
            position_final = th.ones_like(position_final)
            position_final[:, :, :, 0] = th.zeros_like(position_final[:, :, :, 0])
        return position_final.int()


    def cal_active_mask(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        obs = batch["obs"]
        start = self.args.n_agents * 5 + 4 + 4
        end = (self.args.n_agents - 1) * 5 + start
        position_mask = th.where(obs[:, :, :, start:end:5]==0, 0, 1)

        position_new = []
        for i in range(self.args.batch_size):
            pos_new_tmp = []
            for j in range(self.args.n_adv_agents):
                pos_new_tmp.append(th.cat([position_mask[i, :, j, :self.adv_agent_ids[i, j]], th.zeros_like(position_mask[i, :, j, 0]).unsqueeze(-1), position_mask[i, :, j, self.adv_agent_ids[i, j]:]], dim=-1))
            pos_new_tmp = th.stack(pos_new_tmp, dim=-2)
            position_new.append(pos_new_tmp)
        position_new = th.stack(position_new, dim=0)

        position_final = th.zeros_like(position_new[:, :, 0, :])
        for i in range(self.args.n_adv_agents):
            position_final = th.logical_or(position_final, position_new[:, :, i, :])
        position_final = position_final.unsqueeze(2)
        
        if not self.args.use_position_mask:
            position_final = th.ones_like(position_final)

        return position_final.int()

    def get_adv_agent_mask(self, batch):
        return batch["adv_agent_mask"][:, 0, :, 0].int()

    def get_adv_agent_id(self, batch):
        adv_mask = self.get_adv_agent_mask(batch)
        self.adv_agent_ids = []
        for i in range(self.args.batch_size):
            adv_agent_tmp = []
            for j in range (self.args.n_agents):
                if adv_mask[i][j] == 1:
                    adv_agent_tmp.append(j)
            adv_agent_tmp = th.tensor(np.array(adv_agent_tmp))
            self.adv_agent_ids.append(adv_agent_tmp)
        self.adv_agent_ids = th.stack(self.adv_agent_ids, dim=0)
        return self.adv_agent_ids

    def get_target_sensitivity_reward(self, batch: EpisodeBatch, t_env: int, episode_num: int, oracle_out):
        reward = self.cal_sensitivity_target(batch, t_env, episode_num, oracle_out)
        return reward

    def cal_sensitivity_target(self, batch: EpisodeBatch, t_env: int, episode_num: int, oracle_out):
        avail_actions_all = batch["avail_actions_all"][:, 1:]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        adv_agent_id = self.get_adv_agent_id(batch)
        assert adv_agent_id.shape[-1] == 1

        maa_out_all = []
        for k in range(10):
            mac_out_no_softmax = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward_no_softmax(batch, t=t)
                mac_out_no_softmax.append(agent_outs)
            mac_out_no_softmax = th.stack(mac_out_no_softmax, dim=1)
            mac_out_gumbel = th.nn.functional.gumbel_softmax(mac_out_no_softmax, hard=True, dim=-1).reshape(batch.batch_size, -1,
                                                                                                            self.n_actions)

            if mask.sum() == 0:
                self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
                self.logger.console_logger.error("Actor Critic Learner: mask.sum() == 0 at t_env {}".format(t_env))
                return
            mask = mask.repeat(1, 1, self.n_agents)

            maa_out = []
            self.mac.init_hidden_maa(batch.batch_size)
            
            for t in range(batch.max_seq_length - 1):
                maa_inputs = self.mac._build_maa_inputs(batch, t=t)
                for i in range(self.args.batch_size):
                    input_start = self.mac.state_shape + self.args.n_actions * adv_agent_id[i, 0]
                    maa_inputs[i, input_start:input_start + mac_out_gumbel.shape[2]] = mac_out_gumbel[i, t]
                maa_outs = self.mac.forward_maa_inputs(maa_inputs, batch, t=t)
                maa_out.append(maa_outs.detach())
            maa_out = th.stack(maa_out, dim=1)
            if self.args.use_sampled_maa:
                maa_out = th.argmax(maa_out, dim=-1)
                maa_out = th.nn.functional.one_hot(maa_out, num_classes=self.args.n_actions).float()
            maa_out_all.append(maa_out)
        maa_out = th.mean(th.stack(maa_out_all, dim=1), dim=1)

        avail_actions_all = avail_actions_all.unsqueeze(2).expand_as(maa_out)
        oracle_out = oracle_out.unsqueeze(2)
        position_mask = self.cal_position_mask(batch, t_env, 0)[:, :-1, :, :]
        maa_out = th.einsum('ijkln, ijkl -> ijkln', maa_out, position_mask)
        oracle_out = th.einsum('ijkln, ijkl -> ijkln', oracle_out, position_mask)

        if self.args.sensitivity_metric == 'L1':
            sensitivity = th.norm(maa_out - oracle_out, p=1, dim=-1)
        elif self.args.sensitivity_metric == 'L2':
            sensitivity = th.norm(maa_out - oracle_out, p=2, dim=-1)
        elif self.args.sensitivity_metric == 'Linf':
            sensitivity = th.norm(maa_out - oracle_out, p=float('inf'), dim=-1)
        elif self.args.sensitivity_metric == 'ce':
            sensitivity = th.log(th.sum(maa_out * oracle_out, dim=-1) + 1e-10)
        elif self.args.sensitivity_metric == 'selectprob':
            sensitivity = th.sum(maa_out * oracle_out, dim=-1)
        
        reward = th.sum(sensitivity, dim=-1)
        return reward

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        with th.no_grad():
            target_vals = target_critic(batch)
            target_vals = target_vals.squeeze(3)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns(
            rewards, mask, target_vals, self.args.q_nstep)
        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        v = critic(batch)[:, :-1].squeeze(3)
        td_error = (target_returns.detach() - v)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((target_returns * mask).sum().item() / mask_elems)

        return masked_td_error, running_log

    def train_oracleCritic_sequential(self, critic, target_critic, batch, rewards, mask):
        mask_tmp = th.zeros_like(mask[:, :, :1])
        for i in range(self.args.n_adv_agents):
            mask_tmp = th.logical_or(mask_tmp, mask[..., i].unsqueeze(-1))
        mask = mask_tmp
        with th.no_grad():
            target_vals = target_critic(batch)
            target_vals = target_vals.squeeze(3)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns_oracle(
            rewards, mask, target_vals, self.args.q_nstep)
        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        v = critic(batch)[:, :-1].squeeze(3)
        td_error = (target_returns.detach() - v)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.oracleCritic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.grad_norm_clip)
        self.oracleCritic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((target_returns * mask).sum().item() / mask_elems)

        return masked_td_error, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** (
                        step) * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma ** (
                        step) * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (
                        step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.args.gamma ** (
                        step) * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def nstep_returns_oracle(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** (
                        step) * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma ** (
                        step) * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (
                        step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.args.gamma ** (
                        step) * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_oracleCritic.load_state_dict(self.oracleCritic.state_dict())

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_oracleCritic.load_state_dict(self.oracleCritic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_oracleCritic.parameters(), self.oracleCritic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
                
    def cuda(self):
        self.old_mac.cuda()
        self.mac.cuda()
        self.cal_mac.cuda()
        self.prev_mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()
        self.oracleCritic.cuda()
        self.target_oracleCritic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(),"{}/agent_opt.th".format(path))
        th.save(self.maa_optimiser.state_dict(), "{}/maa_opt.th".format(path))
        th.save(self.oracle_optimiser.state_dict(), "{}/oracle_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
        th.save(self.oracleCritic.state_dict(), "{}/oraclecritic.th".format(path))
        th.save(self.oracleCritic_optimiser.state_dict(), "{}/oraclecritic_opt.th".format(path))


    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.oracle_optimiser.load_state_dict(th.load("{}/oracle_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.maa_optimiser.load_state_dict(th.load("{}/maa_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.oracleCritic.load_state_dict(th.load("{}/oraclecritic.th".format(path), map_location=lambda storage, loc: storage))
        self.target_oracleCritic.load_state_dict(self.oracleCritic.state_dict())
        self.oracleCritic_optimiser.load_state_dict(th.load("{}/oraclecritic_opt.th".format(path), map_location=lambda storage, loc: storage))


    def load_models_without_optim(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.target_critic.load_state_dict(self.critic.state_dict())
