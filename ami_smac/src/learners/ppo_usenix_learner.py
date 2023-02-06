import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import Adam
from modules.critics import REGISTRY as critic_resigtry
from components.standarize_stream import RunningMeanStd


class PPOUsenixLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        if self.args.mode == "adv_policy":
            self.n_agents = args.n_adv_agents
        else:
            self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.state_params = list(mac.state_parameters())
        self.state_optimiser = Adam(params=self.state_params, lr=args.lr)
        self.action_params = list(mac.action_parameters())
        self.action_optimiser = Adam(params=self.action_params, lr=args.lr)

        self.critic = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents, ), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # No experiences to train on in this minibatch
        if mask.sum() == 0:
            self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
            self.logger.console_logger.error("Actor Critic Learner: mask.sum() == 0 at t_env {}".format(t_env))
            return

        mask = mask.repeat(1, 1, self.n_agents)
        critic_mask = mask.clone()

        # forward target policy network
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

        for k in range(self.args.epochs):
            # USENIX: update state model and action model
            state_train_stats = self.train_state_model(batch, t_env, episode_num)
            action_train_stats = self.train_action_model(batch, t_env, episode_num)

        for k in range(self.args.epochs):
            # forward policy network
            mac_out_no_softmax = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward_no_softmax(batch, t=t)
                mac_out_no_softmax.append(agent_outs)
            mac_out_no_softmax = th.stack(mac_out_no_softmax, dim=1)  # Concat over time

            # USENIX: calculate adv loss
            lamb = self.get_lambda(batch, t_env, episode_num)
            adv_loss, adv_train_stats = self.get_adv_loss(mac_out_no_softmax, lamb, batch, t_env, episode_num)

            mac_out = th.nn.functional.softmax(mac_out_no_softmax, dim=-1)
            pi = mac_out.clone()
            
            # calculate advantages
            advantages, critic_train_stats = self.train_critic_sequential(self.critic, self.target_critic, batch, rewards,
                                                                          critic_mask)
            advantages = advantages.detach()

            # Calculate policy grad with mask
            
            pi[mask == 0] = 1.0

            pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
            log_pi_taken = th.log(pi_taken + 1e-10)

            ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratios * advantages
            surr2 = th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages

            entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
            pg_loss = -((th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

            total_loss = pg_loss + adv_loss

            # Optimise agents
            self.agent_optimiser.zero_grad()
            total_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()

        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (
                self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat("learner/" + key, sum(critic_train_stats[key]) / ts_logged, t_env)

            for key in state_train_stats.keys():
                self.logger.log_stat("learner/" + key, sum(state_train_stats[key]) / len(state_train_stats[key]), t_env)

            for key in action_train_stats.keys():
                self.logger.log_stat("learner/" + key, sum(action_train_stats[key]) / len(action_train_stats[key]), t_env)

            for key in adv_train_stats.keys():
                self.logger.log_stat("learner/" + key, sum(adv_train_stats[key]) / len(adv_train_stats[key]), t_env)

            self.logger.log_stat("learner/advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("learner/pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("learner/lambda_mean", lamb.mean().item(), t_env)
            self.logger.log_stat("learner/agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("learner/pi_max", (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        # Optimise critic
        with th.no_grad():
            target_vals = target_critic(batch)
            target_vals = target_vals.squeeze(3)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns(rewards, mask, target_vals, self.args.q_nstep)
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
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((target_returns * mask).sum().item() / mask_elems)

        return masked_td_error, running_log

    def train_state_model(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # get relavant variables, without mask sum check
        obs_tplus1 = batch["obs_all"][:, 1:]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.squeeze(2)

        running_log = {
            "state_gap_linf_norm": [],
            "state_loss": [],
            "state_grad_norm": []
        }
        mac_out = []
        self.mac.init_hidden_state(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward_state(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # [batch_size, timestep, n_agents, obs_shape]

        state_gap = th.abs(mac_out - obs_tplus1)[mask != 0]
        state_gap_linf_norm = th.norm(state_gap, p=float("inf"))
        state_loss = th.norm(state_gap - self.args.epsilon_state, p=2)

        self.state_optimiser.zero_grad()
        state_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.state_params, self.args.grad_norm_clip)
        self.state_optimiser.step()

        running_log["state_gap_linf_norm"].append(state_gap_linf_norm.item())
        running_log["state_loss"].append(state_loss.item())
        running_log["state_grad_norm"].append(grad_norm.item())

        return running_log

    def train_action_model(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # get relavant variables, without mask sum check
        actions_all = batch["actions_all"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.squeeze(2)

        running_log = {
            "action_loss": [],
            "action_grad_norm": []
        }
        mac_out = []
        self.mac.init_hidden_action(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward_action(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # [batch_size, timestep, n_victim_agents, n_actions]

        actions_all = actions_all[:, :, self.n_agents:][mask != 0].reshape(-1)
        mac_out_reshape = mac_out[mask != 0].reshape(-1, self.args.n_actions)
        action_loss = th.nn.functional.cross_entropy(mac_out_reshape, actions_all)

        self.action_optimiser.zero_grad()
        action_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.action_params, self.args.grad_norm_clip)
        self.action_optimiser.step()

        running_log["action_loss"].append(action_loss.item())
        running_log["action_grad_norm"].append(grad_norm.item())

        return running_log

    def get_lambda(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # get relavant variables, without mask sum check
        adv_agent_mask = batch["adv_agent_mask"][:, 0, :, 0]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.squeeze(2)

        grad_inputs = []
        mac_out = []
        self.mac.init_hidden_action(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_inputs = self.mac._build_action_inputs(batch, t)
            agent_inputs.requires_grad_(True)
            agent_outs = self.mac.forward_action_input(agent_inputs, batch, t=t)
            mac_out.append(agent_outs)

            agent_coe = th.ones_like(agent_outs)
            agent_outs.backward(agent_coe)
            # the observation of action of adversarial policy
            aap_mask = th.zeros_like(agent_inputs)
            # mask the adv agent observation
            start = 4 + self.args.n_agents * 5
            for z1 in range(adv_agent_mask.shape[0]):
                for z2 in range(adv_agent_mask.shape[1]):
                    if adv_agent_mask[z1, z2]:
                        aap_mask[z1, start+z2*5:start+(z2+1)*5] = 1
            grad_input = agent_inputs * (aap_mask * agent_inputs.grad)
            grad_inputs.append(grad_input)

        grad_out = []
        self.mac.init_hidden_action(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward_action_input(grad_inputs[t], batch, t=t)
            grad_out.append(agent_outs)

        mac_out = th.stack(mac_out, dim=1)  # [batch_size, timestep, n_victim_agents, n_actions]
        grad_out = th.stack(grad_out, dim=1)

        I_t = (mac_out - grad_out).detach()
        I_t[mask == 0] = 0
        I_t = I_t.reshape(batch.batch_size, I_t.shape[1], I_t.shape[2] * I_t.shape[3])
        I_t = th.max(th.abs(I_t), dim=-1)[0]

        return 1 / (1 + I_t)

    def get_adv_loss(self, at_out, lamb, batch: EpisodeBatch, t_env: int, episode_num: int):
        # get relavant variables, without mask sum check
        obs_tplus1 = batch["obs_all"][:, 1:]
        actions_all_tplus1 = batch["actions_all"][:, 1:]
        terminated = batch["terminated"][:, :-1].float()
        adv_agent_mask = batch["adv_agent_mask"][:, 0, :, 0]
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.squeeze(2)

        running_log = {
            "adv_loss_state": [],
            "adv_loss_action": [],
            "adv_loss": [],
        }

        mac_out = []
        action_out = []
        self.mac.init_hidden_state(batch.batch_size)
        self.mac.init_hidden_action(batch.batch_size)
        # the forward of action starts at timestep 1
        self.mac.forward_action(batch, t=0)
        for t in range(batch.max_seq_length - 1):
            agent_inputs = self.mac._build_state_inputs(batch, t)
            # find the place of action of adversarial policy
            adv_action = at_out[:, t] # bav
            adv_action_gumbel = th.nn.functional.gumbel_softmax(adv_action, hard=True, dim=-1).reshape(batch.batch_size, self.args.n_adv_agents, self.n_actions)
            prefix_len = self.mac.input_shape + self.n_actions
            for z1 in range(adv_agent_mask.shape[0]):
                z = 0
                for z2 in range(adv_agent_mask.shape[1]):
                    if adv_agent_mask[z1, z2]:
                        agent_inputs[z1, prefix_len*z2+self.mac.input_shape:prefix_len*(z2+1)] = adv_action_gumbel[z1, z]
                        z += 1
            agent_outs = self.mac.forward_state_input(agent_inputs, batch, t=t)
            mac_out.append(agent_outs)
            action_inputs = self.mac._build_inputs_with_obs(agent_outs, batch, t)
            action_outs = self.mac.forward_action_input(action_inputs, batch, t=t+1)
            action_out.append(action_outs)
        mac_out = th.stack(mac_out, dim=1)  # [batch_size, timestep, n_agents, obs_shape]
        action_out = th.stack(action_out, dim=1)  # [batch_size, timestep, n_victim_agents, n_actions]

        lamb = lamb[mask != 0]

        state_gap = th.abs(mac_out - obs_tplus1)[mask != 0]
        state_gap = state_gap.reshape(state_gap.shape[0], -1)
        state_loss = th.norm(state_gap - self.args.epsilon_state, p=2, dim=-1)

        actions_all_tplus1 = actions_all_tplus1[:, :, self.n_agents:][mask != 0].reshape(-1)
        action_out_reshape = action_out[mask != 0].reshape(-1, self.args.n_actions)
        
        action_loss = th.nn.functional.cross_entropy(action_out_reshape, actions_all_tplus1, reduction="none")
        action_loss = action_loss.reshape(-1, (self.args.n_agents - self.args.n_adv_agents)).mean(dim=1)
        
        adv_loss = ((state_loss - action_loss) * lamb).mean()

        running_log["adv_loss_state"].append(state_loss.mean())
        running_log["adv_loss_action"].append(action_loss.mean())
        running_log["adv_loss"].append(adv_loss)

        return adv_loss, running_log


    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.old_mac.cuda()
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))

    def load_models_without_optim(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
