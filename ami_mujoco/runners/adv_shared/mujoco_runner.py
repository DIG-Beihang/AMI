import time
import numpy as np
from functools import reduce
import torch
from runners.adv_shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class MujocoRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for Mujoco. See parent class for details."""

    def __init__(self, config):
        super(MujocoRunner, self).__init__(config)

    def run(self):
        self.warmup()

        if self.all_args.evaluate:
            self.eval(0)
            return

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            done_episodes_rewards = []

            for step in range(self.episode_length):
                # Sample actions
                values_victim, actions_victim, action_log_probs_victim, \
                    rnn_states_victim, rnn_states_critic_victim = self.collect_victim(step)
                if self.all_args.adv_algorithm_name == "mappo_icml":
                    values, values_icml, actions, action_log_probs, rnn_states, rnn_states_critic, \
                        rnn_states_critic_icml = self.collect_icml(step)
                else:
                    values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                actions_use = actions_victim.copy()
                actions_use[:, self.adv_agent_ids] = actions
                action_log_probs_use = action_log_probs_victim.copy()
                action_log_probs_use[:, self.adv_agent_ids] = action_log_probs

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions_use)

                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).flatten()
                train_episode_rewards += reward_env
                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0

                if self.config['all_args'].adv_algorithm_name == 'mappo_ami':
                    maa_pred, maa_log_probs, rnn_states_maa, oracle_actions, oracle_action_log_probs, \
                        rnn_states_oracle, oracle_values, rnn_oracle_states_critic, adv_action_prob = self.collect_ami(step, actions_use, available_actions)
                elif self.all_args.adv_algorithm_name == 'mappo_usenix':
                    rnn_states_state, rnn_states_action = self.collect_usenix(step, actions_victim)

                data_victim = obs, share_obs, rewards, dones, infos, \
                       values_victim, actions_victim, action_log_probs_victim, \
                       rnn_states_victim, rnn_states_critic_victim 

                if self.all_args.adv_algorithm_name == "mappo_iclr" or \
                    self.all_args.adv_algorithm_name == "mappo_fgsm":
                    data = obs, share_obs, rewards, dones, infos, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic
                    self.insert(data)
                elif self.all_args.adv_algorithm_name == "mappo_usenix":
                    data = obs, share_obs, rewards, dones, infos, available_actions, values, actions, actions_victim, \
                        action_log_probs, rnn_states, rnn_states_critic, rnn_states_state, rnn_states_action
                    self.insert_usenix(data)
                elif self.all_args.adv_algorithm_name == "mappo_icml":
                    data = obs, share_obs, rewards, dones, infos, available_actions, values, values_icml, \
                        actions, action_log_probs, rnn_states, rnn_states_critic, rnn_states_critic_icml  
                    self.insert_icml(data)
                elif self.all_args.adv_algorithm_name == 'mappo_ami':
                    data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions_use, action_log_probs_use, \
                       rnn_states, rnn_states_critic, maa_pred, maa_log_probs, rnn_states_maa,\
                     oracle_actions, oracle_action_log_probs, adv_action_prob, rnn_states_oracle, oracle_values, rnn_oracle_states_critic
                    self.insert_ami(data)

                # insert data into buffer
                self.insert_victim(data_victim)

            # compute influence reward
            if self.all_args.adv_algorithm_name == 'mappo_ami':
                ami_rewards = self.trainer.get_target_influence_reward_continuous(self.buffer)
                self.buffer.ami_rewards = ami_rewards.detach().cpu().numpy()

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(total_num_steps)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                self.log_train(train_infos, total_num_steps)

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    print("some episodes done, average rewards: ", aver_episode_rewards)
                    self.writter.add_scalar("train/aver_episode_rewards", aver_episode_rewards,
                                             total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.victim_buffer.share_obs[0] = share_obs.copy()
        self.victim_buffer.obs[0] = obs.copy()

        self.buffer.share_obs[0] = share_obs[:, self.adv_agent_ids].copy()
        self.buffer.obs[0] = obs[:, self.adv_agent_ids].copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    @torch.no_grad()
    def collect_victim(self, step):
        self.victim_policy.actor.eval()
        self.victim_policy.critic.eval()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.victim_policy.get_actions(np.concatenate(self.victim_buffer.share_obs[step]),
                                            np.concatenate(self.victim_buffer.obs[step]),
                                            np.concatenate(self.victim_buffer.rnn_states[step]),
                                            np.concatenate(self.victim_buffer.rnn_states_critic[step]),
                                            np.concatenate(self.victim_buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        return values, actions, action_log_probs, rnn_states, rnn_states_critic
    
    @torch.no_grad()
    def collect_icml(self, step):
        self.trainer.prep_rollout()
        value, value_icml, action, action_log_prob, rnn_state, rnn_state_critic, rnn_state_critic_icml \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                        np.concatenate(self.buffer.obs[step]),
                                        np.concatenate(self.buffer.rnn_states[step]),
                                        np.concatenate(self.buffer.rnn_states_critic[step]),
                                        np.concatenate(self.buffer.rnn_states_critic_icml[step]),
                                        np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        values_icml = np.array(np.split(_t2n(value_icml), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        rnn_states_critic_icml = np.array(np.split(_t2n(rnn_state_critic_icml), self.n_rollout_threads))
        return values, values_icml, actions, action_log_probs, rnn_states, rnn_states_critic, rnn_states_critic_icml
    
    @torch.no_grad()
    def collect_usenix(self, step, actions_all):
        # forward state and transition model here to collect its rnn_states. since there's no grad, we do not use the predictions and log probs for optimization
        self.trainer.prep_rollout()

        _, rnn_states_state \
            = self.trainer.policy.state_transition(np.concatenate(self.buffer.share_obs[step]),
                                            actions_all,
                                            np.concatenate(self.buffer.rnn_states_state[step]),
                                            np.concatenate(self.buffer.masks[step]))

        _, rnn_states_action \
            = self.trainer.policy.action_transition(np.concatenate(self.buffer.obs_all[step]),
                                            np.concatenate(self.buffer.rnn_states_action[step]),
                                            np.concatenate(self.buffer.masks_all[step]))

        rnn_states_state = np.array(np.split(_t2n(rnn_states_state), self.n_rollout_threads))
        rnn_states_action = np.array(np.split(_t2n(rnn_states_action), self.n_rollout_threads))

        return rnn_states_state, rnn_states_action

    @torch.no_grad()
    def collect_ami(self, step, action_use, available_actions=None):
        # forward maa and oracle here to collect its rnn_states. since there's no grad, we do not use the predictions and log probs for optimization
        self.trainer.prep_rollout()
        adv_action_prob = self.trainer.policy.actor.get_logit(
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.masks[step]))

        maa_pred, maa_log_probs, rnn_states_maa \
            = self.trainer.policy.get_maa_results(np.concatenate(self.buffer.share_obs[step]),
                                            action_use,
                                            np.concatenate(self.buffer.rnn_states_maa[step]),
                                            np.concatenate(self.buffer.masks[step]))
        
        oracle_actions, oracle_action_log_probs, rnn_states_oracle \
            = self.trainer.policy.get_oracle_actions(np.concatenate(self.buffer.share_obs[step]),
                                            action_use,
                                            np.concatenate(self.buffer.rnn_states_oracle[step]),
                                            np.concatenate(self.buffer.masks[step]))
        
        oracle_values, rnn_oracle_states_critic = self.trainer.policy.get_oracle_critic(np.concatenate(self.buffer.share_obs[step]),
                                                        action_use,
                                                        np.concatenate(self.buffer.rnn_states_critic_oracle[step]),
                                                        np.concatenate(self.buffer.masks[step]))

        maa_pred = np.array(np.split(_t2n(maa_pred), self.n_rollout_threads))
        maa_log_probs = np.array(np.split(_t2n(maa_log_probs), self.n_rollout_threads))
        rnn_states_maa = np.array(np.split(_t2n(rnn_states_maa), self.n_rollout_threads))
        oracle_actions = np.array(np.split(_t2n(oracle_actions), self.n_rollout_threads))
        oracle_action_log_probs = np.array(np.split(_t2n(oracle_action_log_probs), self.n_rollout_threads))
        rnn_states_oracle = np.array(np.split(_t2n(rnn_states_oracle), self.n_rollout_threads))
        oracle_values = np.array(np.split(_t2n(oracle_values), self.n_rollout_threads))
        rnn_oracle_states_critic = np.array(np.split(_t2n(rnn_oracle_states_critic), self.n_rollout_threads))

        adv_action_prob = np.array(np.split(_t2n(adv_action_prob), self.n_rollout_threads))
        return maa_pred, maa_log_probs, rnn_states_maa, oracle_actions, oracle_action_log_probs, rnn_states_oracle, oracle_values, rnn_oracle_states_critic, adv_action_prob
    
    def insert(self, data):
        obs, share_obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)
        dones = dones[:, self.adv_agent_ids]

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_adv_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_adv_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_adv_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_adv_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_adv_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_adv_agents, 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs

        adv_rewards = self.build_adv_rewards(rewards, infos)

        self.buffer.insert(share_obs[:, self.adv_agent_ids], obs[:, self.adv_agent_ids], rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, adv_rewards, masks, None, active_masks, None)

    def insert_victim(self, data):
        obs, share_obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs

        self.victim_buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, None, active_masks, None)

    def insert_usenix(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, values, actions, actions_victim, \
        action_log_probs, rnn_states, rnn_states_critic, rnn_states_state, rnn_states_action = data

        dones_env = np.all(dones, axis=1)
        dones = dones[:, self.adv_agent_ids]

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_adv_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_adv_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        rnn_states_state[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_adv_agents, *self.buffer.rnn_states_state.shape[3:]), dtype=np.float32)
        rnn_states_action[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_adv_agents, *self.buffer.rnn_states_action.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_adv_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_adv_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_adv_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_adv_agents, 1), dtype=np.float32)
        
        if not self.use_centralized_V:
            share_obs = obs

        adv_rewards = self.build_adv_rewards(rewards, infos)

        self.buffer.insert({
            "current_adv": self.adv_agent_ids,
            "share_obs": share_obs[:, self.adv_agent_ids],
            "obs": obs[:, self.adv_agent_ids],
            "obs_all": obs,
            "rnn_states": rnn_states,
            "rnn_states_critic": rnn_states_critic,
            "rnn_states_state": rnn_states_state,
            "rnn_states_action": rnn_states_action,
            "value_preds": values,
            "rewards": adv_rewards,
            "actions": actions,
            "actions_all": actions_victim,
            "action_log_probs": action_log_probs,
            "masks": masks,
            "masks_all": masks.repeat(self.num_agents, 1),
            "active_masks": active_masks,
        })

    def insert_icml(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, values, values_icml, actions, action_log_probs, \
            rnn_states, rnn_states_critic, rnn_states_critic_icml = data

        dones_env = np.all(dones, axis=1)
        dones = dones[:, self.adv_agent_ids]

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_adv_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_adv_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        rnn_states_critic_icml[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_adv_agents, *self.buffer.rnn_states_critic_icml.shape[3:]), dtype=np.float32)
    
        masks = np.ones((self.n_rollout_threads, self.num_adv_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_adv_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_adv_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_adv_agents, 1), dtype=np.float32)
        
        if not self.use_centralized_V:
            share_obs = obs

        adv_rewards = self.build_adv_rewards(rewards, infos)
        adv_rewards_positive = rewards[:, self.adv_agent_ids]

        self.buffer.insert(share_obs[:, self.adv_agent_ids], obs[:, self.adv_agent_ids], rnn_states, rnn_states_critic, rnn_states_critic_icml,
                           actions, action_log_probs, values, values_icml, adv_rewards, adv_rewards_positive, masks, None, active_masks, None)

    def insert_ami(self, data):
        # insert all obs and 
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, maa_pred, maa_log_probs, rnn_states_maa,\
                     oracle_actions, oracle_action_log_probs, adv_action_prob, rnn_states_oracle, oracle_values, rnn_oracle_states_critic = data
        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_adv_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_adv_agents, *self.victim_buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_adv_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_adv_agents, 1), dtype=np.float32)

        active_masks_all = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks_all[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks_all[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        dones = dones[:, self.adv_agent_ids]

        active_masks = np.ones((self.n_rollout_threads, self.num_adv_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_adv_agents, 1), dtype=np.float32)

        adv_rewards = self.build_adv_rewards(rewards, infos)
        
        if not self.use_centralized_V:
            share_obs = obs
        # (8, 11, 377), (8, 11, 312), (8, 1, 1, 64), (8, 1, 1, 64), (8, 11, 1), (8, 11, 1), (8, 1, 1), (8, 1, 1), (8, 11, 1), (8, 1, 1), (8, 1), (8, 11, 17)
        
        self.buffer.insert(share_obs[:, self.adv_agent_ids], obs[:, self.adv_agent_ids], rnn_states, rnn_states_critic, actions, action_log_probs, 
                           values, adv_rewards, masks, maa_pred, maa_log_probs, rnn_states_maa, oracle_actions, oracle_action_log_probs, adv_action_prob, rnn_states_oracle, oracle_values, rnn_oracle_states_critic,
                           None, active_masks, active_masks_all, None, None)

    def build_adv_rewards(self, rewards, infos):
        adv_rewards = np.zeros_like(rewards)
        for t1 in range(self.n_rollout_threads):
            for t2 in range(self.num_agents):
                if self.all_args.scenario == "HalfCheetah-v2":
                    adv_rewards[t1, t2, 0] = - infos[t1][t2]["reward_run"] + self.all_args.reward_ctrl_param * infos[t1][t2]["reward_ctrl"]
                elif self.all_args.scenario == "Humanoid-v2":
                    adv_rewards[t1, t2, 0] = - (infos[t1][t2]["reward_alive"] + infos[t1][t2]["reward_linvel"]) + self.all_args.reward_ctrl_param * (infos[t1][t2]["reward_quadctrl"] + infos[t1][t2]["reward_impact"])
                elif self.all_args.scenario == "Ant-v2":
                    adv_rewards[t1, t2, 0] = - (infos[t1][t2]["reward_survive"] + infos[t1][t2]["reward_forward"]) + self.all_args.reward_ctrl_param * (infos[t1][t2]["reward_ctrl"] + infos[t1][t2]["reward_contact"])
                elif self.all_args.scenario == "HumanoidStandup-v2":
                    adv_rewards[t1, t2, 0] = - infos[t1][t2]["reward_linup"] + self.all_args.reward_ctrl_param * (infos[t1][t2]["reward_quadctrl"] + infos[t1][t2]["reward_impact"])
                elif self.all_args.scenario == "manyagent_swimmer":
                    adv_rewards[t1, t2, 0] = - infos[t1][t2]["reward_fwd"] + self.all_args.reward_ctrl_param * infos[t1][t2]["reward_ctrl"]
                else: # Walker2d, Hopper
                    adv_rewards[t1, t2, 0] = - rewards[t1, t2, 0]
        adv_rewards = adv_rewards[:, self.adv_agent_ids]

        return adv_rewards

    def log_train(self, train_infos, total_num_steps):
        print("average_step_rewards is {}.".format(np.mean(self.buffer.rewards)))
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        if self.all_args.adv_algorithm_name == 'mappo_ami':
            train_infos["average_step_ami_rewards"] = np.mean(self.buffer.ami_rewards)
        for k, v in train_infos.items():
            self.writter.add_scalar("train/" + k, v, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []
        eval_episode_reward_run = []
        eval_episode_reward_ctrl = []
        reward_run = 0
        reward_ctrl = 0
        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])
            eval_episode_reward_run.append(0)
            eval_episode_reward_ctrl.append(0)

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()

        eval_rnn_states_victim = np.zeros((self.n_eval_rollout_threads, self.num_adv_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks_victim = np.ones((self.n_eval_rollout_threads, self.num_adv_agents, 1), dtype=np.float32)

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.victim_policy.actor.eval()
            self.victim_policy.critic.eval()
            if self.all_args.adv_algorithm_name == "mappo_fgsm":
                obs = self.attack.forward(np.concatenate(eval_obs[:, self.adv_agent_ids]),
                                    np.concatenate(eval_rnn_states),
                                    np.concatenate(eval_rnn_states_victim[:, self.adv_agent_ids]),
                                    np.concatenate(eval_masks))
                eval_obs[:, self.adv_agent_ids] = obs
                
            eval_actions_victim, eval_rnn_states_victim = \
                self.victim_policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        deterministic=True)
            eval_actions_victim = np.array(np.split(_t2n(eval_actions_victim), self.n_eval_rollout_threads))
            eval_rnn_states_victim = np.array(np.split(_t2n(eval_rnn_states_victim), self.n_eval_rollout_threads))

            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_obs[:, self.adv_agent_ids]),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            eval_actions_use = eval_actions_victim.copy()
            if not self.all_args.adv_algorithm_name == "mappo_fgsm":
                eval_actions_use[:, self.adv_agent_ids] = eval_actions

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, _ = self.eval_envs.step(eval_actions_use)
            
            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])
                if self.all_args.scenario == "HalfCheetah-v2":
                    eval_episode_reward_run[eval_i] += eval_infos[0][eval_i]["reward_run"]
                    eval_episode_reward_ctrl[eval_i] += eval_infos[0][eval_i]["reward_ctrl"]
                elif self.all_args.scenario == "Humanoid-v2":
                    eval_episode_reward_run[eval_i] += (eval_infos[0][eval_i]["reward_alive"] + eval_infos[0][eval_i]["reward_linvel"])
                    eval_episode_reward_ctrl[eval_i] += (eval_infos[0][eval_i]["reward_quadctrl"] + eval_infos[0][eval_i]["reward_impact"])
                elif self.all_args.scenario == "Ant-v2":
                    eval_episode_reward_run[eval_i] += (eval_infos[0][eval_i]["reward_survive"] + eval_infos[0][eval_i]["reward_forward"])
                    eval_episode_reward_ctrl[eval_i] += (eval_infos[0][eval_i]["reward_ctrl"] + eval_infos[0][eval_i]["reward_contact"])
                elif self.all_args.scenario == "HumanoidStandup-v2":
                    eval_episode_reward_run[eval_i] += eval_infos[0][eval_i]["reward_linup"]
                    eval_episode_reward_ctrl[eval_i] += (eval_infos[0][eval_i]["reward_quadctrl"] + eval_infos[0][eval_i]["reward_impact"])
                elif self.all_args.scenario == "manyagent_swimmer":
                    eval_episode_reward_run[eval_i] += eval_infos[0][eval_i]["reward_fwd"]
                    eval_episode_reward_ctrl[eval_i] += eval_infos[0][eval_i]["reward_ctrl"]
                else: # Walker2d, Hopper
                    eval_episode_reward_run[eval_i] += eval_rewards[eval_i]
                    eval_episode_reward_ctrl[eval_i] += 0

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states_victim[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks_victim = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks_victim[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                          dtype=np.float32)

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_adv_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_adv_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_adv_agents, 1),
                                                          dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(np.sum(one_episode_rewards[eval_i], axis=0))
                    one_episode_rewards[eval_i] = []
                    reward_run += eval_episode_reward_run[eval_i]
                    eval_episode_reward_run[eval_i] = 0
                    reward_ctrl += eval_episode_reward_ctrl[eval_i]
                    eval_episode_reward_ctrl[eval_i] = 0

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.concatenate(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards,
                                  'eval_max_episode_rewards': [np.max(eval_episode_rewards)],
                                  'eval_average_episode_reward_run': reward_run / eval_episode,
                                  'eval_average_episode_reward_ctrl': reward_ctrl / eval_episode}
                self.log_env(eval_env_infos, total_num_steps)
                print(f"Evaluate for {eval_episode} episodes in t_env {total_num_steps}, results:")
                for k in eval_env_infos:
                    print(f"{k}: {np.mean(eval_env_infos[k])}")
                break
