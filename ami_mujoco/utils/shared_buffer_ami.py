import torch
import numpy as np
from utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    # [401, 2, 1, 377] -> [2, 1, 401, 377]
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


class SharedReplayBuffer_ami(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_agents, num_adv_agents, obs_space, cent_obs_space, act_space):
        self.args = args
        self.action_type = act_space.__class__.__name__
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_proper_time_limits = args.use_proper_time_limits

        self.num_agents = num_agents
        self.num_adv_agents = num_adv_agents

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_adv_agents, *share_obs_shape),
                                  dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_adv_agents, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_adv_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)
        self.rnn_states_critic_oracle = np.zeros_like(self.rnn_states)
        self.rnn_states_maa = np.zeros_like(self.rnn_states)
        self.rnn_states_oracle = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_adv_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.oracle_returns = np.zeros_like(self.value_preds)
        self.oracle_value_preds = np.zeros_like(self.value_preds)

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions_all = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, act_space.n),
                                             dtype=np.float32)
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_adv_agents, act_space.n),
                                             dtype=np.float32)
        else:
            self.available_actions_all = None
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)
        self.act_shape = act_shape

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.oracle_actions = np.ones_like(self.actions)
        if self.action_type == "Discrete":
            self.adv_actions_prob = np.zeros((self.episode_length, self.n_rollout_threads, num_adv_agents, self.args.n_actions), dtype=np.float32)
        elif self.action_type == "Box":
            self.adv_actions_prob = np.zeros((self.episode_length, self.n_rollout_threads, num_adv_agents, self.args.n_actions * 2), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.oracle_action_log_probs = np.ones_like(self.action_log_probs)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_adv_agents, 1), dtype=np.float32)
        self.ami_rewards = np.ones_like(self.rewards)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_adv_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks_all = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.active_masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_adv_agents, 1), dtype=np.float32)

        self.step = 0
        

    def insert(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, maa_pred, maa_log_probs, rnn_states_maa, oracle_actions, oracle_action_log_probs, adv_action_prob,
               rnn_states_oracle, oracle_values, rnn_states_critic_oracle, bad_masks=None, active_masks=None, active_masks_all=None, available_actions_all=None, available_actions=None):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        we only use the rnn hidden state for maa and oracle. others are thrown away.
        """
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.rnn_states_maa[self.step + 1] = rnn_states_maa.copy()
        self.rnn_states_oracle[self.step + 1] = rnn_states_oracle.copy()
        self.rnn_states_critic_oracle[self.step + 1] = rnn_states_critic_oracle.copy()
        self.actions[self.step] = actions.copy()
        self.adv_actions_prob[self.step] = adv_action_prob.copy()
        self.oracle_actions[self.step] = oracle_actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.oracle_action_log_probs[self.step] = oracle_action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.oracle_value_preds[self.step] = oracle_values.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
            self.active_masks_all[self.step + 1] = active_masks_all.copy()
        if available_actions_all is not None:
            self.available_actions_all[self.step + 1] = available_actions_all.copy()
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions_all=None, available_actions=None):
        """
        Insert data into the buffer. This insert function is used specifically for Hanabi, which is turn based.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) denotes indicate whether whether true terminal state or due to episode limit
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions_all[self.step + 1] = available_actions_all.copy()
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length


    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.rnn_states_maa[0] = self.rnn_states_maa[-1].copy()
        self.rnn_states_oracle[0] = self.rnn_states_oracle[-1].copy()
        self.rnn_states_critic_oracle[0] = self.rnn_states_critic_oracle[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        self.active_masks_all[0] = self.active_masks_all[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()
            self.available_actions_all[0] = self.available_actions_all[-1].copy()

    def chooseafter_update(self):
        """Copy last timestep data to first index. This method is used for Hanabi."""
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        if self.action_type == 'Discrete':
            reward_all = self.rewards - self.ami_rewards * self.args.ami_param
        elif self.action_type == 'Box':
            reward_all = self.rewards + self.ami_rewards * self.args.ami_param
        
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(reward_all.shape[0])):
                    if self._use_popart:
                        # step + 1
                        delta = reward_all[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = reward_all[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(reward_all.shape[0])):
                    if self._use_popart:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + reward_all[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(
                            self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + reward_all[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(reward_all.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = reward_all[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = reward_all[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(reward_all.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + reward_all[step]


    def compute_oracle_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        rewards_tplus1 = np.zeros_like(self.rewards)
        rewards_tplus1[:-1] = self.rewards[1:]
        if self._use_proper_time_limits:
            if self._use_gae:
                self.oracle_value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards_tplus1.shape[0])):
                    if self._use_popart:
                        # step + 1
                        delta = rewards_tplus1[step] + self.gamma * value_normalizer.denormalize(
                            self.oracle_value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.oracle_value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        gae = gae * self.bad_masks[step + 1]
                        self.oracle_returns[step] = gae + value_normalizer.denormalize(self.oracle_value_preds[step])
                    else:
                        delta = rewards_tplus1[step] + self.gamma * self.oracle_value_preds[step + 1] * self.masks[step + 1] - \
                                self.oracle_value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.oracle_returns[step] = gae + self.oracle_value_preds[step]
            else:
                self.oracle_returns[-1] = next_value
                for step in reversed(range(rewards_tplus1.shape[0])):
                    if self._use_popart:
                        self.oracle_returns[step] = (self.oracle_returns[step + 1] * self.gamma * self.masks[step + 1] + rewards_tplus1[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(
                            self.oracle_value_preds[step])
                    else:
                        self.oracle_returns[step] = (self.oracle_returns[step + 1] * self.gamma * self.masks[step + 1] + rewards_tplus1[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self.oracle_value_preds[step]
        else:
            if self._use_gae:
                self.oracle_value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards_tplus1.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = rewards_tplus1[step] + self.gamma * value_normalizer.denormalize(
                            self.oracle_value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.oracle_value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.oracle_returns[step] = gae + value_normalizer.denormalize(self.oracle_value_preds[step])
                    else:
                        delta = rewards_tplus1[step] + self.gamma * self.oracle_value_preds[step + 1] * self.masks[step + 1] - \
                                self.oracle_value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.oracle_returns[step] = gae + self.oracle_value_preds[step]
            else:
                self.oracle_returns[-1] = next_value
                for step in reversed(range(rewards_tplus1.shape[0])):
                    self.oracle_returns[step] = self.oracle_returns[step + 1] * self.gamma * self.masks[step + 1] + rewards_tplus1[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        print('not implemented')
        exit(0)

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        """
        Yield training data for non-chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        """
        print('not implemented')
        exit(0)
    
    def recurrent_generator_continuous(self, advantages, oracle_advantages, num_mini_batch, data_chunk_length):
        """
        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param data_chunk_length: (int) length of sequence chunks with which to train RNN.
        """
        # [400, 2, 1]
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        # I add this to make sure that each data chunk only contains one agent and one rollout thread.
        # this was not added in HATRPO version.
        episode_length = episode_length // data_chunk_length * data_chunk_length
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        if len(self.share_obs.shape) > 4:
            print('share_obs shape larger than 4 not considered. need to fix it!')
            exit(-1)
            share_obs = self.share_obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
        else:
            share_obs = _cast(self.share_obs[:episode_length])
            obs = _cast(self.obs[:episode_length])
        # [400, 2, 11, 1]
        # [1000, 4, 2, 3]
        actions_all = self.actions[:episode_length].transpose(1, 0, 2, 3).reshape(-1, self.args.num_agents, self.act_shape)
        # for maa
        actions_all_next = np.concatenate([self.actions[1:episode_length], np.ones_like(self.actions[:1])], axis=0).transpose(1, 0, 2, 3).reshape(-1, self.args.num_agents, self.act_shape)
        actions = actions_all[:, self.args.adv_agent_ids, :].squeeze(1)
        adv_actions_prob = self.adv_actions_prob.transpose(1, 0, 2, 3).reshape(-1, self.args.num_adv_agents, self.act_shape*2).squeeze(1)

        oracle_actions = self.oracle_actions[:episode_length].transpose(1, 0, 2, 3).reshape(-1, self.args.num_agents, self.act_shape)

        action_log_probs_all = self.action_log_probs[:episode_length].transpose(1, 0, 2, 3).reshape(-1, self.args.num_agents, self.act_shape)
        action_log_probs = action_log_probs_all[:, self.args.adv_agent_ids, :]
        oracle_action_log_probs = self.oracle_action_log_probs[:episode_length].transpose(1, 0, 2, 3).reshape(-1, self.args.num_agents, self.act_shape)

        advantages = _cast(advantages[:episode_length])
        oracle_advantages = _cast(oracle_advantages[:episode_length])
        value_preds = _cast(self.value_preds[:episode_length])
        oracle_value_preds = _cast(self.oracle_value_preds[:episode_length])
        returns = _cast(self.returns[:episode_length])
        oracle_returns = _cast(self.oracle_returns[:episode_length])
        masks = _cast(self.masks[:episode_length])
        masks_maa = _cast(np.concatenate([self.masks[:episode_length-1], np.zeros_like(self.masks[:1])], axis=0))

        active_masks = _cast(self.active_masks[:episode_length])
        active_masks_all = self.active_masks_all[:episode_length].transpose(1, 0, 3, 2).reshape(-1, self.args.num_agents)
        active_masks_all_next = np.concatenate([self.active_masks_all[1:episode_length], np.zeros_like(self.active_masks_all[:1])], axis=0).transpose(1, 0, 3, 2).reshape(-1, self.args.num_agents)

        rnn_states = self.rnn_states[:episode_length].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:episode_length].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_critic.shape[3:])
        rnn_states_maa = self.rnn_states_maa[:episode_length].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_maa.shape[3:])
        rnn_states_oracle = self.rnn_states_oracle[:episode_length].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_oracle.shape[3:])
        rnn_states_critic_oracle = self.rnn_states_critic_oracle[:episode_length].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_critic_oracle.shape[3:])

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:episode_length])
            available_actions_all = self.available_actions_all[:episode_length].transpose(1, 0, 2, 3).reshape(-1, self.args.num_agents, self.args.n_actions)
            available_actions_all_next = np.concatenate([self.available_actions_all[1:episode_length], np.ones_like(self.available_actions_all[:1])], axis=0).transpose(1, 0, 2, 3).reshape(-1, self.args.num_agents, self.args.n_actions)
        
        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            rnn_states_maa_batch = []
            rnn_states_oracle_batch = []
            rnn_states_critic_oracle_batch = []
            actions_batch = []
            oracle_actions_batch = []
            actions_all_batch = []
            actions_all_next_batch = []
            adv_actions_prob_batch = []
            available_actions_batch = []
            available_actions_all_batch = []
            available_actions_all_next_batch = []
            value_preds_batch = []
            value_preds_oracle_batch = []
            return_batch = []
            oracle_return_batch = []
            masks_batch = []
            masks_maa_batch = []
            active_masks_batch = []
            active_masks_all_batch = []
            active_masks_all_next_batch = []
            old_action_log_probs_batch = []
            oracle_action_log_probs_batch = []
            old_action_log_probs_all_batch = []
            adv_targ = []
            oracle_adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                oracle_actions_batch.append(oracle_actions[ind:ind + data_chunk_length])
                actions_all_batch.append(actions_all[ind:ind + data_chunk_length])
                adv_actions_prob_batch.append(adv_actions_prob[ind:ind + data_chunk_length])
                actions_all_next_batch.append(actions_all_next[ind:ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind + data_chunk_length])
                    available_actions_all_batch.append(available_actions_all[ind:ind + data_chunk_length])
                    available_actions_all_next_batch.append(available_actions_all_next[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                value_preds_oracle_batch.append(oracle_value_preds[ind:ind + data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                oracle_return_batch.append(oracle_returns[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                masks_maa_batch.append(masks_maa[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                active_masks_all_batch.append(active_masks_all[ind:ind + data_chunk_length])
                active_masks_all_next_batch.append(active_masks_all_next[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind + data_chunk_length])
                oracle_action_log_probs_batch.append(oracle_action_log_probs[ind:ind + data_chunk_length])
                old_action_log_probs_all_batch.append(action_log_probs_all[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])
                oracle_adv_targ.append(oracle_advantages[ind:ind + data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])
                rnn_states_maa_batch.append(rnn_states_maa[ind])
                rnn_states_oracle_batch.append(rnn_states_oracle[ind])
                rnn_states_critic_oracle_batch.append(rnn_states_critic_oracle[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)           
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)
            
            actions_batch = np.stack(actions_batch, axis=1)
            oracle_actions_batch = np.stack(oracle_actions_batch, axis=1)
            actions_all_batch = np.stack(actions_all_batch, axis=1)
            adv_actions_prob_batch = np.stack(adv_actions_prob_batch, axis=1)
            actions_all_next_batch = np.stack(actions_all_next_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
                available_actions_all_batch = np.stack(available_actions_all_batch, axis=1)
                available_actions_all_next_batch = np.stack(available_actions_all_next_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            value_preds_oracle_batch = np.stack(value_preds_oracle_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            oracle_return_batch = np.stack(oracle_return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            masks_maa_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            active_masks_all_batch = np.stack(active_masks_all_batch, axis=1)
            active_masks_all_next_batch = np.stack(active_masks_all_next_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            oracle_action_log_probs_batch = np.stack(oracle_action_log_probs_batch, axis=1)
            old_action_log_probs_all_batch = np.stack(old_action_log_probs_all_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)
            oracle_adv_targ = np.stack(oracle_adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])
            rnn_states_maa_batch = np.stack(rnn_states_maa_batch).reshape(N, *self.rnn_states_maa.shape[3:])
            rnn_states_oracle_batch = np.stack(rnn_states_oracle_batch).reshape(N, *self.rnn_states_oracle.shape[3:])
            rnn_states_critic_oracle_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic_oracle.shape[3:])

            
            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            oracle_actions_batch = _flatten(L, N, oracle_actions_batch)
            actions_all_next_batch = _flatten(L, N, actions_all_next_batch)
            actions_all_batch = _flatten(L, N, actions_all_batch)
            adv_actions_prob_batch = _flatten(L, N, adv_actions_prob_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
                available_actions_all_batch = _flatten(L, N, available_actions_all_batch)
                available_actions_all_next_batch = _flatten(L, N, available_actions_all_next_batch)
            else:
                available_actions_batch = None
                available_actions_all_batch = None
                available_actions_all_next_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            value_preds_oracle_batch = _flatten(L, N, value_preds_oracle_batch)
            return_batch = _flatten(L, N, return_batch)
            oracle_return_batch = _flatten(L, N, oracle_return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            masks_maa_batch = _flatten(L, N, masks_maa_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            active_masks_all_batch = _flatten(L, N, active_masks_all_batch)
            active_masks_all_next_batch = _flatten(L, N, active_masks_all_next_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            old_action_log_probs_all_batch = _flatten(L, N, old_action_log_probs_all_batch)
            oracle_action_log_probs_batch = _flatten(L, N, oracle_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)
            oracle_adv_targ = _flatten(L, N, oracle_adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, rnn_states_maa_batch, rnn_states_oracle_batch, actions_batch, oracle_actions_batch, actions_all_batch, actions_all_next_batch,\
                  value_preds_batch, return_batch, masks_batch, masks_maa_batch, active_masks_all_batch, active_masks_batch, old_action_log_probs_batch.squeeze(1), oracle_action_log_probs_batch.squeeze(), old_action_log_probs_all_batch,\
                  adv_targ, available_actions_batch, available_actions_all_batch, available_actions_all_next_batch, rnn_states_critic_oracle_batch, value_preds_oracle_batch, oracle_adv_targ, adv_actions_prob_batch, oracle_return_batch, active_masks_all_next_batch
    
    def recurrent_generator(self, advantages, oracle_advantages, num_mini_batch, data_chunk_length):
        """
        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param data_chunk_length: (int) length of sequence chunks with which to train RNN.
        """
        # [400, 2, 1]
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        # I add this to make sure that each data chunk only contains one agent and one rollout thread.
        # this was not added in HATRPO version.
        episode_length = episode_length // data_chunk_length * data_chunk_length
        
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        if len(self.share_obs.shape) > 4:
            print('share_obs shape larger than 4 not considered. need to fix it!')
            exit(-1)
            share_obs = self.share_obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
        else:
            share_obs = _cast(self.share_obs[:episode_length])
            obs = _cast(self.obs[:episode_length])
        # [400, 2, 11, 1]
        # [1000, 4, 2, 3]
        actions_all = self.actions[:episode_length].transpose(1, 0, 3, 2).reshape(-1, self.args.num_agents)
        # for maa
        actions_all_next = np.concatenate([self.actions[1:episode_length], np.ones_like(self.actions[:1])], axis=0).transpose(1, 0, 3, 2).reshape(-1, self.args.num_agents)
        actions = actions_all[:, self.args.adv_agent_ids]
        adv_actions_prob = self.adv_actions_prob.transpose(1, 0, 3, 2).reshape(-1, self.adv_actions_prob.shape[-1])

        oracle_actions = self.oracle_actions[:episode_length].transpose(1, 0, 3, 2).reshape(-1, self.args.num_agents)

        action_log_probs_all = self.action_log_probs[:episode_length].transpose(1, 0, 3, 2).reshape(-1, self.args.num_agents)
        action_log_probs = action_log_probs_all[:, self.args.adv_agent_ids]
        action_log_probs = action_log_probs_all[:, self.args.adv_agent_ids]
        oracle_action_log_probs = self.oracle_action_log_probs[:episode_length].transpose(1, 0, 3, 2).reshape(-1, self.args.num_agents)

        advantages = _cast(advantages[:episode_length])
        oracle_advantages = _cast(oracle_advantages[:episode_length])
        value_preds = _cast(self.value_preds[:episode_length])
        oracle_value_preds = _cast(self.oracle_value_preds[:episode_length])
        returns = _cast(self.returns[:episode_length])
        oracle_returns = _cast(self.oracle_returns[:episode_length])
        masks = _cast(self.masks[:episode_length])
        masks_maa = _cast(np.concatenate([self.masks[:episode_length-1], np.zeros_like(self.masks[:1])], axis=0))
        
        active_masks = _cast(self.active_masks[:episode_length])
        active_masks_all = self.active_masks_all[:episode_length].transpose(1, 0, 3, 2).reshape(-1, self.args.num_agents)
        active_masks_all_next = np.concatenate([self.active_masks_all[1:episode_length], np.zeros_like(self.active_masks_all[:1])], axis=0).transpose(1, 0, 3, 2).reshape(-1, self.args.num_agents)
        
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = self.rnn_states[:episode_length].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:episode_length].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_critic.shape[3:])
        rnn_states_maa = self.rnn_states_maa[:episode_length].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_maa.shape[3:])
        rnn_states_oracle = self.rnn_states_oracle[:episode_length].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_oracle.shape[3:])
        rnn_states_critic_oracle = self.rnn_states_critic_oracle[:episode_length].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_critic_oracle.shape[3:])

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:episode_length])
            available_actions_all = self.available_actions_all[:episode_length].transpose(1, 0, 2, 3).reshape(-1, self.args.num_agents, self.args.n_actions)
            available_actions_all_next = np.concatenate([self.available_actions_all[1:episode_length], np.ones_like(self.available_actions_all[:1])], axis=0).transpose(1, 0, 2, 3).reshape(-1, self.args.num_agents, self.args.n_actions)
        
        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            rnn_states_maa_batch = []
            rnn_states_oracle_batch = []
            rnn_states_critic_oracle_batch = []
            actions_batch = []
            oracle_actions_batch = []
            actions_all_batch = []
            actions_all_next_batch = []
            adv_actions_prob_batch = []
            available_actions_batch = []
            available_actions_all_batch = []
            available_actions_all_next_batch = []
            value_preds_batch = []
            value_preds_oracle_batch = []
            return_batch = []
            oracle_return_batch = []
            masks_batch = []
            masks_maa_batch = []
            active_masks_batch = []
            active_masks_all_batch = []
            active_masks_all_next_batch = []
            old_action_log_probs_batch = []
            oracle_action_log_probs_batch = []
            old_action_log_probs_all_batch = []
            adv_targ = []
            oracle_adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                oracle_actions_batch.append(oracle_actions[ind:ind + data_chunk_length])
                actions_all_batch.append(actions_all[ind:ind + data_chunk_length])
                adv_actions_prob_batch.append(adv_actions_prob[ind:ind + data_chunk_length])
                actions_all_next_batch.append(actions_all_next[ind:ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind + data_chunk_length])
                    available_actions_all_batch.append(available_actions_all[ind:ind + data_chunk_length])
                    available_actions_all_next_batch.append(available_actions_all_next[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                value_preds_oracle_batch.append(oracle_value_preds[ind:ind + data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                oracle_return_batch.append(oracle_returns[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                masks_maa_batch.append(masks_maa[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                active_masks_all_batch.append(active_masks_all[ind:ind + data_chunk_length])
                active_masks_all_next_batch.append(active_masks_all_next[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind + data_chunk_length])
                oracle_action_log_probs_batch.append(oracle_action_log_probs[ind:ind + data_chunk_length])
                old_action_log_probs_all_batch.append(action_log_probs_all[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])
                oracle_adv_targ.append(oracle_advantages[ind:ind + data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])
                rnn_states_maa_batch.append(rnn_states_maa[ind])
                rnn_states_oracle_batch.append(rnn_states_oracle[ind])
                rnn_states_critic_oracle_batch.append(rnn_states_critic_oracle[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)           
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)
            
            actions_batch = np.stack(actions_batch, axis=1)
            oracle_actions_batch = np.stack(oracle_actions_batch, axis=1)
            actions_all_batch = np.stack(actions_all_batch, axis=1)
            adv_actions_prob_batch = np.stack(adv_actions_prob_batch, axis=1)
            actions_all_next_batch = np.stack(actions_all_next_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
                available_actions_all_batch = np.stack(available_actions_all_batch, axis=1)
                available_actions_all_next_batch = np.stack(available_actions_all_next_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            value_preds_oracle_batch = np.stack(value_preds_oracle_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            oracle_return_batch = np.stack(oracle_return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            masks_maa_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            active_masks_all_batch = np.stack(active_masks_all_batch, axis=1)
            active_masks_all_next_batch = np.stack(active_masks_all_next_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            oracle_action_log_probs_batch = np.stack(oracle_action_log_probs_batch, axis=1)
            old_action_log_probs_all_batch = np.stack(old_action_log_probs_all_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)
            oracle_adv_targ = np.stack(oracle_adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])
            rnn_states_maa_batch = np.stack(rnn_states_maa_batch).reshape(N, *self.rnn_states_maa.shape[3:])
            rnn_states_oracle_batch = np.stack(rnn_states_oracle_batch).reshape(N, *self.rnn_states_oracle.shape[3:])
            rnn_states_critic_oracle_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic_oracle.shape[3:])

            
            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            oracle_actions_batch = _flatten(L, N, oracle_actions_batch)
            actions_all_next_batch = _flatten(L, N, actions_all_next_batch)
            actions_all_batch = _flatten(L, N, actions_all_batch)
            adv_actions_prob_batch = _flatten(L, N, adv_actions_prob_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
                available_actions_all_batch = _flatten(L, N, available_actions_all_batch)
                available_actions_all_next_batch = _flatten(L, N, available_actions_all_next_batch)
            else:
                available_actions_batch = None
                available_actions_all_batch = None
                available_actions_all_next_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            value_preds_oracle_batch = _flatten(L, N, value_preds_oracle_batch)
            return_batch = _flatten(L, N, return_batch)
            oracle_return_batch = _flatten(L, N, oracle_return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            masks_maa_batch = _flatten(L, N, masks_maa_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            active_masks_all_batch = _flatten(L, N, active_masks_all_batch)
            active_masks_all_next_batch = _flatten(L, N, active_masks_all_next_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            old_action_log_probs_all_batch = _flatten(L, N, old_action_log_probs_all_batch)
            oracle_action_log_probs_batch = _flatten(L, N, oracle_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)
            oracle_adv_targ = _flatten(L, N, oracle_adv_targ)
            
            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, rnn_states_maa_batch, rnn_states_oracle_batch, actions_batch, oracle_actions_batch, actions_all_batch, actions_all_next_batch,\
                  value_preds_batch, return_batch, masks_batch, masks_maa_batch, active_masks_all_batch, active_masks_batch, old_action_log_probs_batch, oracle_action_log_probs_batch, old_action_log_probs_all_batch,\
                  adv_targ, available_actions_batch, available_actions_all_batch, available_actions_all_next_batch, rnn_states_critic_oracle_batch, value_preds_oracle_batch, oracle_adv_targ, adv_actions_prob_batch, oracle_return_batch, active_masks_all_next_batch

