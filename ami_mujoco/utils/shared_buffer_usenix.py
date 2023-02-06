import torch
import numpy as np
from utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


class SharedReplayBufferUSENIX(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_victim_agents, obs_space, cent_obs_space, act_space, scheme=None):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_proper_time_limits = args.use_proper_time_limits

        self.num_victim_agents = num_victim_agents
        self.num_adv_agents = len(args.adv_agent_ids)

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        act_shape = get_shape_from_act_space(act_space)

        self.step = 0
        self.available_actions = None
        self.available_actions_all = None
        if scheme is not None:
            self.scheme = scheme
        else:
            self.scheme = [
                {"type": np.float32, "name": "current_adv", "shape": (1, ), "init_value": 0, "offset": 0},
                {"type": np.float32, "name": "share_obs", "shape": share_obs_shape, "init_value": 0, "offset": 1},
                {"type": np.float32, "name": "obs", "shape": obs_shape, "init_value": 0, "offset": 1, "actions":["t+1"]},
                {"type": np.float32, "name": "obs_all", "shape": (num_victim_agents, *obs_shape), "init_value": 0, "offset": 1, "actions": ["victim", "t+1"]},
                {"type": np.float32, "name": "rnn_states", "shape": (self.recurrent_N, self.hidden_size), "init_value": 0, "offset": 1, "actions": ["state"]},
                {"type": np.float32, "name": "rnn_states_critic", "shape": (self.recurrent_N, self.hidden_size), "init_value": 0, "offset": 1, "actions": ["state"]},
                {"type": np.float32, "name": "rnn_states_state", "shape": (self.recurrent_N, self.hidden_size), "init_value": 0, "offset": 1, "actions": ["state"]},
                {"type": np.float32, "name": "rnn_states_action", "shape": (self.recurrent_N, self.hidden_size), "offset": 1, "init_value": 0, "actions": ["state", "t+1"]},
                {"type": np.float32, "name": "value_preds", "shape": (1,), "init_value": 0, "offset": 0},
                {"type": np.float32, "name": "returns", "shape": (1,), "init_value": 0, "offset": 0},
                {"type": np.float32, "name": "actions", "shape": (act_shape, ), "init_value": 0, "offset": 0},
                {"type": np.float32, "name": "actions_all", "shape": (num_victim_agents, act_shape), "init_value": 0, "offset": 0, "actions": ["victim", "t+1"]},
                {"type": np.float32, "name": "action_log_probs", "shape": (act_shape, ), "init_value": 0, "offset": 0},
                {"type": np.float32, "name": "rewards", "shape": (1,), "init_value": 0, "offset": 0},
                {"type": np.float32, "name": "masks", "shape": (1,), "init_value": 1, "offset": 1},
                {"type": np.float32, "name": "masks_all", "shape": (num_victim_agents, 1), "init_value": 1, "offset": 1, "actions": ["victim", "t+1"]},
                {"type": np.float32, "name": "bad_masks", "shape": (1,), "init_value": 1, "offset": 1},
                {"type": np.float32, "name": "active_masks", "shape": (1,), "init_value": 1, "offset": 1},
            ]

        if act_space.__class__.__name__ == 'Discrete':
            self.scheme.extend([
                {"type": np.float32, "name": "available_actions", "shape": (act_space.n,), "init_value": 1, "offset": 1},
                {"type": np.float32, "name": "available_actions_all", "shape": (num_victim_agents, act_space.n,), "init_value": 1, "offset": 1, "actions": ["victim", "t+1"]},
            ])
        elif act_space.__class__.__name__ == 'Box':
            self.scheme.extend([
                {"type": np.float32, "name": "available_actions", "shape": (act_space.shape[0],), "init_value": 1, "offset": 1},
                {"type": np.float32, "name": "available_actions_all", "shape": (num_victim_agents, act_space.shape[0],), "init_value": 1, "offset": 1, "actions": ["victim", "t+1"]},
            ])
            
        for dic in self.scheme:
            if "actions" not in dic:
                dic["actions"] = []
            setattr(self, dic["name"], np.ones((self.episode_length + 1, self.n_rollout_threads, self.num_adv_agents, *dic["shape"]), dtype=dic["type"]) * dic["init_value"])

    def insert(self, insert_dic):
        for dic in self.scheme:
            if dic["name"] in insert_dic:
                if "victim" in dic["actions"]:
                    getattr(self, dic["name"])[self.step + dic["offset"]] = np.expand_dims(insert_dic[dic["name"]].copy(), 1).repeat(self.num_adv_agents, 1)
                else:
                    getattr(self, dic["name"])[self.step + dic["offset"]] = insert_dic[dic["name"]].copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        for dic in self.scheme:
            if dic["offset"] == 1:
                getattr(self, dic["name"])[0] = getattr(self, dic["name"])[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(self.episode_length):
                    if self._use_popart:
                        # step + 1
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.episode_length)):
                    if self._use_popart:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(
                            self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.episode_length)):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.episode_length)):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        episode_length = self.episode_length
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents,
                          n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        advantages = advantages.reshape(-1, 1)

        # size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
        for indices in sampler:
            yield_list = []
            for dic in self.scheme:
                attr = getattr(self, dic["name"])
                attr = attr[:-1].reshape(-1, *attr.shape[3:])
                yield_list.append(attr[indices])
                if "t+1" in dic["actions"]:
                    attr = getattr(self, dic["name"])
                    attr = attr[1:].reshape(-1, *attr.shape[3:])
                    yield_list.append(attr[indices])

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]
            yield_list.append(adv_targ)

            yield yield_list
            

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        """
        Yield training data for non-chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        """
        raise NotImplementedError()
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * num_agents
        assert n_rollout_threads * num_agents >= num_mini_batch, (
            "PPO requires the number of processes ({})* number of agents ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_agents, num_mini_batch))
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size).numpy()

        share_obs = self.share_obs.reshape(-1, batch_size, *self.share_obs.shape[3:])
        obs = self.obs.reshape(-1, batch_size, *self.obs.shape[3:])
        rnn_states = self.rnn_states.reshape(-1, batch_size, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic.reshape(-1, batch_size, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions.reshape(-1, batch_size, self.available_actions.shape[-1])
        value_preds = self.value_preds.reshape(-1, batch_size, 1)
        returns = self.returns.reshape(-1, batch_size, 1)
        masks = self.masks.reshape(-1, batch_size, 1)
        active_masks = self.active_masks.reshape(-1, batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(-1, batch_size, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, batch_size, 1)

        for start_ind in range(0, batch_size, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(share_obs[:-1, ind])
                obs_batch.append(obs[:-1, ind])
                rnn_states_batch.append(rnn_states[0:1, ind])
                rnn_states_critic_batch.append(rnn_states_critic[0:1, ind])
                actions_batch.append(actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[:-1, ind])
                value_preds_batch.append(value_preds[:-1, ind])
                return_batch.append(returns[:-1, ind])
                masks_batch.append(masks[:-1, ind])
                active_masks_batch.append(active_masks[:-1, ind])
                old_action_log_probs_batch.append(action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            share_obs_batch = np.stack(share_obs_batch, 1)
            obs_batch = np.stack(obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, dim) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
                  adv_targ, available_actions_batch

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        """
        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param data_chunk_length: (int) length of sequence chunks with which to train RNN.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        episode_length = self.episode_length
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        # other: size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[L,Dim]
        # state: size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
        L, N = data_chunk_length, mini_batch_size

        for indices in sampler:
            yield_list = []
            for dic in self.scheme:
                attr = getattr(self, dic["name"])
                attr = attr[:-1].swapaxes(0, 1).swapaxes(1, 2)
                attr = attr.reshape(-1, *attr.shape[3:])
                attr_batch = []
                if "state" in dic["actions"]:
                    for index in indices:
                        ind = index * data_chunk_length
                        attr_batch.append(attr[ind])
                    attr_batch = np.stack(attr_batch).reshape(N, *attr.shape[1:])
                else:
                    for index in indices:
                        ind = index * data_chunk_length
                        attr_batch.append(attr[ind:ind+data_chunk_length])
                    attr_batch = np.stack(attr_batch, axis=1)
                    attr_batch = _flatten(L, N, attr_batch)
                yield_list.append(attr_batch)

                if "t+1" in dic["actions"]:
                    attr = getattr(self, dic["name"])
                    attr = attr[1:].swapaxes(0, 1).swapaxes(1, 2)
                    attr = attr.reshape(-1, *attr.shape[3:])
                    attr_batch = []
                    if "state" in dic["actions"]:
                        for index in indices:
                            ind = index * data_chunk_length
                            attr_batch.append(attr[ind])
                        attr_batch = np.stack(attr_batch).reshape(N, *attr.shape[1:])
                    else:
                        for index in indices:
                            ind = index * data_chunk_length
                            attr_batch.append(attr[ind:ind+data_chunk_length])
                        attr_batch = np.stack(attr_batch, axis=1)
                        attr_batch = _flatten(L, N, attr_batch)
                    yield_list.append(attr_batch)
            
            advantages_tmp = advantages.swapaxes(0, 1).swapaxes(1, 2)
            advantages_tmp = advantages_tmp.reshape(-1, *advantages_tmp.shape[3:])
            adv_targ = []
            for index in indices:
                ind = index * data_chunk_length
                adv_targ.append(advantages_tmp[ind:ind + data_chunk_length])
            adv_targ = np.stack(adv_targ, axis=1)
            adv_targ = _flatten(L, N, adv_targ)
            yield_list.append(adv_targ)

            yield yield_list
