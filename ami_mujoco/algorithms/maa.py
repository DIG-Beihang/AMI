import torch
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act_ami import ACTLayer
from utils.util import get_maa_shape
from algorithms.utils.distributions import FixedCategorical, FixedNormal


class MAA(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(MAA, self).__init__()
        self.action_type = action_space.__class__.__name__
        self.hidden_size = args.hidden_size
        self.args = args
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_maa_shape(args, obs_space, action_space)
        base = MLPBase
        self.base = base(args, [obs_shape])

        if self.args.attack_use_recurrent or self.args.attack_use_naive_recurrent:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)

    def forward(self, cent_obs, actions_all, rnn_states_maa, masks, available_actions_all=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        actions_all = check(actions_all).to(**self.tpdv)
        rnn_states_maa = check(rnn_states_maa).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions_all is not None:
            available_actions_all = check(available_actions_all).to(**self.tpdv)
        maa_input = self._get_maa_input(cent_obs, actions_all)

        actor_features = self.base(maa_input)

        if self.args.attack_use_recurrent or self.args.attack_use_naive_recurrent:
            actor_features, rnn_states_maa = self.rnn(actor_features, rnn_states_maa, masks)

        actions, action_log_probs = self.act(actor_features, available_actions_all, deterministic)

        return actions, action_log_probs, rnn_states_maa

    def get_mean(self, cent_obs, actions_all, rnn_states_maa, masks, available_actions_all_next=None, deterministic=False):

        """
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        batch_size = masks.shape[0]
        cent_obs = check(cent_obs).to(**self.tpdv).squeeze()
        actions_all = check(actions_all).to(**self.tpdv).squeeze()
        rnn_states_maa = check(rnn_states_maa).to(**self.tpdv)
        
        masks = check(masks).to(**self.tpdv)
        if available_actions_all_next is not None:
            available_actions_all_next = check(available_actions_all_next).to(**self.tpdv)
        maa_input = self._get_maa_input(cent_obs, actions_all)
        actor_features = self.base(maa_input)
        
        if self.args.attack_use_recurrent or self.args.attack_use_naive_recurrent:
            actor_features, rnn_states_maa = self.rnn(actor_features, rnn_states_maa, masks)
        actions_prob = self.act.get_probs(actor_features, available_actions_all_next)
        actions_prob = actions_prob.reshape(batch_size, self.args.num_agents, self.args.n_actions)
        return actions_prob

    def get_prob(self, cent_obs, actions_all, rnn_states_maa, masks, action_all_next, available_actions_all_next=None, deterministic=False):

        """
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        batch_size = masks.shape[0]
        cent_obs = check(cent_obs).to(**self.tpdv).squeeze()
        actions_all = check(actions_all).to(**self.tpdv).squeeze()
        rnn_states_maa = check(rnn_states_maa).to(**self.tpdv)
        action_all_next = check(action_all_next).to(**self.tpdv)
        
        masks = check(masks).to(**self.tpdv)
        if available_actions_all_next is not None:
            available_actions_all_next = check(available_actions_all_next).to(**self.tpdv)
        maa_input = self._get_maa_input(cent_obs, actions_all)
        actor_features = self.base(maa_input)
        
        if self.args.attack_use_recurrent or self.args.attack_use_naive_recurrent:
            actor_features, rnn_states_maa = self.rnn(actor_features, rnn_states_maa, masks)
        actions_logits = self.act.get_distri(actor_features, available_actions_all_next)
        action_all_next = action_all_next.view(-1, action_all_next.shape[-1])
        actions_prob = - actions_logits.log_probs(action_all_next)
        return actions_prob

    def get_counterfactual_prob(self, cent_obs, actions_all, adv_actions_prob, rnn_states_maa, masks, available_actions_all_next=None, deterministic=False):

        """
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        batch_size = masks.shape[0]
        cent_obs = check(cent_obs).to(**self.tpdv).squeeze()
        actions_all = check(actions_all).to(**self.tpdv).squeeze()
        rnn_states_maa = check(rnn_states_maa).to(**self.tpdv)
        rnn_states_maa = rnn_states_maa[0].squeeze().unsqueeze(1)
        
        masks = check(masks).to(**self.tpdv)
        if available_actions_all_next is not None:
            available_actions_all_next = check(available_actions_all_next).to(**self.tpdv)

        adv_actions_prob = check(adv_actions_prob.reshape(-1, adv_actions_prob.shape[-1]))
        adv_actions_distrib = FixedCategorical(logits=adv_actions_prob)
        counterfactual_adv_actions = adv_actions_distrib.sample()
        counterfactual_adv_actions = counterfactual_adv_actions.reshape(-1, self.args.n_rollout_threads, counterfactual_adv_actions.shape[-1])
        # SMAC: [160, 2, 11], [160, 2, 1]
        # Mujoco: [1000, 4, 2, 3], [1000, 4, 3]
        actions_all[:, :, self.args.adv_agent_ids] = counterfactual_adv_actions.to(**self.tpdv).float()

        maa_input = self._get_maa_input(cent_obs, actions_all)
        maa_input = maa_input.reshape(-1, maa_input.shape[-1])
        actor_features = self.base(maa_input)
        if self.args.attack_use_recurrent or self.args.attack_use_naive_recurrent:
            actor_features, rnn_states_maa = self.rnn(actor_features, rnn_states_maa, masks)
        
        actions_prob = self.act.get_probs(actor_features, available_actions_all_next)
        actions_prob = actions_prob.reshape(-1, self.args.n_rollout_threads, self.args.num_agents, self.args.n_actions)
        
        return actions_prob

    def get_counterfactual_prob_oracle(self, cent_obs, actions_all, adv_actions_prob, rnn_states_maa, masks, oracle_actions, available_actions_all_next=None, deterministic=False):

        """
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        batch_size = masks.shape[0]
        cent_obs = check(cent_obs).to(**self.tpdv).squeeze()
        actions_all = check(actions_all).to(**self.tpdv).squeeze()
        rnn_states_maa = check(rnn_states_maa).to(**self.tpdv)
        oracle_actions = check(oracle_actions).to(**self.tpdv)
        rnn_states_maa = rnn_states_maa[0].squeeze().unsqueeze(1)
        
        masks = check(masks).to(**self.tpdv)
        if available_actions_all_next is not None:
            available_actions_all_next = check(available_actions_all_next).to(**self.tpdv)
        
        adv_actions_prob = check(adv_actions_prob.reshape(-1, adv_actions_prob.shape[-1]))
        if self.args.env_name == 'StarCraft2':
            adv_actions_distrib = FixedCategorical(adv_actions_prob)
        elif self.args.env_name == 'mujoco':
            adv_actions_distrib = FixedNormal(adv_actions_prob[..., :self.args.n_actions], adv_actions_prob[..., self.args.n_actions:])
        
        counterfactual_adv_actions = adv_actions_distrib.sample()
        counterfactual_adv_actions = counterfactual_adv_actions.reshape(-1, self.args.n_rollout_threads, counterfactual_adv_actions.shape[-1])
        if len(actions_all.shape) != len(counterfactual_adv_actions.shape):
            counterfactual_adv_actions = counterfactual_adv_actions.unsqueeze(-2)
        # SMAC: [160, 2, 11], [160, 2, 1]
        # Mujoco: [1000, 4, 2, 3], [1000, 4, 3]

        # [160, 2, 11], [160, 2, 1, 1]
        # [1000, 4, 2, 3], [1000, 4, 1, 3]
        actions_all[:, :, self.args.adv_agent_ids] = counterfactual_adv_actions.to(**self.tpdv).float()
        
        maa_input = self._get_maa_input(cent_obs, actions_all)
        maa_input = maa_input.reshape(-1, maa_input.shape[-1])
        actor_features = self.base(maa_input)
        if self.args.attack_use_recurrent or self.args.attack_use_naive_recurrent:
            actor_features, rnn_states_maa = self.rnn(actor_features, rnn_states_maa, masks[:, :, 0, 0])
        actions_prob = self.act.get_action_probs(actor_features, oracle_actions, available_actions_all_next)
        if self.action_type == 'Discrete':
            actions_prob = actions_prob.reshape(-1, self.args.n_rollout_threads, self.args.num_agents, 1)
        elif self.action_type == 'Box':
            actions_prob = actions_prob.reshape(-1, self.args.n_rollout_threads, self.args.num_agents, self.args.n_actions)
        return actions_prob

    def _get_maa_input(self, cent_obs, actions_all):
        if len(cent_obs.shape) == 2:
            batch_size = cent_obs.shape[0]
            if self.args.env_name == 'StarCraft2':
                actions_all_onehot = torch.nn.functional.one_hot(actions_all.long(), num_classes=self.args.n_actions).float()
                maa_input = torch.cat([cent_obs.view(batch_size, -1), actions_all_onehot.view(batch_size, -1)], dim=-1)
            elif self.args.env_name =='mujoco':
                action_reshaped = actions_all.reshape(batch_size, -1)
                maa_input = torch.cat([cent_obs, action_reshaped], dim=-1)
        elif len(cent_obs.shape) == 3:
            batch_size = cent_obs.shape[1]
            timestep = cent_obs.shape[0]
            if self.args.env_name == 'StarCraft2':
                actions_all_onehot = torch.nn.functional.one_hot(actions_all.long(), num_classes=self.args.n_actions).float()
                maa_input = torch.cat([cent_obs.view(timestep, batch_size, -1), actions_all_onehot.view(timestep, batch_size, -1)], dim=-1)
            elif self.args.env_name =='mujoco':
                actions_reshaped = actions_all.reshape(timestep, batch_size, -1)
                maa_input = torch.cat([cent_obs, actions_reshaped], dim=-1)
        else:
            raise ValueError('invalid input shape, should be 2 or 3')
        return maa_input