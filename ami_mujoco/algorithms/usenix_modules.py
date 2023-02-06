import torch
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act_usenix import ACTLayer
from utils.util import get_shape_from_obs_space, get_shape_from_act_space


class StateTransition(nn.Module):
    """
    State transition network class for USENIX method. Outputs obs(t+1) given share_obs(t) and action(t).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param share_obs_space: (gym.Space) share observation space.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, share_obs_space, obs_space, action_space, device=torch.device("cpu")):
        super(StateTransition, self).__init__()
        self.hidden_size = args.hidden_size
        self.args = args
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent = args.attack_use_naive_recurrent
        self._use_recurrent = args.attack_use_recurrent
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        share_obs_shape = get_shape_from_obs_space(share_obs_space)
        obs_shape = get_shape_from_obs_space(obs_space)
        act_shape = get_shape_from_act_space(action_space)

        if action_space.__class__.__name__ == 'Discrete':
            input_shape = share_obs_shape[0] + act_shape * args.num_agents * action_space.n
        elif action_space.__class__.__name__ == "Box":
            input_shape = share_obs_shape[0] + args.num_agents * act_shape
        else:
            raise NotImplementedError
        base = MLPBase
        self.base = base(args, [input_shape])

        if self._use_naive_recurrent or self._use_recurrent:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.obs_fc = nn.Linear(self.hidden_size, obs_shape[0] * args.num_agents)

        self.to(device)

    def forward(self, share_obs, actions_all, rnn_states, masks):
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
        share_obs = check(share_obs).to(**self.tpdv)
        actions_all = check(actions_all).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        state_input = self._build_input(share_obs, actions_all)
        features = self.base(state_input)

        if self._use_naive_recurrent or self._use_recurrent:
            features, rnn_states = self.rnn(features, rnn_states, masks)

        obs_tplus1 = self.obs_fc(features)

        return obs_tplus1, rnn_states

    def transit(self, state_input, rnn_states, masks):
        features = self.base(state_input)

        if self._use_naive_recurrent or self._use_recurrent:
            features, rnn_states = self.rnn(features, rnn_states, masks)

        obs_tplus1 = self.obs_fc(features)

        return obs_tplus1

    def _build_input(self, share_obs, actions_all):
        assert len(share_obs.shape) == 2
        batch_size = share_obs.shape[0]
        if self.args.env_name == 'StarCraft2':
            actions_all_onehot = torch.nn.functional.one_hot(actions_all.long(), num_classes=self.args.n_actions).float()
            state_input = torch.cat([share_obs.view(batch_size, -1), actions_all_onehot.view(batch_size, -1)], dim=-1)
        elif self.args.env_name =='mujoco':
            action_reshaped = actions_all.reshape(batch_size, -1)
            state_input = torch.cat([share_obs, action_reshaped], dim=-1)
        return state_input

class ActionTransition(nn.Module):
    """
    Action transition network class for USENIX method. Outputs action_probs(t) given obs(t) for each agent.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(ActionTransition, self).__init__()
        self.hidden_size = args.hidden_size
        self.args = args
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)

        base = MLPBase
        self.base = base(args, [obs_shape[0] * self.args.num_agents])

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None):
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
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            available_actions = available_actions.view(-1, available_actions.shape[-1])
        
        obs = obs.view(obs.shape[0], -1)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
            
        action_logits = self.act.get_logits(actor_features, available_actions)
        action_logits = action_logits.view(-1, self.args.num_agents, action_logits.shape[-1])

        return action_logits, rnn_states

    def transit(self, obs, rnn_states, masks, available_actions=None):
        if available_actions is not None:
            available_actions = available_actions.view(-1, available_actions.shape[-1])

        obs = obs.view(obs.shape[0], -1)
        
        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_probs = self.act.get_logits(actor_features, available_actions)
        action_probs = action_probs.view(-1, self.args.num_agents, action_probs.shape[-1])

        return action_probs
