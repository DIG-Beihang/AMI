import torch
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act_ami import ACTLayer
from utils.util import get_oracle_shape


class Oracle(nn.Module):
    """
    oracle network for ami. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(Oracle, self).__init__()
        self.hidden_size = args.hidden_size
        self.args = args
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_oracle_shape(args, obs_space, action_space)
        base = MLPBase
        self.base = base(args, [obs_shape])

        if self.args.attack_use_recurrent or self.args.attack_use_naive_recurrent:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)

    def forward(self, cent_obs, actions_all, rnn_states_oracle, masks, available_actions_all=None, deterministic=False):
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
        rnn_states_oracle = check(rnn_states_oracle).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions_all is not None:
            available_actions_all = check(available_actions_all).to(**self.tpdv)
        oracle_input = self._get_oracle_input(cent_obs, actions_all)

        actor_features = self.base(oracle_input)

        if self.args.attack_use_recurrent or self.args.attack_use_naive_recurrent:
            actor_features, rnn_states_oracle = self.rnn(actor_features, rnn_states_oracle, masks)
        actions, action_log_probs = self.act(actor_features, available_actions_all, deterministic)

        return actions, action_log_probs, rnn_states_oracle

    '''
    def get_actions(self, cent_obs, actions_all, rnn_states_oracle, masks, available_actions_all=None, deterministic=False):
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
        rnn_states_oracle = torch.zeros_like(rnn_states_oracle[])
        masks = check(masks).to(**self.tpdv)
        if available_actions_all is not None:
            available_actions_all = check(available_actions_all).to(**self.tpdv)
        oracle_input = torch.cat([cent_obs.squeeze(), actions_all.squeeze()], dim=-1)
        actor_features = self.base(oracle_input)

        if self.args.attack_use_recurrent or self.args.attack_use_naive_recurrent:
            actor_features, rnn_states_oracle = self.rnn(actor_features, rnn_states_oracle, masks)

        actions, action_log_probs = self.act(actor_features, available_actions_all, deterministic)

        return actions, action_log_probs, rnn_states_oracle
        '''

    def evaluate_actions(self, cent_obs, actions_all, oracle_actions_batch, rnn_states, masks, available_actions_all_next, active_masks_batch):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        batch_size = masks.shape[0]

        cent_obs = check(cent_obs).to(**self.tpdv)
        actions_all = check(actions_all).to(**self.tpdv)
        oracle_actions = check(oracle_actions_batch).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if active_masks_batch is not None:
            active_masks = check(active_masks_batch).to(**self.tpdv)
        if available_actions_all_next is not None:
            available_actions_all_next = check(available_actions_all_next).to(**self.tpdv)
        oracle_input = self._get_oracle_input(cent_obs, actions_all)
        actor_features = self.base(oracle_input)

        if self.args.attack_use_recurrent or self.args.attack_use_naive_recurrent:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        
        action_log_probs, dist_entropy, action_logits_all = self.act.evaluate_actions(actor_features, oracle_actions, available_actions_all_next, active_masks)
        # actions_prob = actions_prob.reshape(batch_size, self.args.num_agents, self.args.n_actions)
        # actions, action_log_probs = self.act(actor_features, available_actions_all, deterministic)

        return action_log_probs, dist_entropy, action_logits_all


    def _get_oracle_input(self, cent_obs, actions_all):
        if len(cent_obs.shape) == 2:
            batch_size = cent_obs.shape[0]
            if self.args.env_name == 'StarCraft2':
                actions_all_onehot = torch.nn.functional.one_hot(actions_all.long(), num_classes=self.args.n_actions).float()
                oracle_input = torch.cat([cent_obs.view(batch_size, -1), actions_all_onehot.view(batch_size, -1)], dim=-1)
            elif self.args.env_name =='mujoco':
                action_reshaped = actions_all.reshape(batch_size, -1)
                oracle_input = torch.cat([cent_obs, action_reshaped], dim=-1)
        elif len(cent_obs.shape) == 3:
            batch_size = cent_obs.shape[1]
            timestep = cent_obs.shape[0]
            if self.args.env_name == 'StarCraft2':
                actions_all_onehot = torch.nn.functional.one_hot(actions_all.long(), num_classes=self.args.n_actions).float()
                oracle_input = torch.cat([cent_obs.view(timestep, batch_size, -1), actions_all_onehot.view(timestep, batch_size, -1)], dim=-1)
            elif self.args.env_name =='mujoco':
                raise NotImplementedError('coming soon!')
        else:
            raise ValueError('invalid input shape, should be 2 or 3')
        return oracle_input


class OracleCritic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, action_space, device=torch.device("cpu")):
        super(OracleCritic, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        oracle_shape = get_oracle_shape(args, cent_obs_space, action_space)
        base = MLPBase
        self.base = base(args, [oracle_shape])

        if self.args.attack_use_recurrent or self.args.attack_use_naive_recurrent:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, actions_all, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        actions_all = check(actions_all).to(**self.tpdv)

        oracle_input = self._get_oracle_input(cent_obs, actions_all)

        critic_features = self.base(oracle_input)
        if self.args.attack_use_recurrent or self.args.attack_use_naive_recurrent:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states

    def _get_oracle_input(self, cent_obs, actions_all):
        if len(cent_obs.shape) == 2:
            batch_size = cent_obs.shape[0]
        
            if self.args.env_name == 'StarCraft2':
                actions_all_onehot = torch.nn.functional.one_hot(actions_all.long(), num_classes=self.args.n_actions).float()
                oracle_input = torch.cat([cent_obs.view(batch_size, -1), actions_all_onehot.view(batch_size, -1)], dim=-1)
            elif self.args.env_name =='mujoco':
                action_reshaped = actions_all.reshape(batch_size, -1)
                oracle_input = torch.cat([cent_obs, action_reshaped], dim=-1)
        elif len(cent_obs.shape) == 3:
            batch_size = cent_obs.shape[1]
            timestep = cent_obs.shape[0]
            if self.args.env_name == 'StarCraft2':
                actions_all_onehot = torch.nn.functional.one_hot(actions_all.long(), num_classes=self.args.n_actions).float()
                oracle_input = torch.cat([cent_obs.view(timestep, batch_size, -1), actions_all_onehot.view(timestep, batch_size, -1)], dim=-1)
            elif self.args.env_name =='mujoco':
                raise NotImplementedError('coming soon!')
        else:
            raise ValueError('invalid input shape, should be 2 or 3')
        return oracle_input