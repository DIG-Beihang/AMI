import torch
from algorithms.actor_critic import Actor, Critic
from utils.util import update_linear_schedule
from algorithms.maa import MAA
from algorithms.oracle import Oracle, OracleCritic


class ami_Policy:
    """
    ami Policy  class. Wraps actor and critic networks to compute actions and value function predictions.
    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.maa_lr = args.maa_lr
        self.oracle_lr = args.oracle_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.action_type = act_space.__class__.__name__

        self.actor = Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = Critic(args, self.share_obs_space, self.device)
        self.maa = MAA(args, self.share_obs_space, self.act_space, self.device)
        self.oracle = Oracle(args, self.share_obs_space, self.act_space, self.device)
        self.oracle_critic = OracleCritic(args, self.share_obs_space, self.act_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.maa_optimizer = torch.optim.Adam(self.maa.parameters(),
                                                 lr=self.maa_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.oracle_optimizer = torch.optim.Adam(self.oracle.parameters(),
                                                 lr=self.oracle_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

        self.oracle_critic_optimizer = torch.optim.Adam(self.oracle_critic.parameters(),
                                                 lr=self.args.oracle_critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.args = args

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        update_linear_schedule(self.maa_optimizer, episode, episodes, self.maa_lr)
        update_linear_schedule(self.oracle_optimizer, episode, episodes, self.oracle_lr)
        update_linear_schedule(self.oracle_critic_optimizer, episode, episodes, self.oracle_critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_actions_prob(self, obs, rnn_states_actor, masks, available_actions, deterministic):
        action_prob = self.actor.get_logit(obs, rnn_states_actor, masks, available_actions)
        return action_prob
    
    def get_maa_results(self, cent_obs, actions_all, rnn_states_maa, masks, available_actions_all=None,
                    deterministic=False):
        # predict maa. not acturally a part of policy but a part of training
        maa_pred, maa_log_probs, rnn_states_maa = self.maa(cent_obs, 
                                                                actions_all,
                                                                rnn_states_maa,
                                                                masks,
                                                                available_actions_all,
                                                                deterministic)

        return maa_pred, maa_log_probs, rnn_states_maa


    def get_maa_prob(self, share_obs_batch, actions_all_batch, rnn_states_maa_batch, masks_maa_batch, action_all_next, available_actions_all_next_batch):
        action_prob = self.maa.get_prob(share_obs_batch, actions_all_batch, rnn_states_maa_batch, masks_maa_batch, action_all_next, available_actions_all_next_batch)
        return action_prob
        

    def get_oracle_actions(self, cent_obs, actions_all, rnn_states_oracle, masks, available_actions_all=None,
                    deterministic=False):
        # get oracle result. not acturally a part of policy but a part of training
        # critic of oracle is the same with advantage
        oracle_actions, oracle_action_log_probs, rnn_states_oracle = self.oracle(cent_obs, 
                                                                actions_all,
                                                                rnn_states_oracle,
                                                                masks,
                                                                available_actions_all,
                                                                deterministic)

        return oracle_actions, oracle_action_log_probs, rnn_states_oracle

    def get_oracle_critic(self, cent_obs, actions_all, rnn_states, masks):
        # get oracle result. not acturally a part of policy but a part of training
        # critic of oracle is the same with advantage
        oracle_values, rnn_states_oracle = self.oracle_critic(cent_obs, actions_all, rnn_states, masks)

        return oracle_values, rnn_states_oracle
    
    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def get_oracle_values(self, cent_obs, actions_all, rnn_states, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        oracle_values, _ = self.oracle_critic(cent_obs, actions_all, rnn_states, masks)
        return oracle_values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy


    def evaluate_oracle_actions(self, cent_obs, actions_all, oracle_actions_batch, rnn_states_actor, rnn_states_critic, masks, available_actions_all_next, active_masks_batch):
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
        
        action_log_probs, dist_entropy, action_logits_all = self.oracle.evaluate_actions(cent_obs, actions_all, oracle_actions_batch, rnn_states_actor, masks, available_actions_all_next, active_masks_batch)
        oracle_values, _ = self.oracle_critic(cent_obs, actions_all, rnn_states_critic, masks)

        return oracle_values, action_log_probs.squeeze(), dist_entropy, action_logits_all

    def evaluate_maa_probs(self, cent_obs, actions_all, rnn_states_maa, masks, available_actions_all_next, active_masks_batch, adv_actions_prob):
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
        if available_actions_all_next is not None:
            available_actions_all_next = available_actions_all_next[1:]
        maa_all = []
        for i in range(10):
            action_prob_all = self.maa.get_counterfactual_prob(cent_obs, actions_all.copy(), adv_actions_prob, rnn_states_maa, masks, available_actions_all_next)
            maa_all.append(action_prob_all)
        maa_all = torch.stack(maa_all, dim=1).mean(dim=1)
        return maa_all

    def evaluate_maa_probs_oracle(self, cent_obs, actions_all, rnn_states_maa, masks, available_actions_all_next, active_masks_batch, adv_actions_prob, oracle_actions):
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
        if available_actions_all_next is not None:
            available_actions_all_next = available_actions_all_next[1:]
        maa_all = []

        for i in range(10):
            action_probs_all = self.maa.get_counterfactual_prob_oracle(cent_obs, actions_all, adv_actions_prob, rnn_states_maa, masks, oracle_actions, available_actions_all_next)
            maa_all.append(action_probs_all)
        maa_all = torch.stack(maa_all, dim=1).mean(dim=1)
        return maa_all

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor

    
