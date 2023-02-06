import numpy as np
import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.popart import PopArt
from algorithms.utils.util import check
from algorithms.usenix_policy import USENIX_Policy

class USENIX():
    """
    Trainer class for USENIX to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (USENIX_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy: USENIX_Policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        
        self.args = args
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta
        self.epsilon_state = args.epsilon_state
        self.epsilon_action = args.epsilon_action
        self.n_actions = args.n_actions
        self.adv_agent_ids = args.adv_agent_ids

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._attack_use_recurrent = args.attack_use_recurrent
        self._attack_use_naive_recurrent = args.attack_use_naive_recurrent
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
                
        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart:
            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, adv_loss, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        current_adv_batch, share_obs_batch, obs_batch, obs_tplus1_batch, obs_all_batch, obs_all_tplus1_batch, rnn_states_batch, rnn_states_critic_batch,\
        rnn_states_state_batch, rnn_states_action_batch, rnn_states_action_tplus1_batch, value_preds_batch, return_batch, \
        actions_batch, actions_all_batch, actions_all_tplus1_batch, old_action_log_probs_batch, rewards_batch, \
        masks_batch, masks_all_batch, masks_all_tplus1_batch, bad_masks_batch, active_masks_batch, \
        available_actions_batch, available_actions_all_batch, available_actions_all_tplus1_batch, adv_targ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef + adv_loss).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def state_update(self, sample):
        """
        Update state transition model.
        """
        current_adv_batch, share_obs_batch, obs_batch, obs_tplus1_batch, obs_all_batch, obs_all_tplus1_batch, rnn_states_batch, rnn_states_critic_batch,\
        rnn_states_state_batch, rnn_states_action_batch, rnn_states_action_tplus1_batch, value_preds_batch, return_batch, \
        actions_batch, actions_all_batch, actions_all_tplus1_batch, old_action_log_probs_batch, rewards_batch, \
        masks_batch, masks_all_batch, masks_all_tplus1_batch, bad_masks_batch, active_masks_batch, \
        available_actions_batch, available_actions_all_batch, available_actions_all_tplus1_batch, adv_targ = sample

        share_obs_batch = check(share_obs_batch).to(**self.tpdv)
        actions_all_batch = check(actions_all_batch).to(**self.tpdv)
        rnn_states_state_batch = check(rnn_states_state_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)
        obs_all_batch = check(obs_all_batch).to(**self.tpdv)

        obs_tplus1, _ = self.policy.state_transition(share_obs_batch,
                                                     actions_all_batch,
                                                     rnn_states_state_batch,
                                                     masks_batch)

        state_gap = torch.abs(obs_tplus1 - obs_all_batch.view(obs_tplus1.shape)) * masks_batch
        state_gap_linf_norm = torch.norm(state_gap, p=float("inf"))
        state_loss = torch.norm(state_gap - self.epsilon_state, p=2) / masks_batch.sum()

        self.policy.state_optimizer.zero_grad()
        state_loss.backward()
        if self._use_max_grad_norm:
            state_grad_norm = nn.utils.clip_grad_norm_(self.policy.state_transition.parameters(), self.max_grad_norm)
        else:
            state_grad_norm = get_gard_norm(self.policy.state_transition.parameters())
        self.policy.state_optimizer.step()

        return state_loss, state_gap_linf_norm, state_grad_norm

    def action_update(self, sample):
        """
        Update action transition model.
        """
        current_adv_batch, share_obs_batch, obs_batch, obs_tplus1_batch, obs_all_batch, obs_all_tplus1_batch, rnn_states_batch, rnn_states_critic_batch,\
        rnn_states_state_batch, rnn_states_action_batch, rnn_states_action_tplus1_batch, value_preds_batch, return_batch, \
        actions_batch, actions_all_batch, actions_all_tplus1_batch, old_action_log_probs_batch, rewards_batch, \
        masks_batch, masks_all_batch, masks_all_tplus1_batch, bad_masks_batch, active_masks_batch, \
        available_actions_batch, available_actions_all_batch, available_actions_all_tplus1_batch, adv_targ = sample

        obs_all_batch = check(obs_all_batch).to(**self.tpdv)
        rnn_states_action_batch = check(rnn_states_action_batch).to(**self.tpdv)
        masks_all_batch = check(masks_all_batch).to(**self.tpdv)
        available_actions_all_batch = check(available_actions_all_batch).to(**self.tpdv)
        actions_all_batch = check(actions_all_batch).to(**self.tpdv)

        actions_all, _ = self.policy.action_transition(obs_all_batch,
                                                     rnn_states_action_batch,
                                                     masks_all_batch,
                                                     available_actions_all_batch)

        if self.args.env_name == "StarCraft2":
            n_actions = actions_all.shape[-1]
            action_loss = torch.nn.functional.cross_entropy(actions_all.view(-1, n_actions), actions_all_batch.view(-1).long(), reduction='none')
            action_loss = (action_loss * masks_all_batch.view(-1)).sum() / masks_all_batch.sum()
        elif self.args.env_name == "mujoco":
            n_actions = actions_all_batch.shape[-1]
            action_gap = (actions_all - actions_all_batch) * masks_all_batch
            action_loss = torch.norm(action_gap - self.args.epsilon_action, p=2) / masks_all_batch.sum()

        self.policy.action_optimizer.zero_grad()
        action_loss.backward()
        if self._use_max_grad_norm:
            action_grad_norm = nn.utils.clip_grad_norm_(self.policy.action_transition.parameters(), self.max_grad_norm)
        else:
            action_grad_norm = get_gard_norm(self.policy.action_transition.parameters())
        self.policy.action_optimizer.step()

        return action_loss, action_grad_norm

    def get_adv_loss(self, sample, lamb):
        """
        Calculate adversarial loss for USENIX method.
        """
        current_adv_batch, share_obs_batch, obs_batch, obs_tplus1_batch, obs_all_batch, obs_all_tplus1_batch, rnn_states_batch, rnn_states_critic_batch,\
        rnn_states_state_batch, rnn_states_action_batch, rnn_states_action_tplus1_batch, value_preds_batch, return_batch, \
        actions_batch, actions_all_batch, actions_all_tplus1_batch, old_action_log_probs_batch, rewards_batch, \
        masks_batch, masks_all_batch, masks_all_tplus1_batch, bad_masks_batch, active_masks_batch, \
        available_actions_batch, available_actions_all_batch, available_actions_all_tplus1_batch, adv_targ = sample

        obs_batch = check(obs_batch).to(**self.tpdv)
        rnn_states_batch = check(rnn_states_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)
        available_actions_batch = check(available_actions_batch).to(**self.tpdv)

        share_obs_batch = check(share_obs_batch).to(**self.tpdv)
        actions_all_batch = check(actions_all_batch).to(**self.tpdv)
        current_adv_batch = check(current_adv_batch).to(**self.tpdv)
        rnn_states_state_batch = check(rnn_states_state_batch).to(**self.tpdv)

        masks_all_tplus1_batch = check(masks_all_batch).to(**self.tpdv)
        available_actions_all_tplus1_batch = check(available_actions_all_batch).to(**self.tpdv)
        rnn_states_action_tplus1_batch = check(rnn_states_action_batch).to(**self.tpdv)

        obs_all_tplus1_batch = check(obs_all_tplus1_batch).to(**self.tpdv)
        actions_all_tplus1_batch = check(actions_all_tplus1_batch).to(**self.tpdv)

        # forward policy model and gumbel softmax
        action_logits, _ = self.policy.get_action_logits(obs_batch, rnn_states_batch, masks_batch, available_actions_batch)
        bs = share_obs_batch.shape[0]
        
        if self.args.env_name == "StarCraft2":
            action_gumbel = torch.nn.functional.gumbel_softmax(action_logits, hard=True, dim=-1)
            actions_all_onehot = torch.nn.functional.one_hot(actions_all_batch.long(), num_classes=self.n_actions).float().squeeze(2)
            current_adv_batch = current_adv_batch.unsqueeze(2).expand_as(actions_all_onehot).long()
            action_gumbel = action_gumbel.unsqueeze(1).expand_as(actions_all_onehot)
            actions_all_onehot.scatter_(1, current_adv_batch, action_gumbel)
            state_input = torch.cat([share_obs_batch.view(bs, -1), actions_all_onehot.view(bs, -1)], dim=-1)
        elif self.args.env_name == "mujoco":
            actions_all_copy = actions_all_batch.clone()
            current_adv_batch = current_adv_batch.unsqueeze(2).expand_as(actions_all_copy).long()
            action_logits = action_logits.unsqueeze(1).expand_as(actions_all_copy)
            actions_all_copy.scatter_(1, current_adv_batch, action_logits)
            state_input = torch.cat([share_obs_batch.view(bs, -1), actions_all_copy.view(bs, -1)], dim=-1)
        
        obs_tplus1 = self.policy.state_transition.transit(state_input, rnn_states_state_batch, masks_batch)

        # forward action transition model
        action_logits_tplus1 = self.policy.action_transition.transit(obs_tplus1, 
                                                                    rnn_states_action_tplus1_batch, 
                                                                    masks_all_tplus1_batch, 
                                                                    available_actions_all_tplus1_batch)

        state_gap = torch.abs(obs_tplus1 - obs_all_tplus1_batch.view(obs_tplus1.shape)) * masks_batch
        state_gap = state_gap.view(state_gap.shape[0], -1)
        state_loss = torch.norm(state_gap - self.epsilon_state, p=2, dim=-1)
        
        if self.args.env_name == "StarCraft2":
            # why I wrote this?
            action_loss = torch.nn.functional.cross_entropy(action_logits_tplus1.view(-1, action_logits_tplus1.shape[-1]),
                                                        actions_all_tplus1_batch.view(-1).long(), reduction='none')
            action_loss = action_loss.view(state_loss.shape[0], -1, 1) * masks_all_tplus1_batch
            action_loss = action_loss.sum(dim=1) / (action_loss.shape[1] - len(self.adv_agent_ids)).squeeze()
        elif self.args.env_name == "mujoco":
            action_gap = (action_logits_tplus1 - actions_all_tplus1_batch) * masks_all_tplus1_batch
            action_loss = torch.norm(action_gap - self.args.epsilon_action, p=2, dim=[1,2])

        adv_loss = (((state_loss - action_loss) * lamb) * masks_batch.squeeze()).sum() / masks_batch.sum()

        return adv_loss, state_loss.mean(), action_loss.mean()

    def get_lambda(self, sample):
        """
        Calculate lambda for USENIX method.
        """
        current_adv_batch, share_obs_batch, obs_batch, obs_tplus1_batch, obs_all_batch, obs_all_tplus1_batch, rnn_states_batch, rnn_states_critic_batch,\
        rnn_states_state_batch, rnn_states_action_batch, rnn_states_action_tplus1_batch, value_preds_batch, return_batch, \
        actions_batch, actions_all_batch, actions_all_tplus1_batch, old_action_log_probs_batch, rewards_batch, \
        masks_batch, masks_all_batch, masks_all_tplus1_batch, bad_masks_batch, active_masks_batch, \
        available_actions_batch, available_actions_all_batch, available_actions_all_tplus1_batch, adv_targ = sample

        share_obs_batch = check(share_obs_batch).to(**self.tpdv)
        actions_all_batch = check(actions_all_batch).to(**self.tpdv)
        rnn_states_state_batch = check(rnn_states_state_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)
        obs_all_batch = check(obs_all_batch).to(**self.tpdv)

        share_obs_batch.requires_grad_(True)
        state_input = self.policy.state_transition._build_input(share_obs_batch, actions_all_batch)
        obs_tplus1 = self.policy.state_transition.transit(state_input, rnn_states_state_batch, masks_batch)

        obs_coe = torch.ones_like(obs_tplus1)
        obs_tplus1.backward(obs_coe)
        aap_mask = torch.zeros_like(share_obs_batch)

        if self.args.env_name == "StarCraft2":
            # share obs shape: num_agents * 21 + num_enemies * 20
            for i in self.adv_agent_ids:
                aap_mask[i*21:(i+1)*21] = 1
        elif self.args.env_name == "mujoco":
            pass
        
        obs_grad = share_obs_batch * (aap_mask * share_obs_batch.grad)
        grad_output, _ = self.policy.state_transition(obs_grad, actions_all_batch, rnn_states_state_batch, masks_batch)
        
        I_t = (obs_tplus1 - grad_output).detach() * masks_batch
        I_t = torch.max(torch.abs(I_t), dim=-1)[0]

        return 1 / (I_t + 1)

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {
            'value_loss': 0,
            'policy_loss': 0,
            'dist_entropy': 0,
            'actor_grad_norm': 0,
            'critic_grad_norm': 0,
            'ratio': 0,
            'state_loss': 0,
            'state_gap_linf_norm': 0,
            'state_grad_norm': 0,
            'action_loss': 0,
            'action_grad_norm': 0,
            'lambda_mean': 0,
            'adv_loss': 0,
            'adv_loss_state': 0,
            'adv_loss_action': 0
        }

        # Step 1: update state transition model H
        for _ in range(self.ppo_epoch):
            if self._attack_use_recurrent:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._attack_use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                state_loss, state_gap_linf_norm, state_grad_norm = self.state_update(sample)

                train_info['state_loss'] += state_loss.item()
                train_info['state_gap_linf_norm'] += state_gap_linf_norm.item()
                train_info['state_grad_norm'] += state_grad_norm

        # Step 2: update action transition model F
        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                action_loss, action_grad_norm = self.action_update(sample)

                train_info['action_loss'] += action_loss.item()
                train_info['action_grad_norm'] += action_grad_norm

        # Step 3: update policy by PPO loss + adv loss
        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                lamb = self.get_lambda(sample)
                adv_loss, adv_loss_state, adv_loss_action = self.get_adv_loss(sample, lamb)
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, adv_loss, update_actor)

                train_info['lambda_mean'] += lamb.mean().item()
                train_info['adv_loss'] += adv_loss.item()
                train_info['adv_loss_state'] += adv_loss_state.item()
                train_info['adv_loss_action'] += adv_loss_action.item()
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.state_transition.train()
        self.policy.action_transition.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.state_transition.eval()
        self.policy.action_transition.eval()
