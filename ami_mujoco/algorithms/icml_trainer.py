import numpy as np
import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.popart import PopArt
from algorithms.utils.util import check

class ICML():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
                
        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
            self.value_normalizer_icml = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None
            self.value_normalizer_icml = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch, value_normalizer):
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
            error_clipped = value_normalizer(return_batch) - value_pred_clipped
            error_original = value_normalizer(return_batch) - values
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

    def ppo_update(self, sample, update_actor=True):
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
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, rnn_states_critic_batch_icml, \
        actions_batch, value_preds_batch, value_preds_batch_icml, return_batch, return_batch_icml, masks_batch, \
        active_masks_batch, old_action_log_probs_batch, adv_targ, adv_targ_icml, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        adv_targ_icml = check(adv_targ_icml).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        value_preds_batch_icml = check(value_preds_batch_icml).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        return_batch_icml = check(return_batch_icml).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, values_icml, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              rnn_states_critic_batch_icml, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        surr1_icml = imp_weights * adv_targ_icml
        surr2_icml = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_icml

        if self._use_policy_active_masks:
            policy_action_loss = -(torch.sum(torch.min(surr1, surr2) - torch.min(surr1_icml, surr2_icml),
                                             dim=-1, keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -(torch.sum(torch.min(surr1, surr2) - torch.min(surr1_icml, surr2_icml), 
                                            dim=-1, keepdim=True).mean())

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch, self.value_normalizer)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        # critic_icml update
        value_loss_icml = self.cal_value_loss(values_icml, value_preds_batch_icml, return_batch_icml, active_masks_batch, self.value_normalizer_icml)

        self.policy.critic_optimizer_icml.zero_grad()

        (value_loss_icml * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm_icml = nn.utils.clip_grad_norm_(self.policy.critic_icml.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm_icml = get_gard_norm(self.policy.critic_icml.parameters())

        self.policy.critic_optimizer_icml.step()

        return value_loss, value_loss_icml, critic_grad_norm, critic_grad_norm_icml, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
            advantages_icml = buffer.returns_icml[:-1] - self.value_normalizer_icml.denormalize(buffer.value_preds_icml[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
            advantages_icml = buffer.returns_icml[:-1] - buffer.value_preds_icml[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan

        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        advantages_copy_icml = advantages_icml.copy()
        advantages_copy_icml[buffer.active_masks[:-1] == 0.0] = np.nan

        mean_advantages_icml = np.nanmean(advantages_copy_icml)
        std_advantages_icml = np.nanstd(advantages_copy_icml)
        advantages_icml = (advantages_icml - mean_advantages_icml) / (std_advantages_icml + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['value_loss_icml'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['critic_grad_norm_icml'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, advantages_icml, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, advantages_icml, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, advantages_icml, self.num_mini_batch)

            for sample in data_generator:

                value_loss, value_loss_icml, critic_grad_norm, critic_grad_norm_icml, policy_loss, dist_entropy, \
                actor_grad_norm, imp_weights = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['value_loss_icml'] += value_loss_icml.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['critic_grad_norm_icml'] += critic_grad_norm_icml
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.critic_icml.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.critic_icml.eval()