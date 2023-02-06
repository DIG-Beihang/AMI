from turtle import position
import numpy as np
import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.popart import PopArt
from algorithms.utils.util import check

class ami():
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

        self._use_recurrent_policy = args.attack_use_recurrent
        self._use_naive_recurrent = args.attack_use_naive_recurrent
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
                
        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
            self.oracle_value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None
            self.oracle_value_normalizer = None
        self.args = args

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


    def cal_oracle_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
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
            error_clipped = self.oracle_value_normalizer(return_batch) - value_pred_clipped
            error_original = self.oracle_value_normalizer(return_batch) - values
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
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, rnn_states_maa_batch, rnn_states_oracle_batch, actions_batch, oracle_actions_batch, actions_all_batch, actions_all_next_batch,\
        value_preds_batch, return_batch, masks_batch, masks_maa_batch, active_masks_batch, active_masks_batch_adv, old_action_log_probs_batch, old_oracle_action_log_probs_batch, old_action_log_probs_all_batch,\
        adv_targ, available_actions_batch, available_actions_all_batch, available_actions_all_next_batch, rnn_states_critic_oracle_batch, oracle_value_preds_batch, oracle_adv_targ, adv_actions_prob_batch, oracle_return_batch, active_masks_all_next_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        active_masks_batch_adv = check(active_masks_batch_adv).to(**self.tpdv)
        
        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch_adv)     
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch_adv).sum() / active_masks_batch_adv.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

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
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch_adv)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def ppo_update_oracle(self, sample, update_actor=True):
        """
        Update oracle actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, rnn_states_maa_batch, rnn_states_oracle_batch, actions_batch, oracle_actions_batch, actions_all_batch, actions_all_next_batch,\
        value_preds_batch, return_batch, masks_batch, masks_maa_batch, active_masks_batch, active_masks_batch_adv, old_action_log_probs_batch, old_oracle_action_log_probs_batch, old_action_log_probs_all_batch,\
        adv_targ, available_actions_batch, available_actions_all_batch, available_actions_all_next_batch, rnn_states_critic_oracle_batch, oracle_value_preds_batch, oracle_adv_targ, adv_actions_prob_batch, oracle_return_batch, active_masks_all_next_batch = sample

        old_oracle_action_log_probs_batch = check(old_oracle_action_log_probs_batch).to(**self.tpdv)
        oracle_adv_targ = check(oracle_adv_targ).to(**self.tpdv)
        oracle_value_preds_batch = check(oracle_value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        active_masks_batch_adv = check(active_masks_batch_adv).to(**self.tpdv)
        # Reshape to do in a single forward pass for all steps
        oracle_values, oracle_action_log_probs, dist_entropy, action_logits_all = self.policy.evaluate_oracle_actions(share_obs_batch, actions_all_batch, oracle_actions_batch, rnn_states_oracle_batch, rnn_states_critic_oracle_batch, masks_batch, available_actions_all_next_batch, active_masks_batch)
        # actor update
        if len(oracle_action_log_probs.shape) == 3:
            oracle_action_log_probs = oracle_action_log_probs.reshape(oracle_action_log_probs.shape[0], -1)
            old_oracle_action_log_probs_batch = old_oracle_action_log_probs_batch.reshape(old_oracle_action_log_probs_batch.shape[0], -1)

        if len(oracle_action_log_probs.shape) != len(old_oracle_action_log_probs_batch.shape):
            old_oracle_action_log_probs_batch = old_oracle_action_log_probs_batch.squeeze()
        imp_weights = torch.exp(oracle_action_log_probs - old_oracle_action_log_probs_batch)

        surr1 = imp_weights * oracle_adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * oracle_adv_targ
        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch_adv).sum() / active_masks_batch_adv.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        policy_loss = policy_action_loss / self.args.num_agents

        self.policy.oracle_optimizer.zero_grad()
        '''
        if update_actor:
            policy_loss.backward()
        '''
        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()
        
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.oracle.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.oracle.parameters())

        self.policy.oracle_optimizer.step()

        # critic update
        value_loss = self.cal_oracle_value_loss(oracle_values, oracle_value_preds_batch, oracle_return_batch, active_masks_batch_adv)

        self.policy.oracle_critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.oracle_critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.oracle_critic.parameters())

        self.policy.oracle_critic_optimizer.step()
        
        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train_maa(self, sample):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, rnn_states_maa_batch, rnn_states_oracle_batch, actions_batch, oracle_actions_batch, actions_all_batch, actions_all_next_batch,\
        value_preds_batch, return_batch, masks_batch, masks_maa_batch, active_masks_batch, active_masks_batch_adv, old_action_log_probs_batch, old_oracle_action_log_probs_batch, old_action_log_probs_all_batch,\
        adv_targ, available_actions_batch, available_actions_all_batch, available_actions_all_next_batch, rnn_states_critic_oracle_batch, oracle_value_preds_batch, oracle_adv_targ, adv_actions_prob_batch, oracle_return_batch, active_masks_all_next_batch = sample
        # idea: shape is (10, 80, 377), so we just treat this like a sequence
        if self.policy.action_type == 'Discrete':
            maa_probs = self.policy.get_maa_mean(share_obs_batch, actions_all_batch, rnn_states_maa_batch, masks_batch, available_actions_all_next_batch)
            maa_probs_log = torch.log(maa_probs + 1e-10).transpose(1, 2)
            actions_all_next_batch = torch.tensor(actions_all_next_batch).to(self.args.device).long()
            masks_batch = check(active_masks_all_next_batch).to(**self.tpdv)
            loss_maa = torch.nn.functional.cross_entropy(maa_probs_log, actions_all_next_batch, reduction='none')
            loss_maa = (loss_maa * masks_batch).sum() / masks_batch.sum()
        elif self.policy.action_type == 'Box':
            actions_all_next_batch = check(actions_all_next_batch).to(**self.tpdv)
            maa_prob = self.policy.get_maa_prob(share_obs_batch, actions_all_batch, rnn_states_maa_batch, masks_batch, actions_all_next_batch, available_actions_all_next_batch)
            loss_maa = maa_prob.mean()
        self.policy.maa_optimizer.zero_grad()
        loss_maa.backward()
        if self._use_max_grad_norm:
            maa_grad_norm = nn.utils.clip_grad_norm_(self.policy.maa.parameters(), self.max_grad_norm)
        else:
            maa_grad_norm = get_gard_norm(self.policy.maa.parameters())
        self.policy.maa_optimizer.step()
        if self.policy.action_type == 'Discrete':
            # by the way, calculate maa accuracy
            maa_pred = maa_probs_log.argmax(dim=1)
            correct = (maa_pred == actions_all_next_batch)
            acc = (correct * masks_batch).sum() / masks_batch.sum()
            acc = acc.item()
        else:
            acc = 0

        return loss_maa, maa_grad_norm, acc

    @torch.no_grad()
    def get_target_influence_reward_continuous(self, buffer):
        maa_all = self.policy.evaluate_maa_probs_oracle(buffer.share_obs[:-1], buffer.actions, buffer.rnn_states_maa[:-1], buffer.masks[:-1], buffer.available_actions_all, buffer.active_masks, buffer.adv_actions_prob, buffer.oracle_actions)
        position_mask = self.get_position_mask_continuous()
        position_mask = check(position_mask).to(**self.tpdv)
        active_masks = check(buffer.active_masks_all[:-1]).to(**self.tpdv)
        maa_all = torch.einsum('ijkl, k -> ijkl', maa_all, position_mask)
        maa_all = maa_all * active_masks
        ami = torch.sum(torch.sum(maa_all, dim=-1), dim=-1)
        ami = ami.unsqueeze(-1).unsqueeze(-1)
        return ami

    def get_position_mask_continuous(self):
        # design space: adv agent is -1 (exploration mode, are there better ways other than the best one?)
        # adv agent is 0 (normal mode, adv agent only influence other agents, don't care itself)
        # adv agent is 1 (coorporative mode, adv agent reach largest advantage under coorporative exploration in next timestep. but might fail to explore)
        position_mask = np.ones(self.args.num_agents)
        position_mask[self.args.adv_agent_ids] = 0
        return position_mask

    def get_position_mask(self, obs):
        start = 0
        end = (self.args.num_agents - 1) * self.args.position_pace + start
        position_mask = obs[:, :, :, start:end:5]
        # assume one adv agent
        position_mask = np.concatenate([position_mask[:, :, :, :self.args.adv_agent_ids[0]], np.expand_dims(np.ones_like(position_mask[:, :, :, 0]) * 0, axis=-1), position_mask[:, :, :, self.args.adv_agent_ids[0]:]], axis=-1)
        return position_mask

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

        if self._use_popart:
            oracle_advantages = buffer.oracle_returns[:-1] - self.oracle_value_normalizer.denormalize(buffer.oracle_value_preds[:-1])
        else:
            oracle_advantages = buffer.oracle_returns[:-1] - buffer.oracle_value_preds[:-1]
        oracle_advantages_copy = oracle_advantages.copy()
        oracle_advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_oracle_advantages = np.nanmean(oracle_advantages_copy)
        std_oracle_advantages = np.nanstd(oracle_advantages_copy)
        oracle_advantages = (oracle_advantages - mean_oracle_advantages) / (std_oracle_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        train_info['oracle_value_loss'] = 0
        train_info['oracle_policy_loss'] = 0
        train_info['oracle_dist_entropy'] = 0
        train_info['oracle_actor_grad_norm'] = 0
        train_info['oracle_critic_grad_norm'] = 0
        train_info['oracle_ratio'] = 0

        train_info['loss_maa'] = 0
        train_info['maa_grad_norm'] = 0
        train_info['maa_acc'] = 0

        train_info['average_step_ami_rewards'] = 0

        for _ in range(self.ppo_epoch):
            
            if self.policy.action_type == 'Discrete':
                data_generator = buffer.recurrent_generator(advantages, oracle_advantages, self.num_mini_batch, self.data_chunk_length)
            elif self.policy.action_type == 'Box':
                data_generator = buffer.recurrent_generator_continuous(advantages, oracle_advantages, self.num_mini_batch, self.data_chunk_length)

            for sample in data_generator:
                loss_maa, maa_grad_norm, maa_acc = self.train_maa(sample)
                oracle_value_loss, oracle_critic_grad_norm, oracle_policy_loss, oracle_dist_entropy, oracle_actor_grad_norm, oracle_imp_weights \
                    = self.ppo_update_oracle(sample, update_actor)
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

                train_info['oracle_value_loss'] += oracle_value_loss.item()
                train_info['oracle_policy_loss'] += oracle_policy_loss.item()
                train_info['oracle_dist_entropy'] += oracle_dist_entropy.item()
                train_info['oracle_actor_grad_norm'] += oracle_actor_grad_norm
                train_info['oracle_critic_grad_norm'] += oracle_critic_grad_norm
                train_info['oracle_ratio'] += oracle_imp_weights.mean()

                train_info['loss_maa'] += loss_maa.item()
                train_info['maa_grad_norm'] += maa_grad_norm.item()
                train_info['maa_acc'] += maa_acc

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.maa.train()
        self.policy.oracle.train()
        self.policy.oracle_critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.maa.eval()
        self.policy.oracle.eval()
        self.policy.oracle_critic.eval()
