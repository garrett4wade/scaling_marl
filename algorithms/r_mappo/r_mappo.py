import numpy as np
import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from algorithms.utils.util import check
import torch.distributed as dist


class R_MAPPO:
    def __init__(self, args, policy):
        self.device = policy.device
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.policy = policy
        self.num_mini_batch = args.num_mini_batch
        self.num_trainers = dist.get_world_size()
        self.slots_per_update = args.slots_per_update

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        self.value_loss_fn = lambda x: huber_loss(x, args.huber_delta) if self._use_huber_loss else mse_loss

    def cal_value_loss(self, values, value_preds_batch, v_target_batch, active_masks_batch):
        error_original = v_target_batch - values
        value_loss_original = self.value_loss_fn(error_original)

        if self._use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -self.clip_param, self.clip_param)
            error_clipped = v_target_batch - value_pred_clipped
            value_loss_clipped = self.value_loss_fn(error_clipped)
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        (share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch,
         v_target_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ,
         available_actions_batch) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        v_target_batch = check(v_target_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch, obs_batch,
                                                                              rnn_states_batch, rnn_states_critic_batch,
                                                                              actions_batch, masks_batch,
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) *
                                  active_masks_batch).sum() / active_masks_batch.sum()
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
        value_loss = self.cal_value_loss(values, value_preds_batch, v_target_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        value_loss.backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm

    def step(self, buffer, update_actor=True):
        train_info = {}

        summary_keys = ['value_loss', 'policy_loss', 'dist_entropy', 'actor_grad_norm', 'critic_grad_norm']
        for k in summary_keys:
            train_info[k] = 0

        for i in range(self.ppo_epoch):
            dist.barrier()
            # only train popart parameter in the first epoch
            slots, data_generator = buffer.get(train_popart=(i == 0), recur=self._use_recurrent_policy)

            # ensure all process get different slot ids
            tensor_list = [torch.zeros(self.slots_per_update).to(self.device) for _ in range(self.num_trainers)]
            dist.all_gather(tensor_list, torch.Tensor(slots.tolist()).to(self.device))
            slot_ids = torch.cat(tensor_list).tolist()
            assert len(np.unique(slot_ids)) == len(slot_ids)

            for sample in data_generator:

                infos = self.ppo_update(sample, update_actor)
                for info in infos:
                    dist.all_reduce(info)
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm = infos

                for k in summary_keys:
                    train_info[k] += locals()[k].item()

            if i == self.ppo_epoch - 1:
                train_info["average_step_rewards"] = np.mean(buffer.rewards[slots])
                train_info['dead_ratio'] = 1 - buffer.active_masks[slots].sum() / np.prod(
                    buffer.active_masks[slots].shape)
            buffer.after_training_step(slots)

        reduce_factor = self.ppo_epoch * self.num_mini_batch * self.num_trainers

        for k in summary_keys:
            train_info[k] /= reduce_factor

        return train_info
