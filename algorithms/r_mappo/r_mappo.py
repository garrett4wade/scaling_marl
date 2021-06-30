import torch
import torch.nn as nn
from utils.utils import get_gard_norm, huber_loss, mse_loss
from algorithms.utils.util import check


class R_MAPPO:
    def __init__(self, args, policy):
        self.device = policy.device
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.policy = policy
        self.num_trainers = args.num_trainers

        self.clip_param = args.clip_param
        self.ppo_epoch = args.sample_reuse
        self.entropy_coef = args.entropy_coef
        self.value_coef = args.value_coef
        self.max_grad_norm = args.max_grad_norm

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_active_masks = args.use_active_masks

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

        if self._use_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        assert update_actor, 'currently ppg not supported'
        (share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch,
         v_target_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ,
         available_actions_batch) = (sample['share_obs'], sample['obs'], sample['rnn_states'],
                                     sample['rnn_states_critic'], sample['actions'], sample['values'],
                                     sample['v_target'], sample['masks'], sample['active_masks'],
                                     sample['action_log_probs'], sample['advantages'], sample['available_actions'])

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

        if self._use_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) *
                                  active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, v_target_batch, active_masks_batch)

        self.policy.optimizer.zero_grad()

        (policy_loss + value_loss * self.value_coef - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.actor_critic.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.policy.actor_critic.parameters())

        self.policy.optimizer.step()

        return value_loss, policy_loss, dist_entropy, grad_norm

    def step(self, sample, update_actor=True):
        return self.ppo_update(sample, update_actor)
