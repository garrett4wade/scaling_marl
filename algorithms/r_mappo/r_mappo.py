import torch
import torch.nn as nn
from utils.utils import get_gard_norm, huber_loss, mse_loss


class R_MAPPO:
    def __init__(self, cfg, policy):
        self.cfg = cfg

        self.device = policy.device
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.policy = policy

        self.clip_param = cfg.clip_param
        self.entropy_coef = cfg.entropy_coef
        self.value_coef = cfg.value_coef
        self.max_grad_norm = cfg.max_grad_norm

        self._use_max_grad_norm = cfg.use_max_grad_norm
        self._use_clipped_value_loss = cfg.use_clipped_value_loss
        self._use_huber_loss = cfg.use_huber_loss
        self._no_value_active_masks = cfg.no_value_active_masks
        self._no_policy_active_masks = cfg.no_policy_active_masks

        self.value_loss_fn = (lambda x: huber_loss(x, cfg.huber_delta)) if self._use_huber_loss else mse_loss

    def cal_value_loss(self, values, value_preds_batch, v_target_batch, active_masks_batch=None):
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

        if not self._no_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def step(self, sample, update_actor=True):
        assert update_actor, 'currently ppg not supported'
        # self.policy.optimizer.zero_grad()
        for p in self.policy.actor_critic.parameters():
            p.grad = None

        with torch.cuda.amp.autocast():
            # Reshape to do in a single forward pass for all steps
            values, action_log_probs, dist_entropy, v_target = self.policy.evaluate_actions(**sample)

            # actor update
            assert sample['action_log_probs'].shape[-1] == 5, sample['action_log_probs'].shape
            imp_weights = (action_log_probs - sample['action_log_probs']).sum(-1, keepdim=True).exp()

            assert sample['advantages'].shape[-1] == 1, sample['advantages'].shape
            adv_targ = sample['advantages'].sum(-1, keepdim=True)
            surr1 = imp_weights * adv_targ
            surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

            if not self._no_policy_active_masks:
                policy_loss = (-torch.min(surr1, surr2) * sample['active_masks']).sum() / sample['active_masks'].sum()
            else:
                policy_loss = -torch.min(surr1, surr2).mean()

            # critic update
            value_loss = self.cal_value_loss(values, sample['values'], v_target, sample.get('active_masks'))

            loss = policy_loss + value_loss * self.value_coef - dist_entropy * self.entropy_coef

        self.policy.scaler.scale(loss).backward()

        self.policy.scaler.unscale_(self.policy.optimizer)

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.actor_critic.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.policy.actor_critic.parameters())

        self.policy.scaler.step(self.policy.optimizer)
        self.policy.scaler.update()

        return value_loss.detach(), policy_loss.detach(), dist_entropy.detach(), grad_norm.detach()

# import torch
# import torch.nn as nn
# from utils.utils import get_gard_norm, huber_loss, mse_loss


# class R_MAPPO:
#     def __init__(self, cfg, policy):
#         self.cfg = cfg

#         self.device = policy.device
#         self.tpdv = dict(dtype=torch.float32, device=self.device)
#         self.policy = policy

#         self.clip_param = cfg.clip_param
#         self.entropy_coef = cfg.entropy_coef
#         self.value_coef = cfg.value_coef
#         self.max_grad_norm = cfg.max_grad_norm

#         self._use_max_grad_norm = cfg.use_max_grad_norm
#         self._use_clipped_value_loss = cfg.use_clipped_value_loss
#         self._use_huber_loss = cfg.use_huber_loss
#         self._no_value_active_masks = cfg.no_value_active_masks
#         self._no_policy_active_masks = cfg.no_policy_active_masks

#         self.value_loss_fn = (lambda x: huber_loss(x, cfg.huber_delta)) if self._use_huber_loss else mse_loss

#     def cal_value_loss(self, values, value_preds_batch, v_target_batch, active_masks_batch=None):
#         error_original = v_target_batch - values
#         value_loss_original = self.value_loss_fn(error_original)

#         if self._use_clipped_value_loss:
#             value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
#                 -self.clip_param, self.clip_param)
#             error_clipped = v_target_batch - value_pred_clipped
#             value_loss_clipped = self.value_loss_fn(error_clipped)
#             value_loss = torch.max(value_loss_original, value_loss_clipped)
#         else:
#             value_loss = value_loss_original

#         if not self._no_value_active_masks:
#             value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
#         else:
#             value_loss = value_loss.mean()

#         return value_loss

#     def step(self, sample, update_actor=True):
#         assert update_actor, 'currently ppg not supported'
#         # self.policy.optimizer.zero_grad()
#         for p in self.policy.actor_critic.parameters():
#             p.grad = None

#         with torch.cuda.amp.autocast():
#             # Reshape to do in a single forward pass for all steps
#             values, action_log_probs, dist_entropy, v_target = self.policy.evaluate_actions(**sample)

#             # actor update
#             assert sample['action_log_probs'].shape[-1] == 5, sample['action_log_probs'].shape
#             imp_weights = (action_log_probs - sample['action_log_probs']).sum(-1, keepdim=True).exp()

#             # TODO mean advantage
#             assert sample['advantages'].shape[-1] == 1, sample['advantages'].shape
#             adv_targ = sample['advantages'].sum(-1, keepdim=True)
#             surr1 = imp_weights * adv_targ
#             surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

#             if not self._no_policy_active_masks:
#                 policy_loss = (-torch.min(surr1, surr2) * sample['active_masks']).sum() / sample['active_masks'].sum()
#             else:
#                 policy_loss = -torch.min(surr1, surr2).mean()

#             # critic update
#             value_loss = self.cal_value_loss(values, sample['values'], v_target, sample.get('active_masks'))

#             loss = policy_loss + value_loss * self.value_coef - dist_entropy * self.entropy_coef

#         self.policy.scaler.scale(loss).backward()

#         self.policy.scaler.unscale_(self.policy.optimizer)

#         if self._use_max_grad_norm:
#             grad_norm = nn.utils.clip_grad_norm_(self.policy.actor_critic.parameters(), self.max_grad_norm)
#         else:
#             grad_norm = get_gard_norm(self.policy.actor_critic.parameters())

#         self.policy.scaler.step(self.policy.optimizer)
#         self.policy.scaler.update()

#         return value_loss.detach(), policy_loss.detach(), dist_entropy.detach(), grad_norm.detach()
