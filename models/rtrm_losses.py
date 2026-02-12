from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        loss_type: str,
        stickiness_weight: float = 0.0,
        stickiness_residual_type: str = "logits_l2",
        stickiness_horizon: int = 1,
        stickiness_gamma: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.stickiness_weight = stickiness_weight
        self.stickiness_residual_type = stickiness_residual_type
        self.stickiness_horizon = stickiness_horizon
        self.stickiness_gamma = stickiness_gamma

        if self.stickiness_residual_type not in {"logits_l2"}:
            raise ValueError("stickiness_residual_type must be logits_l2.")

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        stick_loss = torch.zeros((), dtype=lm_loss.dtype, device=lm_loss.device)
        if self.stickiness_weight != 0.0:
            carry = model_kwargs["carry"]
            streak = carry.correct_streak
            reset_flag = carry.halted

            stick_mask = (streak > 0) & (streak <= self.stickiness_horizon) & (~reset_flag)
            if torch.any(stick_mask):
                delta = outputs["logits"].to(torch.float32) - carry.prev_logits.to(torch.float32)
                drift_tok = delta.square().sum(dim=-1)
                drift_seq = (drift_tok * mask.to(drift_tok.dtype)).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1).to(drift_tok.dtype)

                streak_minus_one = (streak - 1).clamp_min(0).to(drift_seq.dtype)
                stick_weight = torch.pow(
                    torch.full_like(drift_seq, self.stickiness_gamma),
                    streak_minus_one,
                )
                stick_loss = (stick_mask.to(drift_seq.dtype) * stick_weight * drift_seq).sum().to(lm_loss.dtype)
        metrics["stick_loss"] = stick_loss.detach()

        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = torch.zeros((), dtype=lm_loss.dtype, device=lm_loss.device)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + self.stickiness_weight * stick_loss
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
