from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    Attention,
    CastedEmbedding,
    CastedLinear,
    CosSin,
    RotaryEmbedding,
    SwiGLU,
    rms_norm,
)
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, empty_value: float) -> torch.Tensor:
    mask_f = mask.to(values.dtype)
    counts = mask_f.sum(dim=-1)
    mean = (values * mask_f).sum(dim=-1) / counts.clamp_min(1.0)
    return torch.where(counts > 0, mean, torch.full_like(mean, empty_value))


def _masked_max(values: torch.Tensor, mask: torch.Tensor, empty_value: float) -> torch.Tensor:
    masked = torch.where(mask, values, torch.full_like(values, empty_value))
    return masked.max(dim=-1).values


def _masked_min(values: torch.Tensor, mask: torch.Tensor, empty_value: float) -> torch.Tensor:
    masked = torch.where(mask, values, torch.full_like(values, empty_value))
    return masked.min(dim=-1).values


@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]

    prev_logits: torch.Tensor
    prev_preds: torch.Tensor
    prev_seq_is_correct: torch.Tensor
    correct_streak: torch.Tensor
    residual_below_count: torch.Tensor

    residual_trace_mean: Optional[torch.Tensor]
    residual_trace_max: Optional[torch.Tensor]
    confidence_trace_mean: Optional[torch.Tensor]
    confidence_trace_min: Optional[torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int  # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False  # use mlp on L instead of transformer
    puzzle_emb_len: int = 16  # if non-zero, its specified to this value
    no_ACT_continue: bool = True  # No continue ACT loss, only use sigmoid(halt)

    # Residual computation / logging
    residual_enabled: bool = False
    residual_type: str = "logits_l2"  # logits_l2 | prob_kl_sym
    residual_temp: float = 1.0
    residual_trace_enabled: bool = False

    # Residual-based halting
    halt_residual_enabled: bool = False
    halt_residual_stat: str = "max"  # mean | max
    halt_residual_tau: float = 1e-3
    halt_residual_patience: int = 2
    halt_residual_min_steps: int = 2

    # Confidence gate
    halt_confidence_min: float = 0.0
    halt_confidence_stat: str = "min"  # mean | min
    halt_confidence_temp: float = 1.0

    # Explicit residual update / damping
    update_damping_enabled: bool = False
    update_damping_alpha_zL: float = 1.0
    update_damping_alpha_zH: float = 1.0


class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = (
                -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
                if self.config.puzzle_emb_len == 0
                else self.config.puzzle_emb_len
            )
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,  # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps,
            )
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        if not (0.0 < self.config.update_damping_alpha_zL <= 1.0):
            raise ValueError("update_damping_alpha_zL must be in (0, 1].")
        if not (0.0 < self.config.update_damping_alpha_zH <= 1.0):
            raise ValueError("update_damping_alpha_zH must be in (0, 1].")

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = (
            -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
            if self.config.puzzle_emb_len == 0
            else self.config.puzzle_emb_len
        )
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )

        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _ in range(self.config.L_layers)]
        )

        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True
        )

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2
            )

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(
                batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype
            ),
            z_L=torch.empty(
                batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def _apply_damped_update(self, old_state: torch.Tensor, proposed_state: torch.Tensor, alpha: float) -> torch.Tensor:
        if not self.config.update_damping_enabled:
            return proposed_state
        return old_state + alpha * (proposed_state - old_state)

    def forward(
        self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        z_H, z_L = carry.z_H, carry.z_L
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    zL_prop = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                    z_L = self._apply_damped_update(z_L, zL_prop, self.config.update_damping_alpha_zL)
                zH_prop = self.L_level(z_H, z_L, **seq_info)
                z_H = self._apply_damped_update(z_H, zH_prop, self.config.update_damping_alpha_zH)

        for _ in range(self.config.L_cycles):
            zL_prop = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_L = self._apply_damped_update(z_L, zL_prop, self.config.update_damping_alpha_zL)
        zH_prop = self.L_level(z_H, z_L, **seq_info)
        z_H = self._apply_damped_update(z_H, zH_prop, self.config.update_damping_alpha_zH)

        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

        if self.config.residual_type not in {"logits_l2", "prob_kl_sym"}:
            raise ValueError("residual_type must be one of: logits_l2, prob_kl_sym.")
        if self.config.halt_residual_stat not in {"mean", "max"}:
            raise ValueError("halt_residual_stat must be one of: mean, max.")
        if self.config.halt_confidence_stat not in {"mean", "min"}:
            raise ValueError("halt_confidence_stat must be one of: mean, min.")
        if self.config.residual_temp <= 0:
            raise ValueError("residual_temp must be > 0.")
        if self.config.halt_confidence_temp <= 0:
            raise ValueError("halt_confidence_temp must be > 0.")

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    @property
    def residual_enabled(self) -> bool:
        return self.config.residual_enabled or self.config.halt_residual_enabled or self.config.residual_trace_enabled

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        residual_trace_mean = None
        residual_trace_max = None
        confidence_trace_mean = None
        confidence_trace_min = None
        if self.config.residual_trace_enabled:
            shape = (batch_size, self.config.halt_max_steps)
            residual_trace_mean = torch.zeros(shape, dtype=torch.float32, device=device)
            residual_trace_max = torch.zeros(shape, dtype=torch.float32, device=device)
            confidence_trace_mean = torch.zeros(shape, dtype=torch.float32, device=device)
            confidence_trace_min = torch.zeros(shape, dtype=torch.float32, device=device)

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
            prev_logits=torch.zeros(
                (batch_size, self.config.seq_len, self.config.vocab_size), dtype=torch.float32, device=device
            ),
            prev_preds=torch.zeros((batch_size, self.config.seq_len), dtype=torch.int32, device=device),
            prev_seq_is_correct=torch.zeros((batch_size,), dtype=torch.bool, device=device),
            correct_streak=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            residual_below_count=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            residual_trace_mean=residual_trace_mean,
            residual_trace_max=residual_trace_max,
            confidence_trace_mean=confidence_trace_mean,
            confidence_trace_min=confidence_trace_min,
        )

    def _compute_residual_token_values(self, logits: torch.Tensor, prev_logits: torch.Tensor) -> torch.Tensor:
        logits_f32 = logits.to(torch.float32)
        prev_logits_f32 = prev_logits.to(torch.float32)
        if self.config.residual_type == "logits_l2":
            return (logits_f32 - prev_logits_f32).pow(2).sum(dim=-1)
        logits_t = logits_f32 / self.config.residual_temp
        prev_logits_t = prev_logits_f32 / self.config.residual_temp
        logp_t = F.log_softmax(logits_t, dim=-1)
        logp_prev = F.log_softmax(prev_logits_t, dim=-1)
        p_t = logp_t.exp()
        p_prev = logp_prev.exp()
        kl_prev_t = (p_prev * (logp_prev - logp_t)).sum(dim=-1)
        kl_t_prev = (p_t * (logp_t - logp_prev)).sum(dim=-1)
        return kl_prev_t + kl_t_prev

    def _write_trace(
        self, trace: torch.Tensor, reset_flag: torch.Tensor, step_indices: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        trace = torch.where(reset_flag.view(-1, 1), torch.zeros_like(trace), trace)
        return trace.scatter(1, step_indices, values.to(trace.dtype).unsqueeze(-1))

    def forward(
        self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        reset_flag = carry.halted

        new_inner_carry = self.inner.reset_carry(reset_flag, carry.inner_carry)
        new_steps = torch.where(reset_flag, torch.zeros_like(carry.steps), carry.steps)
        new_current_data = {
            k: torch.where(reset_flag.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)
        labels = new_current_data["labels"]
        valid_mask = labels != IGNORE_LABEL_ID

        with torch.no_grad():
            if self.residual_enabled:
                residual_tok = self._compute_residual_token_values(logits, carry.prev_logits)
                residual_mean = _masked_mean(residual_tok, valid_mask, empty_value=0.0)
                residual_max = _masked_max(residual_tok, valid_mask, empty_value=0.0)
                residual_mean = torch.where(reset_flag, torch.zeros_like(residual_mean), residual_mean)
                residual_max = torch.where(reset_flag, torch.zeros_like(residual_max), residual_max)
            else:
                residual_mean = torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)
                residual_max = torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)

            conf_tok = torch.softmax(logits.to(torch.float32) / self.config.halt_confidence_temp, dim=-1).amax(dim=-1)
            confidence_mean = _masked_mean(conf_tok, valid_mask, empty_value=1.0)
            confidence_min = _masked_min(conf_tok, valid_mask, empty_value=1.0)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "residual_mean": residual_mean,
            "residual_max": residual_max,
            "confidence_mean": confidence_mean,
            "confidence_min": confidence_min,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.config.halt_residual_enabled:
                r_halt = residual_max if self.config.halt_residual_stat == "max" else residual_mean
                below = (r_halt < self.config.halt_residual_tau) & (~reset_flag)
                residual_below_count = torch.where(
                    below, carry.residual_below_count + 1, torch.zeros_like(carry.residual_below_count)
                )
            else:
                residual_below_count = torch.where(
                    reset_flag, torch.zeros_like(carry.residual_below_count), carry.residual_below_count
                )

            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                if self.config.halt_residual_enabled:
                    c_halt = confidence_min if self.config.halt_confidence_stat == "min" else confidence_mean
                    halt_by_residual = (
                        (new_steps >= self.config.halt_residual_min_steps)
                        & (residual_below_count >= self.config.halt_residual_patience)
                        & (c_halt >= self.config.halt_confidence_min)
                    )
                    halted = halted | halt_by_residual

                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(
                    new_steps, low=2, high=self.config.halt_max_steps + 1
                )
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

            preds = torch.argmax(logits, dim=-1)
            is_correct_or_ignored = torch.where(valid_mask, preds == labels, torch.ones_like(valid_mask))
            seq_is_correct = is_correct_or_ignored.all(dim=-1)

            new_prev_logits = logits.detach().to(torch.float32)
            new_prev_preds = preds.to(torch.int32)
            new_prev_seq_is_correct = torch.where(reset_flag, torch.zeros_like(seq_is_correct), seq_is_correct)
            new_correct_streak = torch.where(
                reset_flag,
                torch.zeros_like(carry.correct_streak),
                torch.where(
                    seq_is_correct,
                    carry.correct_streak + 1,
                    torch.zeros_like(carry.correct_streak),
                ),
            )

            residual_trace_mean = carry.residual_trace_mean
            residual_trace_max = carry.residual_trace_max
            confidence_trace_mean = carry.confidence_trace_mean
            confidence_trace_min = carry.confidence_trace_min

            if self.config.residual_trace_enabled:
                assert residual_trace_mean is not None
                assert residual_trace_max is not None
                assert confidence_trace_mean is not None
                assert confidence_trace_min is not None
                trace_indices = (new_steps - 1).clamp(min=0, max=self.config.halt_max_steps - 1).to(torch.int64).unsqueeze(-1)

                residual_trace_mean = self._write_trace(residual_trace_mean, reset_flag, trace_indices, residual_mean)
                residual_trace_max = self._write_trace(residual_trace_max, reset_flag, trace_indices, residual_max)
                confidence_trace_mean = self._write_trace(
                    confidence_trace_mean, reset_flag, trace_indices, confidence_mean
                )
                confidence_trace_min = self._write_trace(confidence_trace_min, reset_flag, trace_indices, confidence_min)

                outputs["residual_trace_mean"] = residual_trace_mean
                outputs["residual_trace_max"] = residual_trace_max
                outputs["confidence_trace_mean"] = confidence_trace_mean
                outputs["confidence_trace_min"] = confidence_trace_min

        new_carry = TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=new_current_data,
            prev_logits=new_prev_logits,
            prev_preds=new_prev_preds,
            prev_seq_is_correct=new_prev_seq_is_correct,
            correct_streak=new_correct_streak,
            residual_below_count=residual_below_count,
            residual_trace_mean=residual_trace_mean,
            residual_trace_max=residual_trace_max,
            confidence_trace_mean=confidence_trace_mean,
            confidence_trace_min=confidence_trace_min,
        )
        return new_carry, outputs
