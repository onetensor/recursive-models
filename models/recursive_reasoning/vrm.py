from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


@torch._dynamo.disable
def _axial_2dconv_residual_eager(
    grid: torch.Tensor,
    depthwise_conv2d: nn.Module,
    norm_eps: float,
) -> torch.Tensor:
    grid_channels_first = grid.permute(0, 3, 1, 2)
    grid_channels_first = grid_channels_first + depthwise_conv2d(grid_channels_first)
    grid = grid_channels_first.permute(0, 2, 3, 1)
    grid = rms_norm(grid, variance_epsilon=norm_eps)
    return grid

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


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
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
    mlp_t: bool = False # use mlp on L instead of transformer
    axial_t: bool = False
    axial_2dconv: bool = False
    axial_2dconv_kernel: int = 3
    axial_2dconv_padding: str = "same"
    axial_prefix_coupling: str = "pool"
    axial_hw: Optional[Tuple[int, int]] = None
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

    def model_post_init(self, __context) -> None:
        if self.mlp_t and self.axial_t:
            raise ValueError("`mlp_t` and `axial_t` are mutually exclusive. Set only one of them to True.")

        if not self.axial_t:
            return

        if self.axial_hw is None:
            inferred = math.isqrt(self.seq_len)
            if inferred * inferred != self.seq_len:
                raise ValueError(
                    "`axial_t=True` with `axial_hw=None` requires `seq_len` to be a perfect square "
                    f"for grid reshape. Got seq_len={self.seq_len}. "
                    "Set `axial_hw=(H, W)` explicitly for non-square layouts."
                )
        else:
            h, w = self.axial_hw
            if h <= 0 or w <= 0:
                raise ValueError(f"`axial_hw` values must be > 0, got {self.axial_hw}.")
            if h * w != self.seq_len:
                raise ValueError(
                    f"`axial_hw={self.axial_hw}` is incompatible with seq_len={self.seq_len}; "
                    "expected H*W == seq_len."
                )


class CastedDepthwiseConv2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, padding: int) -> None:
        super().__init__()
        self.channels = channels
        self.padding = padding
        self.weight = nn.Parameter(torch.zeros(channels, 1, kernel_size, kernel_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            inputs,
            self.weight.to(inputs.dtype),
            bias=None,
            stride=1,
            padding=self.padding,
            groups=self.channels,
        )


class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t and self.config.axial_t:
            raise ValueError("`mlp_t` and `axial_t` are mutually exclusive. Set only one of them to True.")

        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        elif self.config.axial_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.axial_h, self.axial_w = self._resolve_axial_hw()

            self.row_mlp_t = SwiGLU(
                hidden_size=self.axial_w,
                expansion=config.expansion,
            )
            self.col_mlp_t = SwiGLU(
                hidden_size=self.axial_h,
                expansion=config.expansion,
            )

            if self.config.axial_2dconv:
                if self.config.axial_2dconv_kernel % 2 == 0:
                    raise ValueError(
                        f"`axial_2dconv_kernel` must be odd, got {self.config.axial_2dconv_kernel}."
                    )
                if self.config.axial_2dconv_padding != "same":
                    raise ValueError(
                        f"Only `axial_2dconv_padding=\"same\"` is supported, got {self.config.axial_2dconv_padding}."
                    )
                self.axial_2dconv = CastedDepthwiseConv2d(
                    channels=config.hidden_size,
                    kernel_size=self.config.axial_2dconv_kernel,
                    padding=self.config.axial_2dconv_kernel // 2,
                )

            if self.config.axial_prefix_coupling != "pool":
                raise ValueError(
                    f"Only `axial_prefix_coupling=\"pool\"` is supported, got {self.config.axial_prefix_coupling}."
                )
            self.linear_grid_to_prefix = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
            self.linear_prefix_to_grid = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def _resolve_axial_hw(self) -> Tuple[int, int]:
        if self.config.axial_hw is None:
            inferred = math.isqrt(self.config.seq_len)
            if inferred * inferred != self.config.seq_len:
                raise ValueError(
                    "`axial_t=True` with `axial_hw=None` requires `seq_len` to be a perfect square "
                    f"for grid reshape. Got seq_len={self.config.seq_len}. "
                    "Set `axial_hw=(H, W)` explicitly for non-square layouts."
                )
            return inferred, inferred

        h, w = self.config.axial_hw
        if h <= 0 or w <= 0:
            raise ValueError(f"`axial_hw` values must be > 0, got {self.config.axial_hw}.")
        if h * w != self.config.seq_len:
            raise ValueError(
                f"`axial_hw={self.config.axial_hw}` is incompatible with seq_len={self.config.seq_len}; "
                "expected H*W == seq_len."
            )
        return h, w

    def _forward_axial(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = hidden_states.shape
        grid = hidden_states[:, self.puzzle_emb_len :, :].reshape(batch_size, self.axial_h, self.axial_w, hidden_size)

        if hasattr(self, "axial_2dconv"):
            grid = _axial_2dconv_residual_eager(
                grid=grid,
                depthwise_conv2d=self.axial_2dconv,
                norm_eps=self.norm_eps,
            )

        row_mix = grid.permute(0, 1, 3, 2)
        row_mix = row_mix + self.row_mlp_t(row_mix)
        grid = row_mix.permute(0, 1, 3, 2)
        grid = rms_norm(grid, variance_epsilon=self.norm_eps)

        col_mix = grid.permute(0, 2, 3, 1)
        col_mix = col_mix + self.col_mlp_t(col_mix)
        grid = col_mix.permute(0, 3, 1, 2)
        grid = rms_norm(grid, variance_epsilon=self.norm_eps)

        grid = grid.reshape(batch_size, self.config.seq_len, hidden_size)

        grid_summary = grid.mean(dim=1)
        prefix0 = hidden_states[:, 0, :] + self.linear_grid_to_prefix(grid_summary)
        grid = grid + self.linear_prefix_to_grid(prefix0).unsqueeze(1)

        if self.puzzle_emb_len > 0:
            if self.puzzle_emb_len > 1:
                prefix_tokens = torch.cat((prefix0.unsqueeze(1), hidden_states[:, 1 : self.puzzle_emb_len, :]), dim=1)
            else:
                prefix_tokens = prefix0.unsqueeze(1)
            hidden_states = torch.cat((prefix_tokens, grid), dim=1)
        else:
            hidden_states = torch.cat((prefix0.unsqueeze(1), grid[:, 1:, :]), dim=1)

        return hidden_states

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        elif self.config.axial_t:
            hidden_states = self._forward_axial(hidden_states)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
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

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
