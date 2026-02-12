import argparse
import os
import sys
from typing import Dict

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.losses import ACTLossHead
from models.recursive_reasoning.vrm import TinyRecursiveReasoningModel_ACTV1


def build_model_config(batch_size: int, seq_len: int, forward_dtype: str) -> Dict[str, object]:
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "puzzle_emb_ndim": 128,
        "num_puzzle_identifiers": 1,
        "vocab_size": 32,
        "H_cycles": 2,
        "L_cycles": 2,
        "H_layers": 0,
        "L_layers": 1,
        "hidden_size": 128,
        "expansion": 4,
        "num_heads": 8,
        "pos_encodings": "none",
        "halt_max_steps": 1,
        "halt_exploration_prob": 0.0,
        "forward_dtype": forward_dtype,
        "mlp_t": False,
        "axial_t": True,
        "axial_2dconv": True,
        "axial_2dconv_kernel": 3,
        "axial_2dconv_padding": "same",
        "axial_prefix_coupling": "pool",
        "axial_hw": None,
        "puzzle_emb_len": 16,
        "no_ACT_continue": True,
    }


def run_smoke(seq_len: int, batch_size: int, device: torch.device, use_compile: bool) -> None:
    forward_dtype = "bfloat16" if device.type == "cuda" else "float32"
    model = TinyRecursiveReasoningModel_ACTV1(build_model_config(batch_size, seq_len, forward_dtype)).to(device)
    model_with_loss = ACTLossHead(model, loss_type="stablemax_cross_entropy").to(device)

    if use_compile:
        model_with_loss = torch.compile(model_with_loss)

    model_with_loss.train()
    optimizer = torch.optim.AdamW(model_with_loss.parameters(), lr=1e-3)

    batch = {
        "inputs": torch.randint(0, 32, (batch_size, seq_len), device=device, dtype=torch.int32),
        "labels": torch.randint(0, 32, (batch_size, seq_len), device=device, dtype=torch.int32),
        "puzzle_identifiers": torch.zeros(batch_size, device=device, dtype=torch.int32),
    }

    optimizer.zero_grad(set_to_none=True)
    carry = model_with_loss.initial_carry(batch)
    _, loss, _, _, _ = model_with_loss(carry=carry, batch=batch, return_keys=[])
    loss.backward()
    optimizer.step()

    print(f"seq_len={seq_len}: loss={loss.detach().item():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TRM axial forward/backward smoke test.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device={device}")
    for seq_len in (81, 900):
        run_smoke(seq_len=seq_len, batch_size=args.batch_size, device=device, use_compile=args.compile)

    print("smoke test passed")


if __name__ == "__main__":
    main()
