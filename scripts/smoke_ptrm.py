import argparse
import os
import sys
from typing import Dict

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.losses import ACTLossHead
from models.recursive_reasoning.ptrm import PerceiverRecursiveReasoningModel_ACTV1


def build_model_config(batch_size: int, seq_len: int, vocab_size: int, z_slots: int, forward_dtype: str) -> Dict[str, object]:
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "puzzle_emb_ndim": 128,
        "num_puzzle_identifiers": 8,
        "vocab_size": vocab_size,
        "H_cycles": 3,
        "L_cycles": 2,
        "H_layers": 0,
        "L_layers": 1,
        "hidden_size": 128,
        "expansion": 4,
        "num_heads": 8,
        "pos_encodings": "rope",
        "halt_max_steps": 4,
        "halt_exploration_prob": 0.0,
        "forward_dtype": forward_dtype,
        "mlp_t": False,
        "puzzle_emb_len": 16,
        "no_ACT_continue": True,
        "z_slots": z_slots,
    }


def run_smoke(batch_size: int, seq_len: int, vocab_size: int, z_slots: int, device: torch.device) -> None:
    torch.manual_seed(0)
    forward_dtype = "bfloat16" if device.type == "cuda" else "float32"

    cfg = build_model_config(batch_size, seq_len, vocab_size, z_slots, forward_dtype)
    model = PerceiverRecursiveReasoningModel_ACTV1(cfg).to(device)
    model.train()

    batch = {
        "inputs": torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.int32),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.int32),
        "puzzle_identifiers": torch.randint(0, cfg["num_puzzle_identifiers"], (batch_size,), device=device, dtype=torch.int32),
    }

    # 1) Import + forward shape checks
    carry = model.initial_carry(batch)
    carry2, out = model(carry, batch)

    l_total = seq_len + int(cfg["puzzle_emb_len"])
    hidden_size = int(cfg["hidden_size"])

    assert out["logits"].shape == (batch_size, seq_len, vocab_size), out["logits"].shape
    assert out["q_halt_logits"].shape == (batch_size,), out["q_halt_logits"].shape
    assert out["q_continue_logits"].shape == (batch_size,), out["q_continue_logits"].shape
    assert carry2.inner_carry.z_H.shape == (batch_size, l_total, hidden_size), carry2.inner_carry.z_H.shape
    assert carry2.inner_carry.z_L.shape == (batch_size, z_slots, hidden_size), carry2.inner_carry.z_L.shape
    assert not carry2.inner_carry.z_H.requires_grad
    assert not carry2.inner_carry.z_L.requires_grad

    # 2) Backward test via ACTLossHead
    model_with_loss = ACTLossHead(model, loss_type="stablemax_cross_entropy").to(device)
    optimizer = torch.optim.AdamW(model_with_loss.parameters(), lr=1e-3)

    optimizer.zero_grad(set_to_none=True)
    carry = model_with_loss.initial_carry(batch)
    _, loss, _, _, _ = model_with_loss(carry=carry, batch=batch, return_keys=[])
    loss.backward()

    finite_grad_params = 0
    for param in model_with_loss.parameters():
        if param.grad is None:
            continue
        finite_grad_params += 1
        assert torch.isfinite(param.grad).all(), "Found non-finite gradients."
    assert finite_grad_params > 0, "No gradients were produced."

    optimizer.step()
    print(
        "PTRM smoke passed: "
        f"logits={tuple(out['logits'].shape)}, "
        f"z_H={tuple(carry2.inner_carry.z_H.shape)}, "
        f"z_L={tuple(carry2.inner_carry.z_L.shape)}, "
        f"loss={loss.detach().item():.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PTRM import/forward/backward smoke tests.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=81)
    parser.add_argument("--vocab-size", type=int, default=12)
    parser.add_argument("--z-slots", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_smoke(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        z_slots=args.z_slots,
        device=device,
    )


if __name__ == "__main__":
    main()
