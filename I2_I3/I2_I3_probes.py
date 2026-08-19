from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

import data_pipeline as dp
from identifiability import analyze_identifiability
from mesh_forward import BLOCK_LABELS, N_MODES
from regime import Regime, decode, refine_multistart_regime, simulate, train_model_regime
from train_tandem import DEVICE

probes = torch.eye(N_MODES, dtype=torch.complex64)[1:3].clone()
I2_I3_regime = Regime(observation='power', eta=1.0, basis_only=True, probes=probes)

def _format_param_list(names: list[str], values: torch.Tensor) -> str:
    return ", ".join(f"{n}={values[i].item():.6f}" for i, n in enumerate(names))

def _load_i2_i3_dataset(path: str) -> dict[str, torch.Tensor]:
    """Load rows stored as theta, phi, delta, I2 powers, and I3 powers."""
    rows = np.asarray(np.load(path)["my_matrix"], dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 24:
        raise ValueError(f"Expected an (N, 24) I2/I3 dataset, got {rows.shape}")
    return {
        "theta": torch.from_numpy(rows[:, 0:6]),
        "phi": torch.from_numpy(rows[:, 6:12]),
        "delta": torch.from_numpy(rows[:, 12:16]),
        "obs": torch.from_numpy(rows[:, 16:24]),
    }

def main():
    p = argparse.ArgumentParser(description="Toy I4-only darkness experiment")
    p.add_argument("--n_samples", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--n_blocks", type=int, default=2)
    p.add_argument("--aux_weight", type=float, default=0.0,
                   help="set to 0.0 to avoid supervising the dark phases")
    p.add_argument("--reg_weight", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--refine_steps", type=int, default=120)
    p.add_argument("--refine_lr", type=float, default=0.03)
    p.add_argument("--n_starts", type=int, default=6)
    p.add_argument("--dark_trials", type=int, default=5)
    p.add_argument("--save_dir", type=str, default="toy_runs/i4_dark_phi")
    args = p.parse_args()

    regime = I2_I3_regime
    analysis = analyze_identifiability(I2_I3_regime, sv_tol=1e-4)
    print(analysis)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("[setup]")
    print("  experiment   : I2 + I3 probes")
    print("  observation  : power")
    print(f"  obs_dim      : {regime.obs_dim}")
    print("  model        : narrow inverse net")
    print("  train mode   : physics loss only (no parameter supervision on dark phases)")
    raw = _load_i2_i3_dataset('NEW/training_power_data_I2_I3.npz')
    train_d, val_d = dp.regime_train_val_split(raw, val_frac=0.1, seed=args.seed)

    print("\n[training]")
    model, history = train_model_regime(
        train_d,
        val_d,
        regime,
        mode="physics",
        epochs=args.epochs,
        batch_size=args.batch_size,
        aux_weight=args.aux_weight,
        reg_weight=args.reg_weight,
        seed=args.seed,
        verbose=True,
        model_kwargs={"width": args.width, "n_blocks": args.n_blocks},
    )
    ckpt = save_dir / "I2_I3_net.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"  saved model    : {ckpt}")
    print(f"  best val MSE   : {history['best_val_obs_mse']:.6f}")

    print("\n[inference]")
    obs = val_d["obs"][0:1].to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred_sc = model(obs)
        th0, ph0, de0 = decode(pred_sc, regime)
        pred_obs0 = simulate(th0, ph0, de0, regime)
        obs_err0 = torch.sqrt(((pred_obs0 - obs) ** 2).sum(dim=-1)).item()
    print(f"  network obs RMSE  : {obs_err0:.6e}")
    print(f"  theta             : {_format_param_list([f'theta_{b}' for b in BLOCK_LABELS], th0[0].cpu())}")
    print(f"  phi               : {_format_param_list([f'phi_{b}' for b in BLOCK_LABELS], ph0[0].cpu())}")

    th_r, ph_r, de_r, err_r = refine_multistart_regime(
        model,
        obs.cpu(),
        regime,
        n_starts=args.n_starts,
        n_steps=args.refine_steps,
        lr=args.refine_lr,
        seed=args.seed,
    )
    with torch.no_grad():
        pred_obs_r = simulate(th_r, ph_r, de_r, regime)
        obs_err_r = torch.sqrt(((pred_obs_r.cpu() - obs.cpu()) ** 2).sum(dim=-1)).item()
    print(f"  refined obs RMSE  : {obs_err_r:.6e}")

if __name__ == "__main__":
    main()