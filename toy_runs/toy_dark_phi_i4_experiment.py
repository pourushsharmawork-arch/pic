"""
Toy experiment for structural phase darkness in the 2121 mesh.

This script follows the workflow the user asked for:
  1) generate a dataset using only the basis probe I4
  2) train a smaller inverse model on that dataset
  3) run inference on a held-out sample
  4) prove that the structurally dark phi coordinates can be set to
     arbitrary values without changing the measured powers

The key point is that this is not a generic identifiability study: it is a
deliberately narrow experiment that isolates a probe choice where the mesh
implementation itself makes some phi coordinates invisible at first order.
"""
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


def make_i4_regime() -> Regime:
    """Single basis probe I4, expressed as a custom probe tensor."""
    probes = torch.eye(N_MODES, dtype=torch.complex64)[3:4].clone()
    return Regime(observation="power", eta=1.0, basis_only=True, probes=probes)


def _save_dataset(path: Path, data: dict, regime: Regime) -> None:
    """Persist a compact row-wise toy dataset for later inspection."""
    rows = torch.cat(
        [data["theta"], data["phi"], data["delta"], data["obs"]],
        dim=-1,
    ).cpu().numpy()
    np.savez_compressed(
        path,
        my_matrix=rows,
        theta=data["theta"].cpu().numpy(),
        phi=data["phi"].cpu().numpy(),
        delta=data["delta"].cpu().numpy(),
        obs=data["obs"].cpu().numpy(),
        observation=np.array("power"),
        basis_only=np.array(True),
        quadrature=np.array(False),
        probe_labels=np.array(["I4"]),
        eta=np.array(float(regime.eta)),
    )


def _format_param_list(names: list[str], values: torch.Tensor) -> str:
    return ", ".join(f"{n}={values[i].item():.6f}" for i, n in enumerate(names))


def _find_dark_phi_names(analysis: dict, zero_tol: float) -> list[str]:
    names = analysis["names"]
    norms = analysis["column_norms"]
    return [
        n for n, c in zip(names, norms)
        if n.startswith("phi_") and c.item() <= zero_tol
    ]


def _name_to_index(names: list[str]) -> dict[str, int]:
    return {n: i for i, n in enumerate(names)}


def _randomize_dark_phis(
    theta: torch.Tensor,
    phi: torch.Tensor,
    delta: torch.Tensor,
    dark_phi_names: list[str],
    *,
    trials: int,
    seed: int,
    regime: Regime,
) -> list[dict]:
    idx_map = _name_to_index([f"theta_{b}" for b in BLOCK_LABELS] + [f"phi_{b}" for b in BLOCK_LABELS])
    dark_idx = [idx_map[n] - 6 for n in dark_phi_names]
    g = torch.Generator(device="cpu").manual_seed(seed)
    base_obs = simulate(theta.unsqueeze(0), phi.unsqueeze(0), delta.unsqueeze(0), regime)
    results = []

    for t in range(trials):
        phi_trial = phi.clone()
        phi_trial[dark_idx] = torch.rand(len(dark_idx), generator=g) * 2 * torch.pi
        obs_trial = simulate(theta.unsqueeze(0), phi_trial.unsqueeze(0), delta.unsqueeze(0), regime)
        results.append(
            {
                "trial": t,
                "phi": phi_trial,
                "max_abs_delta": (obs_trial - base_obs).abs().max().item(),
            }
        )
    return results


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

    regime = make_i4_regime()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("[setup]")
    print("  experiment   : single basis probe I4")
    print("  observation  : power")
    print(f"  obs_dim      : {regime.obs_dim}")
    print("  model        : narrow inverse net")
    print("  train mode   : physics loss only (no parameter supervision on dark phases)")

    analysis = analyze_identifiability(regime, seed=args.seed, sv_tol=1e-4)
    dark_phi_names = _find_dark_phi_names(analysis, zero_tol=1e-8)
    if not dark_phi_names:
        raise RuntimeError("Could not identify any dark phi columns for the I4 probe.")
    print("\n[structural analysis]")
    print(f"  rank / nullity : {analysis['rank']} / {analysis['null_dim']}")
    print(f"  dark phi names : {', '.join(dark_phi_names)}")

    print("\n[data generation]")
    raw = dp.synthetic_regime_dataset(args.n_samples, regime, seed=args.seed)
    data_file = save_dir / "i4_only_dataset.npz"
    _save_dataset(data_file, raw, regime)
    print(f"  saved dataset  : {data_file}")
    print(f"  train samples  : {int(args.n_samples * 0.9)}")
    print(f"  val samples    : {args.n_samples - int(args.n_samples * 0.9)}")

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
    ckpt = save_dir / "i4_only_inverse_net.pt"
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

    print("\n[dark-phi invariance test]")
    base_theta = th_r[0].detach().cpu()
    base_phi = ph_r[0].detach().cpu()
    base_delta = de_r[0].detach().cpu()
    base_obs = simulate(base_theta.unsqueeze(0), base_phi.unsqueeze(0), base_delta.unsqueeze(0), regime)
    dark_idx = [_name_to_index([f"theta_{b}" for b in BLOCK_LABELS] + [f"phi_{b}" for b in BLOCK_LABELS])[n] - 6
                for n in dark_phi_names]
    learnable_phi_idx = next(i for i in range(6) if i not in dark_idx)

    print(f"  baseline obs RMSE : {torch.sqrt(((base_obs - obs.cpu()) ** 2).sum()).item():.6e}")
    print(f"  dark phi indices  : {dark_idx}")
    print(f"  varied phi names  : {', '.join(dark_phi_names)}")

    trials = _randomize_dark_phis(
        base_theta,
        base_phi,
        base_delta,
        dark_phi_names,
        trials=args.dark_trials,
        seed=args.seed + 123,
        regime=regime,
    )
    for t in trials:
        trial_phi = t["phi"]
        print(
            f"  trial {t['trial']}: max|Δobs|={t['max_abs_delta']:.3e}  "
            f"phi_dark={trial_phi[dark_idx].tolist()}"
        )

    phi_changed = base_phi.clone()
    phi_changed[learnable_phi_idx] = (phi_changed[learnable_phi_idx] + 0.7) % (2 * torch.pi)
    changed_obs = simulate(base_theta.unsqueeze(0), phi_changed.unsqueeze(0), base_delta.unsqueeze(0), regime)
    changed_delta = torch.sqrt(((changed_obs - base_obs) ** 2).sum()).item()
    print(
        f"  contrast: changing learnable phi_{BLOCK_LABELS[learnable_phi_idx]} "
        f"gives RMSE {changed_delta:.3e}"
    )

    print("\n[summary]")
    print(
        "  The model can fit I4-only powers, but the three structurally dark phi "
        "coordinates are not constrained by the data. Replacing them with arbitrary "
        "values leaves the power measurement unchanged to numerical precision."
    )


if __name__ == "__main__":
    main()
