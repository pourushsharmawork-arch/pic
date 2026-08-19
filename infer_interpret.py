"""
Inference + lightweight interpretability for PARAMS inverse mesh models.

Supports all regime.py settings:
  --observation exact | power
  --eta           known loss (1.0 = lossless)
  --quadrature    power mode: 12 probes (obs dim 48)
  --basis_only    power mode: I1..I4 only (obs dim 16)

Given one observation vector, this script:
  1) loads a checkpoint (InverseNet weights)
  2) predicts mesh params
  3) refines on the tandem physics loss (multi-start by default)
  4) reports observation fidelity
  5) optional Jacobian-based feature sensitivity

Examples:
    # exact field, lossless (32-D Re/Im, 16 angles)
    python infer_interpret.py --checkpoint inverse_net.pt --data data/exact_lossless.npz

    # power, lossy, with quadrature probes (48-D powers, 12 angles predicted)
    python infer_interpret.py --observation power --eta 0.85 --quadrature \\
        --checkpoint inverse_net.pt --data data/power_quad_lossy.npz --index 0

    # raw observation vector (comma-separated; length must match regime)
    python infer_interpret.py --observation power --features "0.1,0.2,..." --quadrature
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

import data_pipeline as dp
from mesh_forward import probe_labels
from regime import (
    Regime, decode, make_model, refine_multistart_regime, simulate,
)
from train_tandem import DEVICE, measurement_to_U, reconstruction_fidelity, normalized_overlap_fidelity


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def obs_fidelity(pred_obs: torch.Tensor, true_obs: torch.Tensor) -> torch.Tensor:
    num = (pred_obs * true_obs).sum(dim=-1) ** 2
    den = (pred_obs ** 2).sum(dim=-1) * (true_obs ** 2).sum(dim=-1)
    return num / den.clamp_min(1e-12)


def _parse_features(features: str, obs_dim: int) -> np.ndarray:
    vals = np.array([float(v.strip()) for v in features.split(",")], dtype=np.float32)
    if vals.shape[0] != obs_dim:
        raise ValueError(
            f"--features must contain exactly {obs_dim} comma-separated values "
            f"for this regime, got {vals.shape[0]}"
        )
    return vals


def _load_row(data_path: str, index: int, npz_key: str | None,
              regime: Regime) -> tuple[np.ndarray, np.ndarray | None]:
    arr = dp._load_array(data_path, npz_key=npz_key)
    row_width = 16 + regime.obs_dim
    if arr.ndim != 2 or arr.shape[1] not in (48, 64) and arr.shape[0] not in (48, 64):
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D array, got shape {arr.shape}")
    # normalize to (N, width)
    if arr.shape[0] in (48, 64) and arr.shape[1] not in (48, 64):
        arr = arr.T
    if arr.shape[1] != row_width and not (arr.shape[0] == row_width and arr.shape[1] != row_width):
        if arr.shape[0] in (32, 48, 64) and arr.shape[1] not in (32, 48, 64):
            arr = arr.T
    if arr.shape[1] != row_width:
        raise ValueError(
            f"Expected dataset width {row_width} for observation={regime.observation} "
            f"quadrature={regime.quadrature}, got shape {arr.shape}"
        )
    if not (0 <= index < arr.shape[0]):
        raise IndexError(f"--index {index} out of range for dataset with {arr.shape[0]} rows")
    row = arr[index].astype(np.float32)
    return row[16:16 + regime.obs_dim], row[:16]


def _format_vec(name: str, vec: np.ndarray):
    print(f"{name:>8}: {np.array2string(vec, precision=6, suppress_small=True)}")


def _interpretability_report(model, obs: torch.Tensor, regime: Regime, topk: int = 8):
    """Jacobian-based local sensitivity: d angle_k / d obs_j."""
    obs = obs.detach().clone().requires_grad_(True)
    pred_sc = model(obs)
    theta, phi, delta = decode(pred_sc, regime)
    if regime.predict_delta:
        angles = torch.cat([theta, phi, delta], dim=-1)
    else:
        angles = torch.cat([theta, phi], dim=-1)

    n_angles = angles.shape[-1]
    obs_dim = regime.obs_dim
    grads = []
    for k in range(n_angles):
        g = torch.autograd.grad(angles[0, k], obs, retain_graph=True)[0]
        grads.append(g[0].abs())
    J_abs = torch.stack(grads, dim=0)
    feat_imp = J_abs.mean(dim=0).detach().cpu().numpy()

    print("\n[interpretability] local sensitivity at this sample")
    if regime.observation == "exact":
        probe_imp = np.array([feat_imp[i * 8:(i + 1) * 8].sum() for i in range(4)])
        probe_imp = probe_imp / (probe_imp.sum() + 1e-12)
        print("  basis-probe contribution (normalized):")
        for p in range(4):
            print(f"    probe e{p}: {probe_imp[p]:.3f}")
        print(f"  top-{topk} sensitive feature indices:")
        idx = np.argsort(-feat_imp)[:topk]
        for j in idx:
            probe = j // 8
            within = j % 8
            part = "Re" if within < 4 else "Im"
            mode = within if within < 4 else within - 4
            print(f"    obs[{j:2d}]  ({part} mode {mode}, probe e{probe})  score={feat_imp[j]:.4e}")
    else:
        n_probes = regime.probes.shape[0]
        labels = probe_labels(regime.quadrature, regime.basis_only)
        probe_imp = np.array([feat_imp[i * 4:(i + 1) * 4].sum() for i in range(n_probes)])
        probe_imp = probe_imp / (probe_imp.sum() + 1e-12)
        print("  probe contribution (normalized):")
        for i, label in enumerate(labels):
            print(f"    {label:>5}: {probe_imp[i]:.3f}")
        print(f"  top-{topk} sensitive power indices:")
        idx = np.argsort(-feat_imp)[:topk]
        for j in idx:
            probe_i = j // 4
            port = j % 4
            print(f"    obs[{j:2d}]  ({labels[probe_i]} port {port})  score={feat_imp[j]:.4e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="inverse_net.pt")
    p.add_argument("--data", type=str, default=None,
                   help="dataset path (.npz/.npy/.csv/.pt); row width 48 or 64 for power")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--npz_key", type=str, default=None)
    p.add_argument("--features", type=str, default=None,
                   help="comma-separated observation (overrides --data)")
    p.add_argument("--observation", type=str, default="exact", choices=["exact", "power"])
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--basis_only", action="store_true",
                   help="power mode: I1..I4 only (obs dim 16)")
    p.add_argument("--quadrature", action="store_true",
                   help="power mode: 12 probes, 48-D observation")
    p.add_argument("--device", type=str, default=DEVICE, choices=["cpu", "cuda"])
    p.add_argument("--refine_steps", type=int, default=150)
    p.add_argument("--refine_lr", type=float, default=0.03)
    p.add_argument("--n_starts", type=int, default=8)
    p.add_argument("--topk", type=int, default=8)
    args = p.parse_args()

    if args.features is None and args.data is None:
        raise ValueError("Provide either --features or --data")
    if args.basis_only and args.quadrature:
        p.error("--basis_only and --quadrature are incompatible")
    if args.basis_only and args.observation != "power":
        p.error("--basis_only applies only to --observation power")
    if args.quadrature and args.observation != "power":
        p.error("--quadrature applies only to --observation power")

    regime = Regime(
        observation=args.observation, eta=args.eta,
        quadrature=args.quadrature, basis_only=args.basis_only,
    )

    if args.features is not None:
        obs_np = _parse_features(args.features, regime.obs_dim)
        true_params = None
    else:
        if args.observation == "power" and args.data is not None:
            raw = dp.load_real_power_dataset(
                args.data, quadrature=args.quadrature, basis_only=args.basis_only,
            )
            if raw["quadrature"] != args.quadrature or raw["basis_only"] != args.basis_only:
                regime = Regime(
                    observation="power", eta=args.eta,
                    quadrature=raw["quadrature"], basis_only=raw["basis_only"],
                )
            row = dp._load_array(args.data, npz_key=args.npz_key)
            if row.shape[0] in (48, 64):
                row = row.T
            if not (0 <= args.index < row.shape[0]):
                raise IndexError(f"--index {args.index} out of range")
            r = row[args.index].astype(np.float32)
            obs_np = r[16:16 + regime.obs_dim]
            true_params = r[:16]
        else:
            obs_np, true_params = _load_row(args.data, args.index, args.npz_key, regime)

    obs = torch.from_numpy(obs_np).float().unsqueeze(0).to(args.device)

    model = make_model(regime).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        pred_sc = model(obs)
        th0, ph0, de0 = decode(pred_sc, regime)
        pred_obs0 = simulate(th0, ph0, de0, regime)
        obs_fid0 = obs_fidelity(pred_obs0, obs).item()
        if regime.observation == "exact":
            from mesh_forward import build_mesh_lossy
            U_target = measurement_to_U(obs).to(args.device)
            U_built = build_mesh_lossy(th0, ph0, de0, regime.eta)
            fid_fn = (
                normalized_overlap_fidelity
                if float(regime.eta) != 1.0
                else reconstruction_fidelity
            )
            fid0 = fid_fn(U_built, U_target).item()
            err0 = torch.linalg.matrix_norm(U_built - U_target, ord="fro").item()
        else:
            fid0 = obs_fid0
            err0 = torch.sqrt(((pred_obs0 - obs) ** 2).sum()).item()

    th, ph, de, err = refine_multistart_regime(
        model, obs, regime,
        n_starts=args.n_starts,
        n_steps=args.refine_steps,
        lr=args.refine_lr,
    )

    with torch.no_grad():
        pred_obs_r = simulate(th, ph, de, regime)
        obs_fid_r = obs_fidelity(pred_obs_r, obs).item()
        if regime.observation == "exact":
            from mesh_forward import build_mesh_lossy
            U_target = measurement_to_U(obs).to(args.device)
            U_built = build_mesh_lossy(th, ph, de, regime.eta)
            fid_fn = (
                normalized_overlap_fidelity
                if float(regime.eta) != 1.0
                else reconstruction_fidelity
            )
            fid_r = fid_fn(U_built, U_target).item()
            err_r = torch.linalg.matrix_norm(U_built - U_target, ord="fro").item()
        else:
            fid_r = obs_fid_r
            err_r = torch.sqrt(err).item()

    pred_params0 = torch.cat([th0[0], ph0[0]] + ([de0[0]] if regime.predict_delta else [])).detach().cpu().numpy()
    pred_params = torch.cat([th[0], ph[0]] + ([de[0]] if regime.predict_delta else [])).detach().cpu().numpy()
    sin_thetas = np.concatenate([np.sin(th[0]), np.cos(th[0])])
    print("[inference]")
    print(f"  regime                  : observation={regime.observation}  eta={regime.eta}  "
          f"basis_only={regime.basis_only}  quadrature={regime.quadrature}  obs_dim={regime.obs_dim}")
    print(f"  device                  : {args.device}")
    if regime.observation == "exact":
        print(f"  network matrix fidelity : {fid0:.6f}")
        print(f"  network frob error      : {err0:.6f}")
        print(f"  network obs fidelity    : {obs_fid0:.6f}")
        print(f"  refined matrix fidelity : {fid_r:.6f}")
        print(f"  refined frob error      : {err_r:.6f}")
        print(f"  refined obs fidelity    : {obs_fid_r:.6f}")
    else:
        print(f"  network obs fidelity    : {fid0:.6f}")
        print(f"  network obs RMSE        : {err0:.6f}")
        print(f"  refined obs fidelity    : {fid_r:.6f}")
        print(f"  refined obs RMSE        : {err_r:.6f}")

    _format_vec("theta", pred_params[:6])
    _format_vec("phi", pred_params[6:12])
    _format_vec("cos, sin theta", sin_thetas)
    if regime.predict_delta:
        _format_vec("delta", pred_params[12:16])
    else:
        print("   delta: (not predicted in power mode)")

    if true_params is not None:
        n_pred = pred_params.shape[0]
        rmse0 = float(np.sqrt(np.mean(_wrap_to_pi(pred_params0 - true_params[:n_pred]) ** 2)))
        rmse = float(np.sqrt(np.mean(_wrap_to_pi(pred_params - true_params[:n_pred]) ** 2)))
        print("\n[vs dataset labels — diagnostic only; high obs fidelity matters more]")
        print(f"  circular RMSE (network) : {rmse0:.6f}")
        print(f"  circular RMSE (refined) : {rmse:.6f}")

    _interpretability_report(model, obs, regime, topk=args.topk)


if __name__ == "__main__":
    main()
