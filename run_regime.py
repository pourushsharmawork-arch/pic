"""
End-to-end driver for the unified (regime.py) pipeline. Covers all four
regimes: exact-field / power-only × lossless (eta=1) / lossy (eta<1).

Usage:
    python run_regime.py --observation exact --eta 1.0 --epochs 30          # synthetic
    python run_regime.py --data data/exact_lossless.npz --eta 1.0
    python run_regime.py --data data/power_lossy.npz --observation power --eta 0.85
    python run_regime.py --data data/power_quad_lossy.npz --observation power --eta 0.85 --quadrature
"""
from __future__ import annotations

import argparse

import torch

import data_pipeline as dp
import verify_convention as vc
from regime import (
    Regime, train_model_regime, refine_regime, refine_multistart_regime, simulate,
)
from train_tandem import DEVICE
from verify_convention import EXPECTED_EXACT, EXPECTED_POWER


def obs_fidelity(pred_obs, true_obs):
    num = (pred_obs * true_obs).sum(dim=-1) ** 2
    den = (pred_obs ** 2).sum(dim=-1) * (true_obs ** 2).sum(dim=-1)
    return num / den.clamp_min(1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None,
                   help="path to dataset (.npz/.npy/.csv/.pt); omit for synthetic smoke test")
    p.add_argument("--n_synthetic", type=int, default=8000)
    p.add_argument("--eta", type=float, default=1.0, help="known loss level used at data-gen time (1.0 = lossless)")
    p.add_argument("--observation", type=str, default="exact", choices=["exact", "power"])
    p.add_argument("--quadrature", action="store_true",
                   help="power mode: use 12 probes incl. (e_i + i e_j)/√2 (obs dim 48)")
    p.add_argument("--basis_only", action="store_true",
                   help="power mode: I1..I4 powers only (obs dim 16, 32-col rows)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--mode", type=str, default="physics", choices=["physics", "param_only"])
    p.add_argument("--n_refine_eval", type=int, default=200)
    p.add_argument("--n_starts", type=int, default=8)
    p.add_argument("--skip_convention_check", action="store_true")
    p.add_argument("--out", type=str, default="inverse_net.pt")
    args = p.parse_args()

    if args.basis_only and args.quadrature:
        p.error("--basis_only and --quadrature are incompatible")
    if args.basis_only and args.observation != "power":
        p.error("--basis_only applies only to --observation power")

    regime = Regime(
        observation=args.observation, eta=args.eta,
        quadrature=args.quadrature, basis_only=args.basis_only,
    )

    # ---- 1. load ---- #
    if args.data is None:
        print(f"[data] no --data given, generating {args.n_synthetic} synthetic "
              f"samples (observation={args.observation}, eta={args.eta}, "
              f"quadrature={args.quadrature}, basis_only={args.basis_only})")
        data = dp.synthetic_regime_dataset(args.n_synthetic, regime, seed=0)
        raw = data
    elif args.observation == "power":
        print(f"[data] loading {args.data}")
        raw = dp.load_real_power_dataset(
            args.data, quadrature=args.quadrature, basis_only=args.basis_only,
        )
        file_quad = raw["quadrature"]
        file_basis = raw["basis_only"]
        if file_quad != args.quadrature or file_basis != args.basis_only:
            print(f"[data] WARNING: file implies basis_only={file_basis} quadrature={file_quad} "
                  f"but flags were basis_only={args.basis_only} quadrature={args.quadrature}; "
                  f"using file metadata")
            regime = Regime(
                observation="power", eta=args.eta,
                quadrature=file_quad, basis_only=file_basis,
            )
        n_probes = regime.probes.shape[0]
        print(f"[data] {raw['obs'].shape[0]} samples (power: {n_probes} probes × 4 ports, "
              f"obs_dim={regime.obs_dim})")
        data = {"theta": raw["theta"], "phi": raw["phi"], "delta": raw["delta"], "obs": raw["obs"]}
    else:
        print(f"[data] loading {args.data}")
        raw = dp.load_real_dataset(args.data)
        print(f"[data] {raw['y'].shape[0]} samples (exact-field mode)")
        data = dp.real_data_to_regime_dict(raw)

    # ---- 2. convention check (real data only) ---- #
    if args.data is not None and not args.skip_convention_check:
        print("\n[convention check] verifying assumed mesh convention against 5 real rows...")
        if args.observation == "power":
            winners = vc.check_power_dataset(
                raw["theta"], raw["phi"], raw["delta"], raw["obs"],
                n_check=5, eta=args.eta, probes=regime.probes,
            )
            expected = EXPECTED_POWER
        else:
            winners = vc.check_dataset(
                raw["theta"], raw["phi"], raw["delta"], raw["y"],
                n_check=5, eta=args.eta,
            )
            expected = EXPECTED_EXACT
        if winners.get(expected, 0) < 5:
            print("\n*** WARNING: convention mismatch detected -- see README.md "
                  "'Convention check failed?' before trusting predicted params. "
                  "Continuing anyway. ***\n")
    train_d, val_d = dp.regime_train_val_split(data, val_frac=0.1)
    print(f"[data] train={train_d['obs'].shape[0]}  val={val_d['obs'].shape[0]}  "
          f"regime={args.observation}  eta={args.eta}  basis_only={regime.basis_only}  "
          f"quadrature={regime.quadrature}  obs_dim={regime.obs_dim}  device={DEVICE}")

    # ---- 4. train ---- #
    model, history = train_model_regime(
        train_d, val_d, regime, mode=args.mode, epochs=args.epochs, verbose=True,
    )
    print(f"\n[train] best val_obs_mse: {history['best_val_obs_mse']:.5f}")

    # ---- 5. eval: network-only vs refine vs multi-start refine ---- #
    n = min(args.n_refine_eval, val_d["obs"].shape[0])
    obs_eval = val_d["obs"][:n]

    model.eval()
    with torch.no_grad():
        from regime import decode
        pred_sc = model(obs_eval.to(DEVICE))
        th, ph, de = decode(pred_sc, regime)
        pred_obs_net = simulate(th, ph, de, regime)
        fid_net = obs_fidelity(pred_obs_net, obs_eval.to(DEVICE))

    th_r, ph_r, de_r, _ = refine_regime(model, obs_eval, regime, n_steps=150, lr=0.03)
    fid_refine = obs_fidelity(simulate(th_r, ph_r, de_r, regime), obs_eval.to(DEVICE))

    th_m, ph_m, de_m, _ = refine_multistart_regime(model, obs_eval, regime, n_starts=args.n_starts, n_steps=150, lr=0.03)
    fid_multi = obs_fidelity(simulate(th_m, ph_m, de_m, regime), obs_eval.to(DEVICE))

    print(f"\n[eval on {n} held-out samples] observation fidelity (found params reproduce measured data):")
    print(f"  network only          : {fid_net.mean().item():.5f}  (min {fid_net.min().item():.5f})")
    print(f"  + single-start refine  : {fid_refine.mean().item():.5f}  (min {fid_refine.min().item():.5f})")
    print(f"  + {args.n_starts}-start refine   : {fid_multi.mean().item():.5f}  (min {fid_multi.min().item():.5f})")

    torch.save(model.state_dict(), args.out)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()