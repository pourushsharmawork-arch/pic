"""
End-to-end driver. Defaults to synthetic (self-consistent) data so this runs
out of the box as a pipeline smoke test; point --data at your real dataset
file to run for real. See data_pipeline.py's module docstring for the
convention caveat before trusting results on real data -- run
verify_convention.py first.

Usage:
    python run_pipeline.py                          # synthetic smoke test
    python run_pipeline.py --data mydata.npy         # real data
"""
from __future__ import annotations

import argparse

import torch

import data_pipeline as dp
import verify_convention as vc
from train_tandem import train_model, refine, refine_multistart, DEVICE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None, help="path to real dataset; omit for synthetic")
    p.add_argument("--n_synthetic", type=int, default=8000)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--mode", type=str, default="physics",
                    choices=["physics", "physics_lossy", "param_only"])
    p.add_argument("--eta", type=float, default=None,
                    help="loss level used at data-gen time (datagen.py's `loss` kwarg); "
                         "required for --mode physics_lossy")
    p.add_argument("--n_refine_eval", type=int, default=200, help="val samples to run refinement on")
    p.add_argument("--n_starts", type=int, default=8)
    p.add_argument("--skip_convention_check", action="store_true")
    args = p.parse_args()

    if args.mode == "physics_lossy" and args.eta is None:
        p.error("--mode physics_lossy requires --eta (the loss level used at data-gen time)")

    # ---- 1. data ---- #
    if args.data is None:
        print(f"[data] no --data given, generating {args.n_synthetic} synthetic samples")
        if args.mode == "physics_lossy":
            print(f"[data] using synthetic_lossy_dataset(eta={args.eta})")
            raw = dp.synthetic_lossy_dataset(args.n_synthetic, eta=args.eta, seed=0)
        else:
            raw = dp.synthetic_dataset(args.n_synthetic, seed=0)
    else:
        print(f"[data] loading real dataset from {args.data}")
        raw = dp.load_real_dataset(args.data)

        if not args.skip_convention_check:
            print("\n[convention check] verifying mesh_forward's assumed convention "
                  "against a few real rows before training anything...")
            winners = vc.check_dataset(
                raw["theta"], raw["phi"], raw["delta"], raw["y"],
                n_check=5, eta=args.eta if args.eta is not None else 1.0,
            )
            expected = vc.EXPECTED_EXACT
            if winners.get(expected, 0) < 5:
                print("\n*** WARNING: real data does not consistently match the assumed "
                      "convention. Training will still run (physics_loss trains against "
                      "y directly, not against build_mesh(real_params)), but predicted "
                      "params will NOT line up with your real theta/phi/delta labels. "
                      "Fix mesh_forward.mzi_2x2 / the y layout before trusting param_aux_loss "
                      "or using predicted params to drive real hardware. Continuing anyway. ***\n")

    data = dp.build_dataset_dict(raw)
    train_d, val_d = dp.train_val_split(data, val_frac=0.1)
    print(f"[data] train={train_d['y'].shape[0]}  val={val_d['y'].shape[0]}  device={DEVICE}")

    # ---- 2. train ---- #
    model, history = train_model(
        train_d, val_d, mode=args.mode, epochs=args.epochs,
        eta=args.eta if args.mode == "physics_lossy" else None, verbose=True,
    )
    print(f"\n[train] best val fidelity: {history['best_val_fidelity']:.4f}")

    # ---- 3. eval: network-only vs single-start refine vs multi-start refine ---- #
    n = min(args.n_refine_eval, val_d["y"].shape[0])
    y_eval = val_d["y"][:n]
    U_eval = val_d["U"][:n]
    eta = args.eta if args.mode == "physics_lossy" else None

    from train_tandem import sincos_to_angles, reconstruction_fidelity, normalized_overlap_fidelity
    from mesh_forward import build_mesh, build_mesh_lossy

    fid_fn = normalized_overlap_fidelity if eta is not None else reconstruction_fidelity
    model.eval()
    with torch.no_grad():
        pred_sc = model(y_eval.to(DEVICE))
        th, ph, de = sincos_to_angles(pred_sc)
        U_pred = build_mesh_lossy(th, ph, de, eta) if eta is not None else build_mesh(th, ph, de)
        fid_net = fid_fn(U_pred, U_eval.to(DEVICE))

    _, _, _, fid_refine, _ = refine(model, y_eval, U_eval, n_steps=150, lr=0.03, eta=eta)
    _, _, _, fid_multi, _ = refine_multistart(model, y_eval, U_eval, n_starts=args.n_starts,
                                               n_steps=150, lr=0.03, eta=eta)

    print(f"\n[eval on {n} held-out samples] mean process fidelity:")
    print(f"  network only          : {fid_net.mean().item():.5f}  (min {fid_net.min().item():.5f})")
    print(f"  + single-start refine  : {fid_refine.mean().item():.5f}  (min {fid_refine.min().item():.5f})")
    print(f"  + {args.n_starts}-start refine   : {fid_multi.mean().item():.5f}  (min {fid_multi.min().item():.5f})")

    torch.save(model.state_dict(), "inverse_net.pt")
    print("\n[saved] inverse_net.pt")


if __name__ == "__main__":
    main()