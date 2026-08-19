from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mesh_forward import BLOCK_LABELS, N_MODES
from regime import Regime, decode, make_model, refine_multistart_regime, simulate
from train_tandem import DEVICE


PROBES = torch.eye(N_MODES, dtype=torch.complex64)[1:3].clone()
I2_I3_REGIME = Regime(
    observation="power",
    eta=1.0,
    basis_only=True,
    probes=PROBES,
)
FULL_POWER_REGIME = Regime(observation="power", eta=1.0, basis_only=True)


def _load_i2_i3_row(
    path: str, index: int,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor | None]:
    archive = np.load(path)
    key = "my_matrix" if "my_matrix" in archive.files else "data"
    rows = np.asarray(archive[key], dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] not in (24, 32):
        raise ValueError(
            f"Expected an (N, 24) I2/I3 or (N, 32) full basis-power dataset, got {rows.shape}"
        )
    if not 0 <= index < rows.shape[0]:
        raise IndexError(f"--index {index} out of range for dataset with {rows.shape[0]} rows")
    row = rows[index]
    if rows.shape[1] == 24:
        # Custom layout: params, I2 powers, I3 powers.
        obs = row[16:24]
        full_obs = None
    else:
        # Standard basis-only layout: params, I1, I2, I3, I4 powers.
        # This checkpoint can use only the I2 and I3 blocks.
        full_obs = torch.from_numpy(row[16:32]).unsqueeze(0)
        obs = np.concatenate([row[20:24], row[24:28]])
    return torch.from_numpy(obs).unsqueeze(0), row[:16], full_obs


def _obs_fidelity(pred_obs: torch.Tensor, target_obs: torch.Tensor) -> float:
    numerator = (pred_obs * target_obs).sum(dim=-1) ** 2
    denominator = (pred_obs ** 2).sum(dim=-1) * (target_obs ** 2).sum(dim=-1)
    return (numerator / denominator.clamp_min(1e-12)).item()


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _print_angles(label: str, theta: torch.Tensor, phi: torch.Tensor) -> None:
    print(f"\n[{label} angles]")
    for name, value in zip(BLOCK_LABELS, theta[0].detach().cpu().numpy()):
        print(f"  theta_{name:>4}: {value:.6f}")
    for name, value in zip(BLOCK_LABELS, phi[0].detach().cpu().numpy()):
        print(f"  phi_{name:>4}:   {value:.6f}")


def _print_observation(label: str, predicted: torch.Tensor, target: torch.Tensor) -> None:
    predicted_np = predicted[0].detach().cpu().numpy()
    target_np = target[0].detach().cpu().numpy()
    print(f"\n[{label} observation]")
    probe_names = ("I2", "I3") if target_np.shape[0] == 8 else ("I1", "I2", "I3", "I4")
    for probe_index, probe_name in enumerate(probe_names):
        start = probe_index * 4
        print(f"  {probe_name} target : {np.array2string(target_np[start:start + 4], precision=6)}")
        print(f"  {probe_name} pred   : {np.array2string(predicted_np[start:start + 4], precision=6)}")
        rmse = np.sqrt(np.mean((predicted_np[start:start + 4] - target_np[start:start + 4]) ** 2))
        print(f"  {probe_name} RMSE    : {rmse:.6e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference and physics refinement for the I2/I3 power-probe model."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("toy_runs/i4_dark_phi/I2_I3_net.pt"),
    )
    parser.add_argument("--data", type=Path, default=Path("NEW/training_power_data_I2_I3.npz"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--refine_steps", type=int, default=150)
    parser.add_argument("--refine_lr", type=float, default=0.03)
    parser.add_argument("--n_starts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=DEVICE)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    regime = I2_I3_REGIME
    full_regime = FULL_POWER_REGIME
    obs, true_params, full_obs = _load_i2_i3_row(str(args.data), args.index)
    obs = obs.to(args.device)
    if full_obs is not None:
        full_obs = full_obs.to(args.device)

    model = make_model(
        regime,
        width=args.width,
        n_blocks=args.n_blocks,
    ).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        predicted_sc = model(obs)
        theta0, phi0, delta0 = decode(predicted_sc, regime)
        predicted_obs0 = simulate(theta0, phi0, delta0, regime)
        predicted_full0 = simulate(theta0, phi0, delta0, full_regime)
        network_error = torch.sqrt(((predicted_obs0 - obs) ** 2).sum()).item()
        network_fidelity = _obs_fidelity(predicted_obs0, obs)

    theta, phi, delta, refined_error = refine_multistart_regime(
        model,
        obs,
        regime,
        n_starts=args.n_starts,
        n_steps=args.refine_steps,
        lr=args.refine_lr,
        seed=args.seed,
    )
    with torch.no_grad():
        predicted_obs = simulate(theta, phi, delta, regime)
        predicted_full = simulate(theta, phi, delta, full_regime)
        refined_error = torch.sqrt(refined_error).item()
        refined_fidelity = _obs_fidelity(predicted_obs, obs)

    network_params = torch.cat([theta0[0], phi0[0]]).detach().cpu().numpy()
    refined_params = torch.cat([theta[0], phi[0]]).detach().cpu().numpy()
    label_rmse0 = np.sqrt(np.mean(_wrap_to_pi(network_params - true_params[:12]) ** 2))
    label_rmse = np.sqrt(np.mean(_wrap_to_pi(refined_params - true_params[:12]) ** 2))

    print("[inference]")
    print("  regime       : power, probes used by checkpoint = I2 + I3")
    print("  input        : I2 + I3 only")
    print("  evaluation   : reconstructed angles simulated on I1 + I2 + I3 + I4")
    print(f"  data         : {args.data}")
    print(f"  checkpoint   : {args.checkpoint}")
    print(f"  index        : {args.index}")
    print(f"  device       : {args.device}")
    print(f"  obs_dim      : {regime.obs_dim}")
    print(f"  network fid  : {network_fidelity:.6f}")
    print(f"  network RMSE : {network_error:.6e}")
    print(f"  refined fid  : {refined_fidelity:.6f}")
    print(f"  refined RMSE : {refined_error:.6e}")
    print(f"  label RMSE   : network={label_rmse0:.6f}, refined={label_rmse:.6f} (diagnostic only)")

    _print_angles("network", theta0, phi0)
    _print_angles("refined", theta, phi)
    _print_observation("network I2/I3 input", predicted_obs0, obs)
    _print_observation("refined I2/I3 input", predicted_obs, obs)
    if full_obs is not None:
        print(f"  full 4-probe network fidelity : {_obs_fidelity(predicted_full0, full_obs):.6f}")
        print(f"  full 4-probe refined fidelity : {_obs_fidelity(predicted_full, full_obs):.6f}")
        _print_observation("network all probes", predicted_full0, full_obs)
        _print_observation("refined all probes", predicted_full, full_obs)
    else:
        print("\n  I1/I4 held-out comparison: unavailable; the input file contains only I2/I3")


if __name__ == "__main__":
    main()
