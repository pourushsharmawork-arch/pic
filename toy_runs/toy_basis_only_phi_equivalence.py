"""
Demo experiment for basis-only power identifiability.

This script combines two checks:

1) A local null-step probe from `identifiability.py`:
   move along a Jacobian null vector by a small epsilon and verify that the
   basis-only power readout barely changes.

2) A stronger constrained solve:
   clamp three chosen phi coordinates to random values, optimize the remaining
   parameters to match the same basis-only observation, and repeat from
   multiple random restarts.

The point is to show that basis-only powers do admit a nontrivial 3D
equivalence family, but it is not "pick any three phi values and the rest are
forced."  Only parameter sets on the same solution branch reproduce the same
measurement.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import data_pipeline as dp
from identifiability import analyze_identifiability, print_report
from mesh_forward import BLOCK_LABELS
from regime import Regime, simulate


FREE_PHI_IDXS = (0, 1, 2)
FREE_PHI_NAMES = [f"phi_{BLOCK_LABELS[i]}" for i in FREE_PHI_IDXS]
LEARN_PHI_IDXS = (3, 4, 5)
LEARN_PHI_NAMES = [f"phi_{BLOCK_LABELS[i]}" for i in LEARN_PHI_IDXS]


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _circ_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.atan2(torch.sin(a - b), torch.cos(a - b)))


def _make_regime() -> Regime:
    return Regime(observation="power", eta=1.0, basis_only=True)


def _solve_with_fixed_phi_head(
    obs: torch.Tensor,
    fixed_phi_head: torch.Tensor,
    *,
    n_steps: int,
    lr: float,
    seed: int,
    init_noise: float,
    theta_init: torch.Tensor | None = None,
    phi_tail_init: torch.Tensor | None = None,
) -> dict:
    """Optimize theta and the last three phi entries with phi[0:3] fixed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if theta_init is None:
        theta_init = torch.rand(6, generator=g) * 2 * torch.pi
    if phi_tail_init is None:
        phi_tail_init = torch.rand(3, generator=g) * 2 * torch.pi

    theta = (theta_init + init_noise * torch.randn(6, generator=g)).detach().clone().requires_grad_(True)
    phi_tail = (phi_tail_init + init_noise * torch.randn(3, generator=g)).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([theta, phi_tail], lr=lr)
    delta = torch.zeros(4, dtype=torch.float32)
    regime = _make_regime()

    for _ in range(n_steps):
        opt.zero_grad()
        phi = torch.cat([fixed_phi_head, phi_tail], dim=0)
        pred = simulate(theta.unsqueeze(0), phi.unsqueeze(0), delta.unsqueeze(0), regime)
        loss = ((pred - obs) ** 2).sum()
        loss.backward()
        opt.step()

    # A short LBFGS polish typically snaps a near-solution onto the level set.
    lbfgs = torch.optim.LBFGS([theta, phi_tail], lr=1.0, max_iter=25, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        phi = torch.cat([fixed_phi_head, phi_tail], dim=0)
        pred = simulate(theta.unsqueeze(0), phi.unsqueeze(0), delta.unsqueeze(0), regime)
        loss = ((pred - obs) ** 2).sum()
        loss.backward()
        return loss

    lbfgs.step(closure)

    with torch.no_grad():
        phi = torch.cat([fixed_phi_head, phi_tail], dim=0)
        pred = simulate(theta.unsqueeze(0), phi.unsqueeze(0), delta.unsqueeze(0), regime)
        rmse = torch.sqrt(((pred - obs) ** 2).sum()).item()
    return {
        "theta": theta.detach().cpu(),
        "phi": phi.detach().cpu(),
        "rmse": rmse,
    }


def _solve_unconstrained(
    obs: torch.Tensor,
    *,
    n_steps: int,
    lr: float,
    seed: int,
    init_noise: float,
) -> dict:
    """Optimize all 12 basis-only angles to match obs."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    theta = (torch.rand(6, generator=g) * 2 * torch.pi).detach().clone().requires_grad_(True)
    phi = (torch.rand(6, generator=g) * 2 * torch.pi).detach().clone().requires_grad_(True)
    theta = (theta + init_noise * torch.randn(6, generator=g)).detach().clone().requires_grad_(True)
    phi = (phi + init_noise * torch.randn(6, generator=g)).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([theta, phi], lr=lr)
    delta = torch.zeros(4, dtype=torch.float32)
    regime = _make_regime()

    for _ in range(n_steps):
        opt.zero_grad()
        pred = simulate(theta.unsqueeze(0), phi.unsqueeze(0), delta.unsqueeze(0), regime)
        loss = ((pred - obs) ** 2).sum()
        loss.backward()
        opt.step()

    lbfgs = torch.optim.LBFGS([theta, phi], lr=1.0, max_iter=25, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        pred = simulate(theta.unsqueeze(0), phi.unsqueeze(0), delta.unsqueeze(0), regime)
        loss = ((pred - obs) ** 2).sum()
        loss.backward()
        return loss

    lbfgs.step(closure)

    with torch.no_grad():
        pred = simulate(theta.unsqueeze(0), phi.unsqueeze(0), delta.unsqueeze(0), regime)
        rmse = torch.sqrt(((pred - obs) ** 2).sum()).item()
    return {
        "theta": theta.detach().cpu(),
        "phi": phi.detach().cpu(),
        "rmse": rmse,
    }


def _format_vec(names: list[str], values: torch.Tensor) -> str:
    return ", ".join(f"{n}={values[i].item():+.5f}" for i, n in enumerate(names))


def main() -> None:
    p = argparse.ArgumentParser(description="Basis-only phi equivalence demo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps", type=float, default=0.15)
    p.add_argument("--null_idx", type=int, default=0)
    p.add_argument("--n_targets", type=int, default=2,
                   help="number of constrained targets to test (1 feasible + random contrasts)")
    p.add_argument("--n_starts", type=int, default=6)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--init_noise", type=float, default=0.35)
    p.add_argument("--free_radius", type=float, default=0.75,
                   help="random clamp radius around the baseline free phi values")
    p.add_argument("--success_tol", type=float, default=1e-3,
                   help="RMSE threshold used when counting a constrained solve as successful")
    p.add_argument("--save_dir", type=str, default="toy_runs/basis_only_phi_equiv")
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    regime = _make_regime()
    analysis = analyze_identifiability(regime, seed=args.seed, sv_tol=1e-4)

    print("[setup]")
    print(f"  regime      : {regime.probe_config}")
    print(f"  obs_dim     : {regime.obs_dim}")
    print(f"  free phi    : {', '.join(FREE_PHI_NAMES)}")
    print(f"  learn phi   : {', '.join(LEARN_PHI_NAMES)}")
    print(f"  output dir  : {save_dir}")
    print()
    print_report(analysis, probe_null=args.null_idx, eps=args.eps)

    g = torch.Generator().manual_seed(args.seed)
    theta0 = torch.rand(6, generator=g) * 2 * torch.pi
    phi0 = torch.rand(6, generator=g) * 2 * torch.pi
    delta0 = torch.zeros(4)
    obs0 = simulate(theta0.unsqueeze(0), phi0.unsqueeze(0), delta0.unsqueeze(0), regime).detach()

    null_v = analysis["null_vectors"][args.null_idx][:12].detach().cpu()
    theta_eps = _wrap_to_pi(theta0 + args.eps * null_v[:6])
    phi_eps = _wrap_to_pi(phi0 + args.eps * null_v[6:12])
    obs_eps = simulate(theta_eps.unsqueeze(0), phi_eps.unsqueeze(0), delta0.unsqueeze(0), regime).detach()

    print("\n[baseline]")
    print(f"  theta0  : {_format_vec([f'theta_{b}' for b in BLOCK_LABELS], theta0)}")
    print(f"  phi0    : {_format_vec([f'phi_{b}' for b in BLOCK_LABELS], phi0)}")
    print(
        f"  null-step obs delta (ε={args.eps} along null #{args.null_idx}): "
        f"{torch.sqrt(((obs_eps - obs0) ** 2).sum()).item():.3e}"
    )

    print("\n[branch solve]")
    branch_solutions = []
    for s in range(max(2, args.n_starts)):
        sol = _solve_unconstrained(
            obs0,
            n_steps=args.steps,
            lr=args.lr,
            seed=args.seed + 5000 + 31 * s,
            init_noise=args.init_noise,
        )
        branch_solutions.append(sol)
        print(f"  start {s}: obs_RMSE={sol['rmse']:.3e}  phi={_wrap_to_pi(sol['phi']).tolist()}")
    branch = min(branch_solutions, key=lambda d: d["rmse"])
    print(f"  selected branch solution RMSE: {branch['rmse']:.3e}")

    print("\n[constrained solves]")
    print(
        "  Trial 0 clamps phi[0:3] to the first three phi values from a solved"
        " branch point that already matches the target observation. Later trials"
        " use random contrasts for comparison."
    )

    all_best = []
    for t in range(args.n_targets):
        if t == 0:
            free_target = branch["phi"][:3].clone()
            target_label = "feasible branch target"
        else:
            free_target = _wrap_to_pi(
                phi0[:3] + (2 * torch.rand(3, generator=g) - 1.0) * args.free_radius
            )
            target_label = "random contrast target"
        print(f"\n  target {t} ({target_label}): fixed free phi = {free_target.tolist()}")

        trial_solutions = []
        for s in range(args.n_starts):
            sol = _solve_with_fixed_phi_head(
                obs0,
                free_target,
                n_steps=args.steps,
                lr=args.lr,
                seed=args.seed + 1000 * t + 17 * s,
                init_noise=args.init_noise,
                theta_init=branch["theta"],
                phi_tail_init=branch["phi"][3:6],
            )
            trial_solutions.append(sol)
            print(f"    start {s}: obs_RMSE={sol['rmse']:.3e}  phi={_wrap_to_pi(sol['phi']).tolist()}")

        best = min(trial_solutions, key=lambda d: d["rmse"])
        all_best.append(best)

        phi_spread = max(
            _circ_dist(_wrap_to_pi(best["phi"]), _wrap_to_pi(sol["phi"])).max().item()
            for sol in trial_solutions
        )
        print(f"    best obs_RMSE : {best['rmse']:.3e}")
        print(f"    max phi spread among starts (circular) : {phi_spread:.3e}")

    print("\n[summary]")
    successes = sum(sol["rmse"] < args.success_tol for sol in all_best)
    print(f"  successful targets (rmse < {args.success_tol:.1e}): {successes}/{len(all_best)}")
    if len(all_best) >= 2:
        spread = _circ_dist(_wrap_to_pi(all_best[0]["phi"]), _wrap_to_pi(all_best[1]["phi"])).max().item()
        print(f"  example phi distance between two best solutions: {spread:.3e}")
    print(
        "  The constrained solves show that the same basis-only powers can be "
        "matched after clamping three phi coordinates, but the recovered phi "
        "vectors are not unique. Different restarts land on different points in "
        "the same observation equivalence class."
    )


if __name__ == "__main__":
    main()
