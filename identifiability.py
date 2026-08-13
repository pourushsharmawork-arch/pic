"""
Local identifiability analysis for the PARAMS observation map.

Answers two questions using the same forward model as training:

  Q1  Basis-probe powers only: which (θ, φ) directions change the measurement
      by zero to first order?  → Jacobian null space = *unobservable* directions.

  Q2  With superposition (± quadrature) probes: which directions still have
      zero obs gradient?  → *gauge* directions — different params, same data.
      Examples: uniform φ shifts on MZIs that only enter as relative phases;
      mesh-decomposition redundancy (several angle sets → same M).

Method: at a generic parameter point, build J = ∂obs/∂params (obs_dim × n_params)
via autograd, SVD, report rank / null-space basis / numerical probe along null
directions.

Usage:
    python identifiability.py --compare              # table over probe configs
    python identifiability.py --observation power --basis_only
    python identifiability.py --observation power --eta 0.85
    python identifiability.py --observation exact --eta 1.0 --probe-null 0
"""
from __future__ import annotations

import argparse

import torch

from mesh_forward import BLOCK_LABELS
from regime import Regime, simulate

PARAM_NAMES_16 = (
    [f"theta_{BLOCK_LABELS[i]}" for i in range(6)]
    + [f"phi_{BLOCK_LABELS[i]}" for i in range(6)]
    + [f"delta_{i}" for i in range(4)]
)
PARAM_NAMES_12 = (
    [f"theta_{BLOCK_LABELS[i]}" for i in range(6)]
    + [f"phi_{BLOCK_LABELS[i]}" for i in range(6)]
)


def param_names(regime: Regime) -> list[str]:
    return list(PARAM_NAMES_16 if regime.predict_delta else PARAM_NAMES_12)


def observation_jacobian(
    regime: Regime,
    theta: torch.Tensor,
    phi: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, list[str]]:
    """J: (obs_dim, n_active_params) at a single parameter point (no batch)."""
    names = param_names(regime)
    de_fixed = delta.reshape(4)

    if regime.predict_delta:
        flat0 = torch.cat([theta.reshape(6), phi.reshape(6), de_fixed]).clone()

        def f(fp):
            return simulate(
                fp[:6].unsqueeze(0), fp[6:12].unsqueeze(0), fp[12:16].unsqueeze(0), regime,
            ).reshape(-1)
    else:
        flat0 = torch.cat([theta.reshape(6), phi.reshape(6)]).clone()

        def f(fp):
            return simulate(
                fp[:6].unsqueeze(0), fp[6:12].unsqueeze(0), de_fixed.unsqueeze(0), regime,
            ).reshape(-1)

    J = torch.autograd.functional.jacobian(f, flat0)
    return J, names


def analyze_identifiability(
    regime: Regime,
    theta: torch.Tensor | None = None,
    phi: torch.Tensor | None = None,
    delta: torch.Tensor | None = None,
    *,
    sv_tol: float = 1e-4,
    seed: int = 0,
) -> dict:
    """SVD-based rank / null-space report for one Regime at a generic point."""
    g = torch.Generator().manual_seed(seed)
    if theta is None:
        theta = torch.rand(6, generator=g) * 2 * torch.pi
    if phi is None:
        phi = torch.rand(6, generator=g) * 2 * torch.pi
    if delta is None:
        delta = torch.rand(4, generator=g) * 2 * torch.pi

    J, names = observation_jacobian(regime, theta, phi, delta)
    sv = torch.linalg.svdvals(J)
    rank = int((sv > sv_tol).sum().item())
    n_params = J.shape[1]
    n_obs = J.shape[0]
    U, S, Vh = torch.linalg.svd(J, full_matrices=True)
    null_mask = S < sv_tol
    null_dim = int(null_mask.sum().item()) if null_mask.any() else max(0, n_params - rank)
    # right null vectors: rows of Vh with small singular values
    null_vecs = Vh[S < sv_tol] if (S < sv_tol).any() else Vh[rank:]

    return {
        "regime": regime,
        "J": J,
        "names": names,
        "singular_values": sv,
        "rank": rank,
        "n_params": n_params,
        "n_obs": n_obs,
        "null_dim": null_dim,
        "null_vectors": null_vecs,
        "theta": theta,
        "phi": phi,
        "delta": delta,
        "sv_tol": sv_tol,
    }


def _format_linear_combo(v: torch.Tensor, names: list[str], topk: int = 6) -> str:
    v = v / (v.abs().max().clamp_min(1e-12))
    idx = torch.argsort(-v.abs())[:topk]
    terms = [f"{v[i].item():+.3f}·{names[i]}" for i in idx if v[i].abs() > 0.05]
    return "  ".join(terms) if terms else "(small)"


def _phi_gauge_hints(v: torch.Tensor, names: list[str]) -> list[str]:
    """Heuristic: flag null vectors dominated by φ with near-uniform weights."""
    hints = []
    phi_idx = [i for i, n in enumerate(names) if n.startswith("phi_")]
    if not phi_idx:
        return hints
    vp = v[phi_idx]
    if vp.abs().sum() < 0.3 * v.abs().sum():
        return hints
    # uniform φ combination
    if vp.std() / (vp.abs().mean().clamp_min(1e-12)) < 0.35:
        active = [names[i] for i in phi_idx if v[i].abs() > 0.15]
        if len(active) >= 2:
            hints.append(
                f"φ-gauge-like: nearly uniform combination of {', '.join(active)} "
                f"(obs unchanged to first order — e.g. φ_a+φ_b+… = const)"
            )
    # single φ dominant
    top = max(phi_idx, key=lambda i: v[i].abs().item())
    if v[top].abs() > 0.7 * v.abs().max():
        hints.append(f"dominant φ direction: {names[top]}")
    return hints


def print_report(result: dict, *, probe_null: int | None = None, eps: float = 0.15):
    r = result
    regime = r["regime"]
    print(f"\n{'=' * 72}")
    print(f"Identifiability: {regime.probe_config}  "
          f"(obs_dim={regime.obs_dim}, n_params={r['n_params']}, eta={regime.eta})")
    print(f"{'=' * 72}")
    print(f"Jacobian rank: {r['rank']}/{r['n_params']}  "
          f"(obs dim {r['n_obs']}, null dim {r['null_dim']}, tol={r['sv_tol']})")
    print(f"Singular values (top 8): {r['singular_values'][:8].tolist()}")
    if r['null_dim'] == 0:
        print("→ Locally fully identifiable: no null directions at this point.")
    else:
        print(f"→ {r['null_dim']} unobservable / gauge direction(s) at first order:")
        for k, v in enumerate(r["null_vectors"]):
            print(f"\n  null #{k}: {_format_linear_combo(v, r['names'])}")
            for hint in _phi_gauge_hints(v, r["names"]):
                print(f"           ** {hint}")

    if probe_null is not None and r["null_dim"] > 0:
        k = min(probe_null, r["null_vectors"].shape[0] - 1)
        v = r["null_vectors"][k]
        th0, ph0, de0 = r["theta"], r["phi"], r["delta"]
        if regime.predict_delta:
            flat0 = torch.cat([th0, ph0, de0])
            n = 16
        else:
            flat0 = torch.cat([th0, ph0])
            n = 12
        with torch.no_grad():
            obs0 = simulate(th0.unsqueeze(0), ph0.unsqueeze(0), de0.unsqueeze(0), regime)
        flat1 = flat0 + eps * v[:n]
        if regime.predict_delta:
            th1, ph1, de1 = flat1[:6], flat1[6:12], flat1[12:16]
        else:
            th1, ph1, de1 = flat1[:6], flat1[6:12], de0
        with torch.no_grad():
            obs1 = simulate(th1.unsqueeze(0), ph1.unsqueeze(0), de1.unsqueeze(0), regime)
        d_obs = (obs1 - obs0).abs().max().item()
        print(f"\n  Numerical probe along null #{k} (ε={eps}): max|Δobs| = {d_obs:.3e}")
        print("  (should be ≪ 1 if truly a gauge/null direction; large → nonlinear or wrong tol)")


def compare_configurations(eta: float = 1.0, sv_tol: float = 1e-4, seed: int = 0):
    """Side-by-side rank table — the main answer to Q1 vs Q2."""
    configs = [
        Regime(observation="exact", eta=eta),
        Regime(observation="power", eta=eta, basis_only=True),
        Regime(observation="power", eta=eta, basis_only=False, quadrature=False),
        Regime(observation="power", eta=eta, basis_only=False, quadrature=True),
    ]
    print(f"\nIdentifiability comparison (eta={eta}, generic random point, seed={seed})")
    print(f"{'config':<32} {'obs':>4} {'params':>6} {'rank':>6} {'null':>6}  interpretation")
    print("-" * 90)
    for regime in configs:
        r = analyze_identifiability(regime, sv_tol=sv_tol, seed=seed)
        if regime.observation == "exact":
            interp = "full local rank on U(4) (16 params)"
        elif regime.basis_only:
            interp = "many φ (and some θ) directions invisible — Q1"
        elif regime.quadrature:
            interp = "residual null = mesh gauge only — Q2"
        else:
            interp = "local rank 12; global gauge via --global-demo — Q2"
        print(
            f"{regime.probe_config:<32} {r['n_obs']:>4} {r['n_params']:>6} "
            f"{r['rank']:>6} {r['null_dim']:>6}  {interp}"
        )
    print("\nRun `python identifiability.py --observation power --basis_only` for null-vector detail.")
    print("High observation fidelity from the net does NOT mean params are unique — check null dim.")


def demo_global_nonuniqueness(
    regime: Regime,
    *,
    seed: int = 0,
    n_pairs: int = 3,
    refine_steps: int = 400,
    lr: float = 0.05,
):
    """Two independent cold-start refines on the SAME observation.

    If max|obs error| ≈ 0 for both but ||params_a - params_b|| is large,
    parameters are only identifiable up to a global equivalence (gauge /
    decomposition branch) even when the local Jacobian has full rank.
    """
    g = torch.Generator().manual_seed(seed)
    th0 = torch.rand(6, generator=g) * 2 * torch.pi
    ph0 = torch.rand(6, generator=g) * 2 * torch.pi
    de0 = torch.rand(4, generator=g) * 2 * torch.pi
    with torch.no_grad():
        obs = simulate(th0.unsqueeze(0), ph0.unsqueeze(0), de0.unsqueeze(0), regime)

    print(f"\nGlobal non-uniqueness demo: {regime.probe_config}  obs_dim={regime.obs_dim}")
    print("(two cold-start gradient solves from different random initial angles)\n")

    def cold_solve(init_seed: int):
        gi = torch.Generator().manual_seed(init_seed)
        th = (torch.rand(6, generator=gi) * 2 * torch.pi).clone().requires_grad_(True)
        ph = (torch.rand(6, generator=gi) * 2 * torch.pi).clone().requires_grad_(True)
        de = de0.clone()
        opt = torch.optim.Adam([th, ph], lr=lr)
        for _ in range(refine_steps):
            opt.zero_grad()
            pred = simulate(th.unsqueeze(0), ph.unsqueeze(0), de.unsqueeze(0), regime)
            loss = ((pred - obs) ** 2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = simulate(th.unsqueeze(0), ph.unsqueeze(0), de.unsqueeze(0), regime)
            err = ((pred - obs) ** 2).sum().sqrt().item()
        return th.detach(), ph.detach(), err

    for i in range(n_pairs):
        ta, pa, ea = cold_solve(seed + 100 + 10 * i)
        tb, pb, eb = cold_solve(seed + 200 + 10 * i)
        d_th = (ta - tb).abs().mean().item()
        d_ph = (pa - pb).abs().mean().item()
        # circular distance on phi
        d_ph_circ = torch.atan2(torch.sin(pa - pb), torch.cos(pa - pb)).abs().mean().item()
        print(f"  pair {i}: obs_RMSE=({ea:.2e}, {eb:.2e})  "
              f"mean|Δθ|={d_th:.3f}  mean|circular Δφ|={d_ph_circ:.3f}")
    print("  → Large param spread at equal obs fidelity = gauge / branch non-uniqueness (Q2)")


def main():
    p = argparse.ArgumentParser(description="Local identifiability / gauge analysis")
    p.add_argument("--compare", action="store_true", help="table over all probe configs")
    p.add_argument("--global-demo", action="store_true",
                   help="two cold-start solves, same obs — shows Q2 global gauge")
    p.add_argument("--observation", choices=["exact", "power"], default="power")
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--basis_only", action="store_true")
    p.add_argument("--quadrature", action="store_true")
    p.add_argument("--sv_tol", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe-null", type=int, default=None,
                   help="numerically move ε along null vector #k and print |Δobs|")
    p.add_argument("--eps", type=float, default=0.15)
    args = p.parse_args()

    if args.compare:
        compare_configurations(eta=args.eta, sv_tol=args.sv_tol, seed=args.seed)
        return

    regime = Regime(
        observation=args.observation,
        eta=args.eta,
        basis_only=args.basis_only,
        quadrature=args.quadrature,
    )
    if args.global_demo:
        demo_global_nonuniqueness(regime, seed=args.seed)
        return

    result = analyze_identifiability(regime, sv_tol=args.sv_tol, seed=args.seed)
    print_report(result, probe_null=args.probe_null, eps=args.eps)


if __name__ == "__main__":
    main()
