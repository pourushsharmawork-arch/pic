"""Generate (params, observation) datasets for the PARAMS regimes.

    observation ∈ {exact, power}  ×  loss (eta) ∈ {1.0 lossless, <1 lossy}
    power mode optionally adds quadrature superposition probes (--quadrature)

Row width = 16 params + obs_dim:
  exact              -> 16 + 32 = 48
  power              -> 16 + 32 = 48   (8 probes × 4 ports)
  power + quadrature -> 16 + 48 = 64   (12 probes × 4 ports)

Usage:
    python datagen.py --observation exact --loss 1.0  --out data/exact_lossless.npz
    python datagen.py --observation power --loss 0.85 --out data/power_lossy.npz
    python datagen.py --observation power --loss 1.0 --quadrature --out data/power_quad_lossless.npz
"""
from __future__ import annotations

import argparse
import numpy as np
from scipy.linalg import block_diag

import qfsim as qf
from mesh_forward import make_power_probes, power_obs_dim


def loss_matrix(η1, η2):
    if not (0 <= η1 <= 1 and 0 <= η2 <= 1):
        raise ValueError("Efficiencies eta1 and eta2 must lie in [0,1].")
    return np.diag([η1, η2]).astype(float)


def lossy_U2mzi(θ, α, β, χ, ηlist):
    U = qf.U2mzi(θ, α, β, χ)
    L_in = loss_matrix(ηlist[0], ηlist[1])
    L_out = loss_matrix(ηlist[2], ηlist[3])
    return L_in @ U @ L_out


def lossy_U4_circuit(U, losses):
    blocks, D, W = qf.one_two_decomposition.decompose_U4_rectangular(U)
    η = losses
    if η.shape != (6, 4):
        raise ValueError(f"Shape of losses is {η.shape} not (6, 4)")

    θlist = []
    ϕlist = []
    for mzi in blocks:
        θlist.append(mzi["theta"])
        ϕlist.append(mzi["phi"])

    L1 = block_diag(
        np.eye(1),
        lossy_U2mzi(θ=θlist[0], α=-ϕlist[0], β=np.pi / 2, χ=0, ηlist=η[0]),
        np.eye(1),
    )
    L2 = block_diag(
        lossy_U2mzi(θ=θlist[1], α=-ϕlist[1], β=np.pi / 2, χ=0, ηlist=η[1]),
        lossy_U2mzi(θ=θlist[2], α=-ϕlist[2], β=np.pi / 2, χ=0, ηlist=η[2]),
    )
    L3 = block_diag(
        np.eye(1),
        lossy_U2mzi(θ=θlist[3], α=-ϕlist[3], β=np.pi / 2, χ=0, ηlist=η[3]),
        np.eye(1),
    )
    L4 = block_diag(
        lossy_U2mzi(θ=θlist[4], α=-ϕlist[4], β=np.pi / 2, χ=0, ηlist=η[4]),
        lossy_U2mzi(θ=θlist[5], α=-ϕlist[5], β=np.pi / 2, χ=0, ηlist=η[5]),
    )

    return L1 @ L2 @ L3 @ L4 @ D, θlist, ϕlist, np.angle(np.diag(D))


def project_to_su4(U):
    phase = np.angle(np.linalg.det(U)) / 4
    return U * np.exp(-1j * phase)


def _basis_probes_numpy():
    return np.eye(4, dtype=complex)  # rows I1..I4; (probes @ U) picks rows of U


def _pack_exact(fields):
    parts = [np.concatenate([f.real, f.imag]) for f in fields]
    return np.concatenate(parts)


def _pack_power(fields):
    return np.concatenate([np.abs(f) ** 2 for f in fields])


def sample_programmed_unitary_statistics(
    loss=1.0,
    n_samples=10_000,
    seed=1234,
    observation="exact",
    quadrature=False,
):
    if observation not in ("exact", "power"):
        raise ValueError(f"observation must be 'exact' or 'power', got {observation!r}")
    if observation == "exact" and quadrature:
        raise ValueError("quadrature probes apply only to observation='power'")
    obs_dim = 32 if observation == "exact" else power_obs_dim(quadrature)
    print(
        f"generating {n_samples} samples  observation={observation}  "
        f"loss={loss}  quadrature={quadrature}  obs_dim={obs_dim}"
    )
    rng = np.random.default_rng(seed)
    all_rows = []
    losses = np.full((6, 4), loss, dtype=float)
    if observation == "exact":
        probes = _basis_probes_numpy()
    else:
        probes = make_power_probes(quadrature).numpy()

    for _ in range(n_samples):
        U = qf.misc.random_unitary(4, seed=int(rng.integers(10**9)))
        U = project_to_su4(U)
        U_circuit, theta_list, phi_list, D_list = lossy_U4_circuit(U, losses)
        parameters = np.concatenate([
            np.asarray(theta_list).reshape(-1),
            np.asarray(phi_list).reshape(-1),
            np.asarray(D_list).reshape(-1),
        ])
        if parameters.size != 16:
            raise ValueError(
                f"Expected 16 decomposition parameters, but received {parameters.size}."
            )

        fields = (probes @ U_circuit)
        if observation == "exact":
            y = _pack_exact([fields[i] for i in range(fields.shape[0])])
        else:
            y = _pack_power([fields[i] for i in range(fields.shape[0])])
        all_rows.append(np.concatenate([parameters, y]))
    return np.vstack(all_rows)


def main():
    p = argparse.ArgumentParser(description="Generate PARAMS training data.")
    p.add_argument("--observation", choices=["exact", "power"], default="exact")
    p.add_argument(
        "--quadrature", action="store_true",
        help="power mode only: add (e_i + i e_j)/√2 probes (48-D obs, 64-col rows)",
    )
    p.add_argument("--loss", type=float, default=1.0,
                   help="per-port amplitude efficiency (1.0 = lossless). Must match --eta at train time.")
    p.add_argument("--n_samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=str, default="data/data.npz")
    args = p.parse_args()

    if not (0.0 < args.loss <= 1.0):
        p.error("--loss must be in (0, 1]")
    if args.quadrature and args.observation != "power":
        p.error("--quadrature applies only to --observation power")

    Big_Matrix = np.array(
        sample_programmed_unitary_statistics(
            loss=args.loss,
            n_samples=args.n_samples,
            seed=args.seed,
            observation=args.observation,
            quadrature=args.quadrature,
        ),
        dtype=float,
    )
    np.savez_compressed(
        args.out,
        my_matrix=Big_Matrix,
        observation=np.array(args.observation),
        loss=np.array(args.loss),
        quadrature=np.array(args.quadrature),
    )
    print(f"saved {Big_Matrix.shape} -> {args.out}  "
          f"(observation={args.observation}, loss={args.loss}, quadrature={args.quadrature})")


if __name__ == "__main__":
    main()
