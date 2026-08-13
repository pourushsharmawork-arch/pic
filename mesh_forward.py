"""
Differentiable forward model for the 6-MZI / 4-phase rectangular mesh described
in decomp.py:

    U = Ga23 Ga12 Ga34 Gb23 Gb12 Gb34 D

Each G_xy is a 2-parameter (theta, phi) unitary embedded in the 2x2 subspace
spanned by modes x,y (1-indexed in the labels, 0-indexed internally); D is a
diagonal matrix of 4 phases.

The 2x2 MZI convention is CONFIRMED from qf.U2mzi (see mzi_2x2 docstring):

    T(theta, phi) = [[ e^{-i*phi} cos(theta),  -sin(theta) ],
                     [  sin(theta),              e^{i*phi} cos(theta) ]]

Every other part of this pipeline (data generation, physics loss, refinement)
goes through `build_mesh` / `build_mesh_lossy`, so it stays correct as long
as `mzi_2x2` matches hardware. An older literature placeholder is kept as
`_mzi_2x2_OLD_PLACEHOLDER_DO_NOT_USE` for diffing only.
"""
from __future__ import annotations

import torch

N_MODES = 4
BLOCK_LABELS = ["Ga23", "Ga12", "Ga34", "Gb23", "Gb12", "Gb34"]
# 1-indexed mode pairs -> 0-indexed matrix indices
_PAIR_OF = {"23": (1, 2), "12": (0, 1), "34": (2, 3)}
BLOCK_PAIRS = [_PAIR_OF[label[2:]] for label in BLOCK_LABELS]


def mzi_2x2(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """theta, phi: real tensors of shape (...,). Returns complex (..., 2, 2).

    CONFIRMED convention, derived exactly (not guessed) from your real
    qf.U2mzi source:

        U2mzi(theta, alpha, beta, chi) = exp(i*chi) * (
            U_PS(alpha+beta+theta-pi, theta-pi/2) @ U_BS(pi/4)
            @ U_PS(pi-2*theta, 0) @ U_BS(pi/4)
            @ U_PS(-beta, pi/2-alpha)
        )
        U_PS(a,b) = diag(e^{ia}, e^{ib})
        U_BS(x)   = [[cos x, i sin x],[i sin x, cos x]]

    called by datagen.py as U2mzi(theta, alpha=-phi, beta=pi/2, chi=0). Fully
    symbolically expanded (sympy) and verified to match the raw matrix
    product to machine precision (max diff ~6e-16 over 200 random points):

        T(theta, phi) = [[ e^{-i*phi} cos(theta),  -sin(theta) ],
                         [  sin(theta),              e^{i*phi} cos(theta) ]]

    Also re-verified full local rank (16/16 -- surjective onto U(4)) for this
    exact topology with this exact convention, same test as before. This
    REPLACES an earlier placeholder convention (kept below, commented out,
    for reference/diffing only) that was reverse-engineered from the
    literature because qf.U2mzi's source wasn't available yet -- that one is
    now known to be wrong (phase placement differs: it put phase on both
    off-diagonal entries and used a differential +-phi split, whereas the
    real hardware puts phase only on the diagonal, with plain real sin/cos
    off-diagonal terms). If anything was trained against the old convention,
    the physics loss still converged fine (it only needs *some* consistent
    param set), but the predicted params do NOT correspond to real hardware
    settings -- retrain, don't fine-tune.
    """
    c = torch.cos(theta).to(torch.complex64)
    s = torch.sin(theta).to(torch.complex64)
    ep = torch.complex(torch.cos(phi), torch.sin(phi))    # e^{+i phi}
    em = torch.complex(torch.cos(phi), -torch.sin(phi))   # e^{-i phi}
    m00 = em * c
    m01 = -s.to(torch.complex64)
    m10 = s.to(torch.complex64)
    m11 = ep * c
    row0 = torch.stack([m00, m01], dim=-1)
    row1 = torch.stack([m10, m11], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _mzi_2x2_OLD_PLACEHOLDER_DO_NOT_USE(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Superseded -- kept only so you can diff against it if you need to
    understand what changed. See mzi_2x2's docstring above."""
    c = torch.cos(theta).to(torch.complex64)
    s = torch.sin(theta).to(torch.complex64)
    ep = torch.complex(torch.cos(phi), torch.sin(phi))
    em = torch.complex(torch.cos(phi), -torch.sin(phi))
    m00 = ep * c
    m01 = -ep * s
    m10 = em * s
    m11 = em * c
    row0 = torch.stack([m00, m01], dim=-1)
    row1 = torch.stack([m10, m11], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def embed_block(block2x2: torch.Tensor, pair: tuple[int, int]) -> torch.Tensor:
    """Embed a batched 2x2 complex block into an NxN identity at (i,j)."""
    batch_shape = block2x2.shape[:-2]
    device = block2x2.device
    i, j = pair
    U = torch.eye(N_MODES, dtype=torch.complex64, device=device)
    U = U.expand(*batch_shape, N_MODES, N_MODES).clone()
    U[..., i, i] = block2x2[..., 0, 0]
    U[..., i, j] = block2x2[..., 0, 1]
    U[..., j, i] = block2x2[..., 1, 0]
    U[..., j, j] = block2x2[..., 1, 1]
    return U


def build_mesh(theta: torch.Tensor, phi: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    theta, phi: (..., 6) real tensors (one per MZI, ordered as BLOCK_LABELS)
    delta:      (..., 4) real tensors (output diagonal phases)
    Returns:    (..., 4, 4) complex64 unitary U = Ga23 Ga12 Ga34 Gb23 Gb12 Gb34 D
    """
    batch_shape = theta.shape[:-1]
    device = theta.device

    D_diag = torch.complex(torch.cos(delta), torch.sin(delta))  # (..., 4)
    U = torch.diag_embed(D_diag)  # (..., 4, 4)

    # Apply blocks right-to-left in the product, i.e. multiply onto U from the left
    # in reverse label order so the final product is Ga23 @ Ga12 @ ... @ D.
    for k in reversed(range(6)):
        block = mzi_2x2(theta[..., k], phi[..., k])
        G = embed_block(block, BLOCK_PAIRS[k])
        U = G @ U

    return U


# --------------------------------------------------------------------------- #
# Lossy forward model -- mirrors datagen.py's lossy_U2mzi / lossy_U4_circuit
# --------------------------------------------------------------------------- #
#
# datagen.py's real circuit, once loss < 1, is NOT the output of build_mesh
# above -- build_mesh's blocks are exactly unitary by construction (products
# of rotation-like 2x2s), and no amount of retraining lets a strictly-unitary
# function approximate a genuinely non-unitary target: the achievable output
# manifold (U(4), a 16-real-dim submanifold of the 32-real-dim space of all
# 4x4 complex matrices) simply does not contain lossy circuit matrices. What
# changes structurally in the lossy case is captured exactly by
# datagen.lossy_U2mzi:
#
#     U = qf.U2mzi(theta, alpha, beta, chi)
#     L_in  = diag(eta1, eta2)     # real, in [0,1], NOT unitary in general
#     L_out = diag(eta3, eta4)
#     return L_in @ U @ L_out
#
# i.e. each MZI's ideal 2x2 unitary core gets sandwiched between two real
# diagonal (possibly sub-unitary) loss matrices. `lossy_mzi_2x2` below
# reproduces that sandwich exactly, and the core (`mzi_2x2`) is now the
# CONFIRMED convention derived directly from your real qf.U2mzi source (see
# mzi_2x2's docstring) -- both pieces of build_mesh_lossy are exact now, not
# guessed.

def lossy_mzi_2x2(theta: torch.Tensor, phi: torch.Tensor, eta: torch.Tensor,
                   core_fn=None) -> torch.Tensor:
    """
    theta, phi: (...,) real. eta: (..., 4) real in [0,1] = (eta1,eta2,eta3,eta4)
    matching datagen.py's per-MZI loss row (eta1,eta2 = input-side port
    efficiencies, eta3,eta4 = output-side). Returns (..., 2, 2) complex,
    generally NON-unitary.
    """
    core_fn = core_fn or mzi_2x2
    U = core_fn(theta, phi)  # (..., 2, 2) complex, exactly unitary
    eta = eta.to(torch.float32)
    L_in = torch.diag_embed(eta[..., 0:2]).to(torch.complex64)
    L_out = torch.diag_embed(eta[..., 2:4]).to(torch.complex64)
    return L_in @ U @ L_out


def build_mesh_lossy(theta: torch.Tensor, phi: torch.Tensor, delta: torch.Tensor,
                      eta: torch.Tensor, core_fn=None) -> torch.Tensor:
    """
    theta, phi: (..., 6), delta: (..., 4) -- same as build_mesh.
    eta: (..., 6, 4) per-MZI loss rows, OR a python float / 0-dim tensor to
    broadcast the same efficiency to every port of every MZI (the common
    case: datagen.py's `sample_programmed_unitary_statistics(loss=...)`
    currently uses one global scalar for the whole dataset).
    Returns (..., 4, 4) complex, generally NON-unitary once any eta < 1.
    """
    batch_shape = theta.shape[:-1]
    device = theta.device

    if not torch.is_tensor(eta):
        eta = torch.tensor(float(eta), device=device)
    if eta.dim() == 0:
        eta = eta.expand(*batch_shape, 6, 4)
    eta = eta.to(device)

    D_diag = torch.complex(torch.cos(delta), torch.sin(delta))
    U = torch.diag_embed(D_diag)

    for k in reversed(range(6)):
        block = lossy_mzi_2x2(theta[..., k], phi[..., k], eta[..., k, :], core_fn=core_fn)
        G = embed_block(block, BLOCK_PAIRS[k])
        U = G @ U

    return U


# --------------------------------------------------------------------------- #
# Probe inputs + power-only observation model
# --------------------------------------------------------------------------- #
#
# datagen.py's probe basis: I1..I4 (computational basis) plus I12,I23,I34,I14
# (equal real superpositions of adjacent modes). Superpositions are unit-norm
# (`v / ||v||`). These match DEFAULT_PROBES below.
#
# These are all REAL superpositions (no relative i-phase between the two
# modes). That matters for how much of phi is recoverable from POWER
# measurements alone: comparing power(I_a), power(I_b), power((I_a+I_b)/sqrt2)
# lets you solve for Re(U[a,j] * conj(U[b,j])) for each output port j (the
# standard two-path-interference formula), which pins the relative phase
# between U[a,j] and U[b,j] up to a SIGN (cos(dphi) determines dphi only up
# to +-). A quadrature probe (I_a + i*I_b)/sqrt2 would additionally give
# Im(U[a,j]*conj(U[b,j])) and resolve that sign. Opt in via make_power_probes(
# quadrature=True) / Regime(quadrature=True) / datagen --quadrature.

PROBE_BASIS = torch.eye(N_MODES, dtype=torch.complex64)  # I1..I4, rows

_adj_pairs = [(0, 1), (1, 2), (2, 3), (0, 3)]  # matches I12, I23, I34, I14


def _real_superposition(i, j):
    v = torch.zeros(N_MODES, dtype=torch.complex64)
    v[i] = 1.0
    v[j] = 1.0
    return v / v.abs().square().sum().sqrt()  # correctly unit-normalized


def _quadrature_superposition(i, j):
    v = torch.zeros(N_MODES, dtype=torch.complex64)
    v[i] = 1.0
    v[j] = 1j
    return v / v.abs().square().sum().sqrt()


PROBE_SUPERPOSITION = torch.stack([_real_superposition(i, j) for i, j in _adj_pairs])   # (4,4)
PROBE_QUADRATURE = torch.stack([_quadrature_superposition(i, j) for i, j in _adj_pairs])  # (4,4)

N_BASIS_PROBES = 4
N_SUPERPOSITION_PROBES = 4
N_QUADRATURE_PROBES = 4
N_PROBES_DEFAULT = N_BASIS_PROBES + N_SUPERPOSITION_PROBES          # 8
N_PROBES_WITH_QUADRATURE = N_PROBES_DEFAULT + N_QUADRATURE_PROBES  # 12

# Probe order: I1..I4, I12,I23,I34,I14 [, I12q,I23q,I34q,I14q if quadrature]
_PROBE_LABELS_BASE = ["I1", "I2", "I3", "I4", "I12", "I23", "I34", "I14"]
_PROBE_LABELS_QUAD = ["I12q", "I23q", "I34q", "I14q"]


def make_power_probes(quadrature: bool = False, basis_only: bool = False) -> torch.Tensor:
    """Unit-norm row probes for power-mode observation.

    basis_only=True:  I1..I4 only → 4 probes, 16-D power vector.
                      Relative phases between matrix entries are largely
                      invisible; many (θ, φ) directions have zero obs gradient.

    basis_only=False (default):
      quadrature=False: 8 probes (basis + real superpositions), 32-D obs.
      quadrature=True:  12 probes (+ quadrature superpositions), 48-D obs.
    """
    if basis_only:
        if quadrature:
            raise ValueError("basis_only and quadrature are incompatible")
        return PROBE_BASIS.clone()
    parts = [PROBE_BASIS, PROBE_SUPERPOSITION]
    if quadrature:
        parts.append(PROBE_QUADRATURE)
    return torch.cat(parts, dim=0)


def power_obs_dim(quadrature: bool = False, basis_only: bool = False) -> int:
    return make_power_probes(quadrature, basis_only).shape[0] * N_MODES


def probe_labels(quadrature: bool = False, basis_only: bool = False) -> list[str]:
    if basis_only:
        return ["I1", "I2", "I3", "I4"]
    labels = list(_PROBE_LABELS_BASE)
    if quadrature:
        labels.extend(_PROBE_LABELS_QUAD)
    return labels


def power_row_width(quadrature: bool = False, basis_only: bool = False) -> int:
    """16 params + obs_dim for power-mode NPZ rows."""
    return 16 + power_obs_dim(quadrature, basis_only)


DEFAULT_PROBES = make_power_probes(quadrature=False, basis_only=False)  # (8,4)


def apply_probes(U: torch.Tensor, probes: torch.Tensor) -> torch.Tensor:
    """U: (..., 4, 4) complex. probes: (P, 4) complex, each row a normalized
    input state. Row-vector convention throughout this pipeline (matches
    datagen.py: output = probe @ U_circuit). Returns (..., P, 4) complex
    output fields, one per probe."""
    return torch.einsum('pi,...ij->...pj', probes.to(U.dtype), U)


def probe_powers(U: torch.Tensor, probes: torch.Tensor) -> torch.Tensor:
    """(..., P, 4) real, nonnegative output powers -- the only thing a plain
    photodetector actually measures. Structurally independent of `delta`
    (D is diagonal, unit-modulus, and applied last -- see module note above
    build_mesh_lossy) for ANY probes, not just these."""
    return apply_probes(U, probes).abs() ** 2


def haar_random_unitary(batch: int, n: int = N_MODES, device=None, generator=None) -> torch.Tensor:
    """Sample Haar-random U(n) matrices via QR of a complex Ginibre ensemble
    (Mezzadri 2007), fully vectorized over the batch dimension."""
    real = torch.randn(batch, n, n, device=device, generator=generator)
    imag = torch.randn(batch, n, n, device=device, generator=generator)
    Z = torch.complex(real, imag) / (2 ** 0.5)
    Q, R = torch.linalg.qr(Z)
    d = torch.diagonal(R, dim1=-2, dim2=-1)
    ph = d / d.abs()
    Q = Q * ph.unsqueeze(-2)
    return Q


def unitarity_error(U: torch.Tensor) -> torch.Tensor:
    n = U.shape[-1]
    I = torch.eye(n, dtype=U.dtype, device=U.device)
    return torch.linalg.matrix_norm(U.mH @ U - I, ord="fro")


if __name__ == "__main__":
    torch.manual_seed(0)
    theta = torch.rand(5, 6) * 2 * torch.pi
    phi = torch.rand(5, 6) * 2 * torch.pi
    delta = torch.rand(5, 4) * 2 * torch.pi
    U = build_mesh(theta, phi, delta)
    print("build_mesh output shape:", U.shape, U.dtype)
    print("max unitarity error:", unitarity_error(U).max().item())

    Uh = haar_random_unitary(2000)
    print("Haar sample unitarity error (max over 2000):", unitarity_error(Uh).max().item())

    # Local full-rank check: is build_mesh's Jacobian rank 16 (= dim U(4)) at a
    # generic point? This is a necessary condition for the parametrization to
    # be locally surjective onto U(4) (i.e. "universal").
    torch.manual_seed(0)
    theta_t = torch.rand(6) * 2 * torch.pi
    phi_t = torch.rand(6) * 2 * torch.pi
    delta_t = torch.rand(4) * 2 * torch.pi
    params = torch.cat([theta_t, phi_t, delta_t]).requires_grad_(True)

    def f(p):
        th, ph, de = p[:6], p[6:12], p[12:16]
        Uf = build_mesh(th.unsqueeze(0), ph.unsqueeze(0), de.unsqueeze(0))[0]
        return torch.cat([Uf.real.flatten(), Uf.imag.flatten()])

    J = torch.autograd.functional.jacobian(f, params)
    sv = torch.linalg.svdvals(J)
    rank = int((sv > 1e-4).sum().item())
    print(f"build_mesh Jacobian rank at a generic point: {rank}/16", "(full rank - good)" if rank == 16 else "(DEFICIENT)")