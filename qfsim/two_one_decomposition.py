"""
two_one_decomposition.py
========================

Robust decomposition of an arbitrary numerical 4 x 4 unitary into

    U = Ta12 @ Ta34 @ Ta23 @ Tb12 @ Tb34 @ Tb23 @ D,

where each T is an embedded two-parameter MZI/Givens cell

    T(theta, phi)
      = [[exp(-i phi) cos(theta), -sin(theta)],
         [sin(theta),             exp(i phi) cos(theta)]],

and D is diagonal.

The routine handles both dense and sparse unitaries.  The important difference
from a no-pivot elimination is that an exact zero-zero pivot is treated as an
underdetermined Givens rotation.  The code searches over admissible rotations
that use the remaining matrix structure, rather than automatically choosing
the identity.

A numerical least-squares fallback is included for pathological floating-point
cases.  Every returned decomposition is explicitly reconstructed and checked.

Dependencies
------------
NumPy is required.
SciPy is required only if the numerical fallback is used.
"""

from __future__ import annotations

from typing import Any

from . import misc 

import numpy as np


# Physical order of the six MZIs.
#
# Each tuple is
#   (label, first_mode, second_mode, target_column, component_to_zero)
#
# Mode indices are zero based internally.
_ELIMINATION_PLAN = (
    ("Ta12", 0, 1, 3, "first"),
    ("Ta34", 2, 3, 0, "second"),
    ("Ta23", 1, 2, 3, "first"),
    ("Tb12", 0, 1, 2, "first"),
    ("Tb34", 2, 3, 1, "second"),
    ("Tb23", 1, 2, 1, "second"),
)


def wrap_phase(phi: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return misc.wrap_to_pi(phi)



# def T_local(theta: float, phi: float) -> np.ndarray:
#     """
#     Return the two-mode MZI/Givens cell

#         [[exp(-i phi) cos(theta), -sin(theta)],
#          [sin(theta),             exp(i phi) cos(theta)]].
#     """
#     c = np.cos(theta)
#     s = np.sin(theta)

#     return np.array(
#         [
#             [np.exp(-1j * phi) * c, -s],
#             [s, np.exp(1j * phi) * c],
#         ],
#         dtype=complex,
#     )


# def embed_T(
#     N: int,
#     i: int,
#     j: int,
#     T2: np.ndarray,
# ) -> np.ndarray:
#     """Embed a 2 x 2 cell into modes i and j of an N-mode identity."""
#     if not (0 <= i < j < N):
#         raise ValueError("Require 0 <= i < j < N.")

#     T2 = np.asarray(T2, dtype=complex)

#     if T2.shape != (2, 2):
#         raise ValueError("T2 must be a 2 x 2 matrix.")

#     result = np.eye(N, dtype=complex)
#     result[np.ix_([i, j], [i, j])] = T2
#     return result


def zero_second_params(
    a: complex,
    b: complex,
    tol: float = 1e-14,
) -> tuple[float, float]:
    """
    Find theta and phi such that

        T(theta, phi)^dagger @ [a, b]^T = [*, 0]^T.

    When both a and b vanish, the rotation is not determined.  The caller must
    handle that degeneracy; this function returns the identity in that case.
    """
    if abs(a) <= tol and abs(b) <= tol:
        return 0.0, 0.0

    if abs(b) <= tol:
        return 0.0, 0.0

    if abs(a) <= tol:
        return np.pi / 2.0, 0.0

    theta = np.arctan2(abs(b), abs(a))
    phi = np.angle(b) - np.angle(a)

    return float(theta), wrap_phase(float(phi))


def zero_first_params(
    a: complex,
    b: complex,
    tol: float = 1e-14,
) -> tuple[float, float]:
    """
    Find theta and phi such that

        T(theta, phi)^dagger @ [a, b]^T = [0, *]^T.

    When both a and b vanish, the rotation is not determined.  The caller must
    handle that degeneracy; this function returns the identity in that case.
    """
    if abs(a) <= tol and abs(b) <= tol:
        return 0.0, 0.0

    if abs(a) <= tol:
        return 0.0, 0.0

    if abs(b) <= tol:
        return np.pi / 2.0, 0.0

    theta = np.arctan2(abs(a), abs(b))
    phi = np.angle(b) - np.angle(a) + np.pi

    return float(theta), wrap_phase(float(phi))


def reconstruct_from_rectangular(
    blocks: list[dict[str, Any]],
    D: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct

        U = Ta12 Ta34 Ta23 Tb12 Tb34 Tb23 D.
    """
    D = np.asarray(D, dtype=complex)

    if D.shape != (4, 4):
        raise ValueError("D must be a 4 x 4 matrix.")

    U_rec = np.eye(4, dtype=complex)

    for block in blocks:
        U_rec = U_rec @ block["T_full"]

    return U_rec @ D


def decomposition_diagnostics(
    U: np.ndarray,
    blocks: list[dict[str, Any]],
    D: np.ndarray,
    W: np.ndarray,
) -> dict[str, float | bool]:
    """Return reconstruction, diagonalization, and unitarity diagnostics."""
    U = np.asarray(U, dtype=complex)
    D = np.asarray(D, dtype=complex)
    W = np.asarray(W, dtype=complex)

    U_rec = reconstruct_from_rectangular(blocks, D)

    reconstruction_error = float(np.linalg.norm(U - U_rec, ord="fro"))
    offdiagonal_error = float(
        np.linalg.norm(W - np.diag(np.diag(W)), ord="fro")
    )
    D_unitarity_error = float(
        np.linalg.norm(D.conj().T @ D - np.eye(4), ord="fro")
    )

    return {
        "reconstruction_error": reconstruction_error,
        "offdiagonal_error": offdiagonal_error,
        "D_unitarity_error": D_unitarity_error,
        "success": reconstruction_error <= 1e-9,
    }


def _deduplicate_rotations(
    candidates: list[tuple[float, float]],
    matrix_tol: float = 1e-10,
) -> list[tuple[float, float]]:
    """Remove candidates that produce numerically identical 2 x 2 cells."""
    unique_parameters: list[tuple[float, float]] = []
    unique_matrices: list[np.ndarray] = []

    for theta, phi in candidates:
        theta = float(theta)
        phi = wrap_phase(float(phi))
        candidate_matrix = misc.T_local(theta, phi)

        duplicate = any(
            np.linalg.norm(candidate_matrix - old, ord="fro") <= matrix_tol
            for old in unique_matrices
        )

        if not duplicate:
            unique_parameters.append((theta, phi))
            unique_matrices.append(candidate_matrix)

    return unique_parameters


def _free_rotation_candidates(
    W: np.ndarray,
    i: int,
    j: int,
    zero_tol: float,
) -> list[tuple[float, float]]:
    """
    Generate admissible rotations when the nominal elimination pair is [0, 0].

    In that situation the current elimination condition imposes no restriction
    on the MZI.  The useful choices are obtained by aligning another nonzero
    two-component column vector in the same row pair with either coordinate
    axis.  Canonical identity, swap, and balanced rotations are also included.
    """
    candidates: list[tuple[float, float]] = [
        (0.0, 0.0),
        (np.pi / 2.0, 0.0),
    ]

    for column in range(W.shape[1]):
        a = W[i, column]
        b = W[j, column]

        if np.hypot(abs(a), abs(b)) <= zero_tol:
            continue

        candidates.append(zero_first_params(a, b, tol=zero_tol))
        candidates.append(zero_second_params(a, b, tol=zero_tol))

    # Additional canonical choices are inexpensive for a 4 x 4 problem and
    # improve robustness for highly symmetric matrices.
    for phi in (0.0, np.pi / 2.0, -np.pi / 2.0, np.pi):
        candidates.append((np.pi / 4.0, phi))

    return _deduplicate_rotations(candidates)


def _nominal_target_residual(
    W: np.ndarray,
    number_of_targets: int | None = None,
) -> float:
    """Heuristic residual used only to order backtracking candidates."""
    plan = (
        _ELIMINATION_PLAN
        if number_of_targets is None
        else _ELIMINATION_PLAN[:number_of_targets]
    )

    values: list[float] = []

    for _, i, j, column, which in plan:
        row = i if which == "first" else j
        values.append(abs(W[row, column]))

    return float(np.linalg.norm(values))


def _parameters_from_blocks_and_W(
    blocks: list[dict[str, Any]],
    W: np.ndarray,
) -> np.ndarray:
    """Create a 16-parameter optimization vector from an approximate branch."""
    parameters: list[float] = []

    for block in blocks:
        parameters.extend([block["theta"], block["phi"]])

    parameters.extend(np.angle(np.diag(W)).tolist())

    return np.asarray(parameters, dtype=float)


def _build_from_parameter_vector(
    parameters: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    """Build the full mesh from 12 MZI parameters and four diagonal phases."""
    parameters = np.asarray(parameters, dtype=float)

    if parameters.shape != (16,):
        raise ValueError("The parameter vector must have length 16.")

    U_mesh = np.eye(4, dtype=complex)
    blocks: list[dict[str, Any]] = []

    for k, (label, i, j, _, _) in enumerate(_ELIMINATION_PLAN):
        theta = float(parameters[2 * k])
        phi = wrap_phase(float(parameters[2 * k + 1]))

        T_full = misc.embed_T(4, i, j, misc.T_local(theta, phi))
        U_mesh = U_mesh @ T_full

        blocks.append(
            {
                "label": label,
                "modes": (i + 1, j + 1),
                "theta": theta,
                "phi": phi,
                "T_full": T_full,
            }
        )

    D = np.diag(np.exp(1j * parameters[12:16]))
    return U_mesh @ D, blocks, D


def _numerical_fallback(
    U: np.ndarray,
    initial_parameters: np.ndarray | None,
    tol: float,
    n_restarts: int,
    seed: int | None,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, float]:
    """
    Solve the full 16-parameter decomposition numerically.

    This is used only when the exact zero-pivot backtracking does not reach the
    requested tolerance.  The variables are intentionally left unbounded
    because the MZI parameterization is periodic.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise RuntimeError(
            "The sparse-pivot search did not converge and SciPy is unavailable "
            "for the numerical fallback."
        ) from exc

    rng = np.random.default_rng(seed)
    starts: list[np.ndarray] = []

    if initial_parameters is not None:
        starts.append(np.asarray(initial_parameters, dtype=float))

    for _ in range(n_restarts):
        starts.append(rng.uniform(-np.pi, np.pi, size=16))

    least_squares_tol = max(float(tol), 1e-14)
    acceptance_tol = max(100.0 * float(tol), 1e-10)

    def residual(parameters: np.ndarray) -> np.ndarray:
        U_trial, _, _ = _build_from_parameter_vector(parameters)
        difference = U_trial - U

        return np.concatenate(
            [difference.real.ravel(), difference.imag.ravel()]
        )

    best_error = np.inf
    best_solution = None

    for start in starts:
        solution = least_squares(
            residual,
            start,
            method="trf",
            xtol=least_squares_tol,
            ftol=least_squares_tol,
            gtol=least_squares_tol,
            max_nfev=20_000,
        )

        error = float(np.linalg.norm(residual(solution.x)))

        if error < best_error:
            best_error = error
            best_solution = solution

        if error <= acceptance_tol:
            break

    if best_solution is None or best_error > acceptance_tol:
        raise RuntimeError(
            "Could not obtain a verified six-MZI decomposition. "
            f"Best numerical residual: {best_error:.3e}."
        )

    _, blocks, D = _build_from_parameter_vector(best_solution.x)

    W = U.copy()
    for block in blocks:
        W = block["T_full"].conj().T @ W

    return blocks, D, W, best_error


def decompose_U4_rectangular(
    U: np.ndarray,
    tol: float = 1e-12,
    *,
    max_search_nodes: int = 100_000,
    numerical_fallback: bool = True,
    n_restarts: int = 16,
    seed: int | None = 0,
    return_diagnostics: bool = False,
):
    """
    Decompose an arbitrary numerical 4 x 4 unitary as

        U = Ta12 Ta34 Ta23 Tb12 Tb34 Tb23 D.

    Parameters
    ----------
    U:
        The target 4 x 4 unitary.

    tol:
        Numerical tolerance used for validation and elimination.

    max_search_nodes:
        Maximum number of branches explored when exact zero-zero pivots occur.

    numerical_fallback:
        Use a full least-squares solve if the sparse-pivot search does not reach
        the requested tolerance.

    n_restarts:
        Number of random starts used by the numerical fallback.

    seed:
        Reproducibility seed for the numerical fallback only.

    return_diagnostics:
        When False, return ``blocks, D, W`` for compatibility with the original
        function.  When True, return ``blocks, D, W, diagnostics``.

    Returns
    -------
    blocks:
        Six dictionaries containing labels, one-indexed modes, theta, phi, and
        the embedded 4 x 4 cell.

    D:
        The final diagonal matrix.

    W:
        The matrix obtained after left-multiplying U by the six cell adjoints.
        A successful decomposition has W approximately equal to D.

    diagnostics:
        Returned only when ``return_diagnostics=True``.

    Notes
    -----
    Dense unitaries normally follow one deterministic elimination path.  Sparse
    unitaries can produce a nominal pivot [0, 0].  That condition leaves the
    corresponding MZI undetermined, so selecting the identity is not generally
    valid.  This implementation searches over rotations compatible with the
    remaining row-pair structure.
    """
    U = np.asarray(U, dtype=complex)

    if U.shape != (4, 4):
        raise ValueError("Input must be a 4 x 4 matrix.")

    if tol <= 0:
        raise ValueError("tol must be positive.")

    unitarity_error = float(
        np.linalg.norm(U.conj().T @ U - np.eye(4), ord="fro")
    )
    allowed_unitarity_error = max(100.0 * tol, 1e-10)

    if unitarity_error > allowed_unitarity_error:
        raise ValueError(
            "Input is not unitary within tolerance. "
            f"Frobenius unitarity error: {unitarity_error:.3e}."
        )

    zero_tol = max(10.0 * tol, 1e-14)
    acceptance_tol = max(100.0 * tol, 1e-10)

    nodes_visited = 0
    best_offdiagonal_error = np.inf
    best_state: tuple[list[dict[str, Any]], np.ndarray] | None = None

    def recurse(
        step: int,
        W: np.ndarray,
        blocks: list[dict[str, Any]],
    ):
        nonlocal nodes_visited
        nonlocal best_offdiagonal_error
        nonlocal best_state

        nodes_visited += 1

        if nodes_visited > max_search_nodes:
            return None

        if step == len(_ELIMINATION_PLAN):
            offdiagonal_error = float(
                np.linalg.norm(
                    W - np.diag(np.diag(W)),
                    ord="fro",
                )
            )

            if offdiagonal_error < best_offdiagonal_error:
                best_offdiagonal_error = offdiagonal_error
                best_state = (blocks.copy(), W.copy())

            if offdiagonal_error <= acceptance_tol:
                return blocks.copy(), W.copy()

            return None

        label, i, j, column, which = _ELIMINATION_PLAN[step]
        a = W[i, column]
        b = W[j, column]

        if abs(a) <= zero_tol and abs(b) <= zero_tol:
            candidates = _free_rotation_candidates(
                W,
                i,
                j,
                zero_tol=zero_tol,
            )
        elif which == "first":
            candidates = [
                zero_first_params(a, b, tol=zero_tol)
            ]
        else:
            candidates = [
                zero_second_params(a, b, tol=zero_tol)
            ]

        scored_candidates = []

        for theta, phi in candidates:
            T_full = misc.embed_T(4, i, j, misc.T_local(theta, phi))
            W_next = T_full.conj().T @ W

            # Earlier targets are weighted more strongly because no later cell
            # should need to repair them.
            score = (
                _nominal_target_residual(W_next, step + 1)
                + 0.05 * _nominal_target_residual(W_next)
            )

            scored_candidates.append(
                (score, theta, phi, T_full, W_next)
            )

        scored_candidates.sort(key=lambda item: item[0])

        for _, theta, phi, T_full, W_next in scored_candidates:
            next_block = {
                "label": label,
                "modes": (i + 1, j + 1),
                "theta": float(theta),
                "phi": float(phi),
                "T_full": T_full,
            }

            result = recurse(
                step + 1,
                W_next,
                blocks + [next_block],
            )

            if result is not None:
                return result

        return None

    search_result = recurse(
        step=0,
        W=U.copy(),
        blocks=[],
    )

    used_numerical_fallback = False
    fallback_residual = 0.0

    if search_result is not None:
        blocks, W = search_result
        D = np.diag(np.diag(W))
    else:
        if not numerical_fallback:
            raise RuntimeError(
                "The sparse-pivot search did not find a verified decomposition. "
                f"Best off-diagonal residual: "
                f"{best_offdiagonal_error:.3e}."
            )

        initial_parameters = None

        if best_state is not None:
            initial_parameters = _parameters_from_blocks_and_W(
                best_state[0],
                best_state[1],
            )

        blocks, D, W, fallback_residual = _numerical_fallback(
            U,
            initial_parameters=initial_parameters,
            tol=tol,
            n_restarts=n_restarts,
            seed=seed,
        )

        used_numerical_fallback = True

    U_reconstructed = reconstruct_from_rectangular(blocks, D)
    reconstruction_error = float(
        np.linalg.norm(U - U_reconstructed, ord="fro")
    )
    offdiagonal_error = float(
        np.linalg.norm(W - np.diag(np.diag(W)), ord="fro")
    )
    D_unitarity_error = float(
        np.linalg.norm(D.conj().T @ D - np.eye(4), ord="fro")
    )

    if reconstruction_error > acceptance_tol:
        raise RuntimeError(
            "A candidate decomposition was found but failed verification. "
            f"Reconstruction error: {reconstruction_error:.3e}."
        )

    diagnostics = {
        "success": True,
        "reconstruction_error": reconstruction_error,
        "offdiagonal_error": offdiagonal_error,
        "D_unitarity_error": D_unitarity_error,
        "unitarity_error_of_input": unitarity_error,
        "nodes_visited": nodes_visited,
        "used_numerical_fallback": used_numerical_fallback,
        "fallback_residual": float(fallback_residual),
    }

    if return_diagnostics:
        return blocks, D, W, diagnostics

    return blocks, D, W


def _permutation_unitary(permutation: tuple[int, ...]) -> np.ndarray:
    """Construct a permutation matrix from a column-to-row permutation."""
    result = np.zeros((4, 4), dtype=complex)

    for column, row in enumerate(permutation):
        result[row, column] = 1.0

    return result


def self_test() -> None:
    """Run dense, sparse, permutation, and boundary-angle consistency tests."""
    from itertools import permutations

    test_matrices: list[tuple[str, np.ndarray]] = []

    identity = np.eye(4, dtype=complex)

    cnot = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=complex,
    )

    test_matrices.extend(
        [
            ("identity", identity),
            ("CNOT", cnot),
        ]
    )

    # Every 4 x 4 permutation matrix.
    for permutation in permutations(range(4)):
        test_matrices.append(
            (
                f"permutation_{permutation}",
                _permutation_unitary(permutation),
            )
        )

    # Dense Haar unitaries.
    for seed in range(50):
        test_matrices.append(
            (f"Haar_{seed}", misc.random_unitary(4, seed=seed))
        )

    # Unitaries generated from mesh parameters lying on singular boundaries.
    rng = np.random.default_rng(12345)

    for test_index in range(100):
        U_mesh = np.eye(4, dtype=complex)

        for label, i, j, _, _ in _ELIMINATION_PLAN:
            del label

            theta = float(
                rng.choice([0.0, np.pi / 4.0, np.pi / 2.0])
            )
            phi = float(
                rng.choice(
                    [
                        -np.pi,
                        -np.pi / 2.0,
                        0.0,
                        np.pi / 2.0,
                    ]
                )
            )

            U_mesh = U_mesh @ misc.embed_T(
                4,
                i,
                j,
                misc.T_local(theta, phi),
            )

        D = np.diag(
            np.exp(1j * rng.uniform(-np.pi, np.pi, size=4))
        )

        test_matrices.append(
            (f"boundary_mesh_{test_index}", U_mesh @ D)
        )

    for name, U in test_matrices:
        blocks, D, W, diagnostics = decompose_U4_rectangular(
            U,
            return_diagnostics=True,
        )

        U_reconstructed = reconstruct_from_rectangular(blocks, D)

        if not np.allclose(
            U,
            U_reconstructed,
            atol=1e-9,
            rtol=0.0,
        ):
            raise AssertionError(
                f"Decomposition failed for {name}. "
                f"Diagnostics: {diagnostics}"
            )

        if diagnostics["reconstruction_error"] > 1e-9:
            raise AssertionError(
                f"Large reconstruction error for {name}: "
                f"{diagnostics['reconstruction_error']:.3e}"
            )

    print(
        f"All {len(test_matrices)} decomposition tests passed."
    )


if __name__ == "__main__":
    self_test()

    CNOT = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=complex,
    )

    blocks, D, W, diagnostics = decompose_U4_rectangular(
        CNOT,
        return_diagnostics=True,
    )

    print("\nCNOT decomposition")
    print("-------------------")

    for block in blocks:
        print(
            f"{block['label']}: "
            f"modes={block['modes']}, "
            f"theta={block['theta']:.12f}, "
            f"phi={block['phi']:.12f}"
        )

    print("\nD =")
    print(np.round(D, 12))

    print("\nDiagnostics =")
    print(diagnostics)
