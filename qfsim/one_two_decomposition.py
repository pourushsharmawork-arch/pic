"""
one_two_decomposition.py
=====================

Robust decomposition of an arbitrary numerical 4 x 4 unitary into

    U = Ga23 @ Ga12 @ Ga34 @ Gb23 @ Gb12 @ Gb34 @ D,

where Ga12 and Ga34 act on disjoint mode pairs, as do Gb12 and Gb34.  Hence this is
equivalently

    U = Ga23 @ (Ga12 direct_sum Ga34) @ Gb23 @ (Gb12 direct_sum Gb34) @ D.

    
The file is called "sin2_decomposition" because the architecture looks like a     
The mode pairs are

    Ga23, Gb23 : (2, 3)
    Ga12, Gb12 : (1, 2)
    Ga34, Gb34 : (3, 4).

Each G is an embedded two-parameter MZI/Givens cell

    T(theta, phi)
      = [[exp(-i phi) cos(theta), -sin(theta)],
         [sin(theta),             exp(i phi) cos(theta)]].


The program uses right-sided column elimination, converts the eliminators into the mirrored physical order,
and commutes the leading diagonal phases to the output.  Exact zero-zero pivots
are handled by a finite backtracking search.  A numerical least-squares
fallback is available for pathological floating-point cases.

Dense NumPy-like inputs and SciPy sparse matrices are accepted.  Sparse 4 x 4
inputs are densified internally.  This standalone module does not import or
invoke the old 2121_decomposition.py.

Dependencies
------------
NumPy is required.
SciPy is required only if the numerical fallback is used.
"""

from __future__ import annotations

from typing import Any, Iterable
from . import misc
import numpy as np


# Right-sided eliminators.  Each tuple is
#   (label, first_mode, second_mode, target_row, component_to_zero).
# Mode indices are zero based internally.
_ELIMINATION_PLAN = (
    ("E1", 0, 1, 3, "first"),
    ("E2", 2, 3, 0, "second"),
    ("E3", 1, 2, 3, "first"),
    ("E4", 0, 1, 2, "first"),
    ("E5", 2, 3, 1, "second"),
    ("E6", 1, 2, 1, "second"),
)

# If U E1 ... E6 = D_in, then
# U = D_in E6^dagger ... E1^dagger.  The disjoint E4/E5 and E1/E2 pairs
# commute, producing the requested mirrored physical order.
_OUTPUT_PLAN = (
    ("Ga23", "E6", 1, 2),
    ("Ga12", "E4", 0, 1),
    ("Ga34", "E5", 2, 3),
    ("Gb23", "E3", 1, 2),
    ("Gb12", "E1", 0, 1),
    ("Gb34", "E2", 2, 3),
)


def _as_dense_matrix(matrix: Any) -> tuple[np.ndarray, bool]:
    """Return a complex dense array and whether the input was sparse-like."""
    input_was_sparse = bool(
        hasattr(matrix, "toarray") and callable(matrix.toarray)
    )

    if input_was_sparse:
        matrix = matrix.toarray()

    try:
        array = np.asarray(matrix, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError("Input must be a numerical matrix.") from exc

    return array, input_was_sparse


def wrap_phase(phi: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return misc.wrap_to_pi(phi)


def _unitarity_error(U: np.ndarray) -> float:
    """Return the larger left/right Frobenius unitarity error."""
    
    identity = np.eye(U.shape[0], dtype=complex)
    
    return float(
        max(
            np.linalg.norm(U.conj().T @ U - identity, ord="fro"),
            np.linalg.norm(U @ U.conj().T - identity, ord="fro"),
        )
    )



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

        [a, b] @ T(theta, phi) = [*, 0].

    When both entries vanish, the rotation is undetermined and the identity is
    returned.  The caller handles that degeneracy during backtracking.
    """
    if abs(a) <= tol and abs(b) <= tol:
        return 0.0, 0.0

    if abs(b) <= tol:
        return 0.0, 0.0

    if abs(a) <= tol:
        return np.pi / 2.0, 0.0

    theta = np.arctan2(abs(b), abs(a))
    phi = np.angle(a) - np.angle(b)

    return float(theta), wrap_phase(float(phi))


def zero_first_params(
    a: complex,
    b: complex,
    tol: float = 1e-14,
) -> tuple[float, float]:
    """
    Find theta and phi such that

        [a, b] @ T(theta, phi) = [0, *].

    When both entries vanish, the rotation is undetermined and the identity is
    returned.  The caller handles that degeneracy during backtracking.
    """
    if abs(a) <= tol and abs(b) <= tol:
        return 0.0, 0.0

    if abs(a) <= tol:
        return 0.0, 0.0

    if abs(b) <= tol:
        return np.pi / 2.0, 0.0

    theta = np.arctan2(abs(a), abs(b))
    phi = np.angle(a) - np.angle(b) + np.pi

    return float(theta), wrap_phase(float(phi))


def _make_block(
    label: str,
    i: int,
    j: int,
    theta: float,
    phi: float,
    **metadata: Any,
) -> dict[str, Any]:
    """Construct one block dictionary; all code paths use this implementation."""
    theta = float(theta)
    phi = wrap_phase(float(phi))
    block = {
        "label": label,
        "modes": (i + 1, j + 1),
        "theta": theta,
        "phi": phi,
        "T_full": misc.embed_T(4, i, j, misc.T_local(theta, phi)),
    }
    block.update(metadata)
    return block


def reconstruct_from_mirrored(
    blocks: list[dict[str, Any]],
    D: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct

        U = Ga23 Ga12 Ga34 Gb23 Gb12 Gb34 D

    from blocks in physical order.  Because Ga12/Ga34 and Gb12/Gb34 act on disjoint
    modes, this is the same as the two direct-sum layers.
    """
    D = np.asarray(D, dtype=complex)

    if D.shape != (4, 4):
        raise ValueError("D must be a 4 x 4 matrix.")

    U_rec = np.eye(4, dtype=complex)

    for block in blocks:
        T_full = np.asarray(block["T_full"], dtype=complex)
        if T_full.shape != (4, 4):
            raise ValueError("Every block T_full must be a 4 x 4 matrix.")
        U_rec = U_rec @ T_full

    return U_rec @ D


def _diagonalized_remainder(
    U: np.ndarray,
    blocks: list[dict[str, Any]],
) -> np.ndarray:
    """Return Gb34^dagger ... Ga23^dagger U for blocks in physical order."""
    W = np.asarray(U, dtype=complex).copy()

    for block in blocks:
        W = np.asarray(block["T_full"], dtype=complex).conj().T @ W

    return W


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

    U_rec = reconstruct_from_mirrored(blocks, D)

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
    candidates: Iterable[tuple[float, float]],
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
    """Generate admissible rotations when the nominal pivot is [0, 0]."""
    candidates: list[tuple[float, float]] = [
        (0.0, 0.0),
        (np.pi / 2.0, 0.0),
    ]

    for row in range(W.shape[0]):
        a = W[row, i]
        b = W[row, j]

        if np.hypot(abs(a), abs(b)) <= zero_tol:
            continue

        candidates.append(zero_first_params(a, b, tol=zero_tol))
        candidates.append(zero_second_params(a, b, tol=zero_tol))

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

    for _, i, j, row, which in plan:
        column = i if which == "first" else j
        values.append(abs(W[row, column]))

    return float(np.linalg.norm(values))


def _canonical_theta_with_right_phase(
    theta: float,
    phi: float,
) -> tuple[float, float, float]:
    """Map theta to [0, pi/2] and return the induced right phase."""
    reduced = float((theta + np.pi / 2.0) % np.pi - np.pi / 2.0)
    periods = int(np.rint((theta - reduced) / np.pi))
    phase_increment = float(periods * np.pi)

    endpoint_tol = 32.0 * np.finfo(float).eps

    if abs(reduced + np.pi / 2.0) <= endpoint_tol:
        reduced = np.pi / 2.0
        phase_increment -= np.pi

    if reduced < 0.0:
        reduced = -reduced
        phi = wrap_phase(phi - np.pi)
        phase_increment += np.pi

    if abs(reduced) <= endpoint_tol:
        reduced = 0.0

    if abs(reduced - np.pi / 2.0) <= endpoint_tol:
        reduced = np.pi / 2.0

    return float(reduced), wrap_phase(phi), phase_increment


def _canonicalize_chain(
    raw_blocks: list[dict[str, Any]],
    *,
    leading_phases: np.ndarray | None = None,
    trailing_phases: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Commute a leading diagonal through the chain and canonicalize theta."""
    phases = (
        np.zeros(4, dtype=float)
        if leading_phases is None
        else np.asarray(leading_phases, dtype=float).copy()
    )

    if phases.shape != (4,):
        raise ValueError("leading_phases must contain four values.")

    blocks: list[dict[str, Any]] = []

    for raw in raw_blocks:
        i, j = raw["zero_based_modes"]
        theta = float(raw["theta"])
        phi = wrap_phase(float(raw["phi"]) + phases[j] - phases[i])

        phases[i], phases[j] = phases[j], phases[i]

        theta, phi, increment = _canonical_theta_with_right_phase(theta, phi)
        phases[i] += increment
        phases[j] += increment
        phases = np.asarray([wrap_phase(value) for value in phases])

        blocks.append(
            _make_block(
                str(raw["label"]),
                i,
                j,
                theta,
                phi,
            )
        )

    if trailing_phases is not None:
        trailing_phases = np.asarray(trailing_phases, dtype=float)

        if trailing_phases.shape != (4,):
            raise ValueError("trailing_phases must contain four values.")

        phases += trailing_phases

    phases = np.asarray([wrap_phase(value) for value in phases])
    return blocks, np.diag(np.exp(1j * phases))


def _blocks_from_eliminators(
    eliminators: list[dict[str, Any]],
    diagonalized: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Convert right eliminators into the mirrored physical cell order."""
    by_label = {str(item["label"]): item for item in eliminators}
    expected = {item[0] for item in _ELIMINATION_PLAN}

    if set(by_label) != expected:
        raise RuntimeError("Column elimination returned incomplete cells.")

    raw_blocks: list[dict[str, Any]] = []

    for output_label, source_label, i, j in _OUTPUT_PLAN:
        source = by_label[source_label]
        raw_blocks.append(
            {
                "label": output_label,
                "zero_based_modes": (i, j),
                # T(theta, phi)^dagger = T(-theta, -phi).
                "theta": -float(source["theta"]),
                "phi": wrap_phase(-float(source["phi"])),
            }
        )

    diagonal_entries = np.diag(np.asarray(diagonalized, dtype=complex))

    if np.any(np.abs(diagonal_entries) == 0.0):
        raise RuntimeError("Column elimination produced a singular diagonal.")

    return _canonicalize_chain(
        raw_blocks,
        leading_phases=np.angle(diagonal_entries),
    )


def _parameters_from_blocks_and_D(
    blocks: list[dict[str, Any]],
    D: np.ndarray,
) -> np.ndarray:
    """Create the 16-parameter optimization vector for a mesh candidate."""
    parameters: list[float] = []

    for block in blocks:
        parameters.extend([block["theta"], block["phi"]])

    parameters.extend(np.angle(np.diag(D)).tolist())

    return np.asarray(parameters, dtype=float)


def _build_from_parameter_vector(
    parameters: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    """Build the full mirrored mesh from 12 MZI and four phase parameters."""
    parameters = np.asarray(parameters, dtype=float)

    if parameters.shape != (16,):
        raise ValueError("The parameter vector must have length 16.")

    blocks: list[dict[str, Any]] = []

    for k, (label, _, i, j) in enumerate(_OUTPUT_PLAN):
        blocks.append(
            _make_block(
                label,
                i,
                j,
                parameters[2 * k],
                parameters[2 * k + 1],
            )
        )

    D = np.diag(np.exp(1j * parameters[12:16]))
    return reconstruct_from_mirrored(blocks, D), blocks, D


def _numerical_fallback(
    U: np.ndarray,
    initial_parameters: np.ndarray | None,
    tol: float,
    n_restarts: int,
    seed: int | None,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, float]:
    """Solve the full 16-parameter mirrored decomposition numerically."""
    if n_restarts < 0:
        raise ValueError("n_restarts must be nonnegative.")

    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise RuntimeError(
            "The column-pivot search did not converge and SciPy is unavailable "
            "for the numerical fallback."
        ) from exc

    rng = np.random.default_rng(seed)
    starts: list[np.ndarray] = []

    if initial_parameters is not None:
        starts.append(np.asarray(initial_parameters, dtype=float))

    for _ in range(n_restarts):
        starts.append(rng.uniform(-np.pi, np.pi, size=16))

    if not starts:
        starts.append(np.zeros(16, dtype=float))

    least_squares_tol = max(float(tol), 1e-14)
    acceptance_tol = max(100.0 * float(tol), 1e-10)

    def residual(parameters: np.ndarray) -> np.ndarray:
        U_trial, _, _ = _build_from_parameter_vector(parameters)
        difference = U_trial - U

        return np.concatenate(
            [difference.real.ravel(), difference.imag.ravel()]
        )

    best_error = np.inf
    best_parameters: np.ndarray | None = None

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
            best_parameters = solution.x.copy()

        if error <= acceptance_tol:
            break

    if best_parameters is None or best_error > acceptance_tol:
        raise RuntimeError(
            "Could not obtain a verified six-MZI mirrored decomposition. "
            f"Best numerical residual: {best_error:.3e}."
        )

    _, raw_blocks, raw_D = _build_from_parameter_vector(best_parameters)
    canonical_input = [
        {
            "label": block["label"],
            "zero_based_modes": (
                block["modes"][0] - 1,
                block["modes"][1] - 1,
            ),
            "theta": block["theta"],
            "phi": block["phi"],
        }
        for block in raw_blocks
    ]
    blocks, D = _canonicalize_chain(
        canonical_input,
        trailing_phases=np.angle(np.diag(raw_D)),
    )
    W = _diagonalized_remainder(U, blocks)

    return blocks, D, W, best_error


def decompose_U4_rectangular(
    U: Any,
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

        U = Ga23 Ga12 Ga34 Gb23 Gb12 Gb34 D
          = Ga23 (Ga12 direct_sum Ga34) Gb23 (Gb12 direct_sum Gb34) D.

    Parameters
    ----------
    U:
        The target dense or sparse 4 x 4 unitary.

    tol:
        Numerical tolerance used for validation and elimination.

    max_search_nodes:
        Maximum number of branches explored when exact zero-zero pivots occur.

    numerical_fallback:
        Use a full least-squares solve if the column-pivot search does not reach
        the requested tolerance.

    n_restarts:
        Number of random starts used by the numerical fallback.

    seed:
        Reproducibility seed for the numerical fallback only.

    return_diagnostics:
        When False, return ``blocks, D, W``.  When True, return
        ``blocks, D, W, diagnostics``.

    Returns
    -------
    blocks:
        Six dictionaries containing labels, one-indexed modes, theta, phi, and
        the embedded 4 x 4 cell, in physical order Ga23 through Gb34.

    D:
        The final diagonal phase matrix.

    W:
        The matrix ``Gb34^dagger ... Ga23^dagger U``.  A successful decomposition
        has W approximately equal to D.

    diagnostics:
        Returned only when ``return_diagnostics=True``.

    Notes
    -----
    Six two-parameter MZIs contain 12 real parameters, whereas U(4) contains
    16.  The four phases in D are therefore necessary.  To write the result
    with no explicit D, absorb its first and last 2 x 2 diagonal blocks into
    Gb12 and Gb34, respectively; those two factors then become general U(2) cells.
    """
    U, input_was_sparse = _as_dense_matrix(U)

    if U.shape != (4, 4):
        raise ValueError("Input must be a 4 x 4 matrix.")

    if not np.all(np.isfinite(U)):
        raise ValueError("Input contains NaN or infinite values.")

    if tol <= 0:
        raise ValueError("tol must be positive.")

    if max_search_nodes < 1:
        raise ValueError("max_search_nodes must be at least 1.")

    unitarity_error = _unitarity_error(U)
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
        V: np.ndarray,
        eliminators: list[dict[str, Any]],
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
                    V - np.diag(np.diag(V)),
                    ord="fro",
                )
            )

            if offdiagonal_error < best_offdiagonal_error:
                best_offdiagonal_error = offdiagonal_error
                best_state = (eliminators.copy(), V.copy())

            if offdiagonal_error <= acceptance_tol:
                return eliminators.copy(), V.copy()

            return None

        label, i, j, row, which = _ELIMINATION_PLAN[step]
        a = V[row, i]
        b = V[row, j]

        if abs(a) <= zero_tol and abs(b) <= zero_tol:
            candidates = _free_rotation_candidates(
                V,
                i,
                j,
                zero_tol=zero_tol,
            )
        elif which == "first":
            candidates = [zero_first_params(a, b, tol=zero_tol)]
        else:
            candidates = [zero_second_params(a, b, tol=zero_tol)]

        scored_candidates = []

        for theta, phi in candidates:
            eliminator = _make_block(
                label,
                i,
                j,
                theta,
                phi,
            )
            V_next = V @ eliminator["T_full"]

            score = (
                _nominal_target_residual(V_next, step + 1)
                + 0.05 * _nominal_target_residual(V_next)
            )

            scored_candidates.append((score, eliminator, V_next))

        scored_candidates.sort(key=lambda item: item[0])

        for _, eliminator, V_next in scored_candidates:
            result = recurse(
                step + 1,
                V_next,
                eliminators + [eliminator],
            )

            if result is not None:
                return result

        return None

    search_result = recurse(
        step=0,
        V=U.copy(),
        eliminators=[],
    )

    used_numerical_fallback = False
    fallback_residual = 0.0

    if search_result is not None:
        eliminators, diagonalized = search_result
        blocks, D = _blocks_from_eliminators(eliminators, diagonalized)
        W = _diagonalized_remainder(U, blocks)
    else:
        if not numerical_fallback:
            raise RuntimeError(
                "The column-pivot search did not find a verified decomposition. "
                f"Best off-diagonal residual: {best_offdiagonal_error:.3e}."
            )

        initial_parameters = None

        if best_state is not None:
            approximate_blocks, approximate_D = _blocks_from_eliminators(
                best_state[0],
                best_state[1],
            )
            initial_parameters = _parameters_from_blocks_and_D(
                approximate_blocks,
                approximate_D,
            )

        blocks, D, W, fallback_residual = _numerical_fallback(
            U,
            initial_parameters=initial_parameters,
            tol=tol,
            n_restarts=n_restarts,
            seed=seed,
        )

        used_numerical_fallback = True

    diagnostics = decomposition_diagnostics(U, blocks, D, W)
    reconstruction_error = diagnostics["reconstruction_error"]

    if reconstruction_error > acceptance_tol:
        raise RuntimeError(
            "A candidate decomposition was found but failed verification. "
            f"Reconstruction error: {reconstruction_error:.3e}."
        )

    diagnostics.update(
        {
            "success": True,
            "unitarity_error_of_input": unitarity_error,
            "input_was_sparse": input_was_sparse,
            "nodes_visited": nodes_visited,
            "used_numerical_fallback": used_numerical_fallback,
            "fallback_residual": float(fallback_residual),
        }
    )

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

    test_matrices: list[tuple[str, Any]] = [
        ("identity", np.eye(4, dtype=complex)),
        (
            "CNOT",
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ],
                dtype=complex,
            ),
        ),
    ]

    for permutation in permutations(range(4)):
        test_matrices.append(
            (
                f"permutation_{permutation}",
                _permutation_unitary(permutation),
            )
        )

    for seed in range(50):
        test_matrices.append((f"Haar_{seed}", misc.random_unitary(4, seed)))

    rng = np.random.default_rng(12345)

    for test_index in range(100):
        parameters = np.empty(16, dtype=float)
        parameters[:12:2] = rng.choice(
            [0.0, np.pi / 4.0, np.pi / 2.0],
            size=6,
        )
        parameters[1:12:2] = rng.choice(
            [-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0],
            size=6,
        )
        parameters[12:] = rng.uniform(-np.pi, np.pi, size=4)
        U_mesh, _, _ = _build_from_parameter_vector(parameters)
        test_matrices.append((f"boundary_{test_index}", U_mesh))

    try:
        from scipy.sparse import csr_matrix

        test_matrices.append(("sparse_CNOT", csr_matrix(test_matrices[1][1])))
    except ImportError:
        pass

    worst_error = 0.0

    for name, U_test in test_matrices:
        U_dense, _ = _as_dense_matrix(U_test)
        blocks, D, W, diagnostics = decompose_U4_rectangular(
            U_test,
            return_diagnostics=True,
        )
        error = float(
            np.linalg.norm(
                U_dense - reconstruct_from_mirrored(blocks, D),
                ord="fro",
            )
        )
        worst_error = max(worst_error, error)

        if error > 1e-9 or not diagnostics["success"]:
            raise AssertionError(f"Self-test failed for {name}: {error:.3e}")

        if not np.allclose(W, D, atol=1e-9, rtol=0.0):
            raise AssertionError(f"W != D for {name}.")

    print(
        f"All {len(test_matrices)} mirrored decomposition tests passed. "
        f"Worst Frobenius reconstruction error: {worst_error:.3e}."
    )


if __name__ == "__main__":
    self_test()
