from typing import Any
import numpy as np

from qfsim.one_two_hardware_decomposition import (_as_dense_matrix, _unitarity_error, _ELIMINATION_PLAN, _make_block,
                                         zero_first_params, _nominal_target_residual, _free_rotation_candidates, zero_second_params,
                                         _numerical_fallback, decomposition_diagnostics, _blocks_from_eliminators, _diagonalized_remainder,
                                         _parameters_from_blocks_and_D) 

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