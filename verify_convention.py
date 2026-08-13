"""
The mzi_2x2 convention question this file was built to answer is now
SETTLED: derived directly (sympy expansion + numeric verification, max diff
~6e-16) from your real qf.U2mzi source, not guessed. See mesh_forward.mzi_2x2's
docstring for the derivation. mesh_forward.py has been updated to use it.

What this file is still useful for:
  1. A regression check -- confirm your real data actually matches
     ('confirmed', 'forward', 'row_reblock') [exact field] or
     ('confirmed', 'forward') [power] before trusting a training run,
     including lossy data (pass eta= the generation loss).
  2. Catching *layout* mistakes (row vs column, interleaved vs block) if you
     ever change datagen.py's probe/readout structure again.

Old candidate mzi variants (differential, single_port, etc.) are kept below,
clearly marked, purely so you can diff against them if something that used
to train under the old guessed convention needs comparing -- they are NOT
expected to match anything anymore.
"""
from __future__ import annotations

import torch

from mesh_forward import (
    BLOCK_PAIRS,
    DEFAULT_PROBES,
    embed_block,
    lossy_mzi_2x2,
    make_power_probes,
    mzi_2x2 as _confirmed_mzi_2x2,
    probe_powers,
)


# --------------------------------------------------------------------------- #
# Candidate mzi_2x2 conventions
# --------------------------------------------------------------------------- #

def _mzi_confirmed(theta, phi):
    """THE real convention -- derived from your qf.U2mzi source, see
    mesh_forward.mzi_2x2's docstring. This should be the match."""
    return _confirmed_mzi_2x2(theta, phi)


def _mzi_differential(theta, phi):
    """SUPERSEDED placeholder (kept for diffing only -- see module docstring).
    diag(e^{i phi}, e^{-i phi}) @ R(theta)."""
    c = torch.cos(theta).to(torch.complex64)
    s = torch.sin(theta).to(torch.complex64)
    ep = torch.complex(torch.cos(phi), torch.sin(phi))
    em = torch.complex(torch.cos(phi), -torch.sin(phi))
    row0 = torch.stack([ep * c, -ep * s], dim=-1)
    row1 = torch.stack([em * s, em * c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _mzi_single_port(theta, phi):
    """SUPERSEDED placeholder (kept for diffing only)."""
    c = torch.cos(theta).to(torch.complex64)
    s = torch.sin(theta).to(torch.complex64)
    ep = torch.complex(torch.cos(phi), torch.sin(phi))
    one = torch.ones_like(c)
    row0 = torch.stack([ep * c, -one * s], dim=-1)
    row1 = torch.stack([ep * s, one * c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _mzi_differential_signflip(theta, phi):
    """SUPERSEDED placeholder (kept for diffing only)."""
    return _mzi_differential(theta, -phi)


def _mzi_differential_transpose(theta, phi):
    """SUPERSEDED placeholder (kept for diffing only).
    Same as differential but with the off-diagonal signs on the other side
    (in case theta's rotation sense / port labeling is mirrored)."""
    c = torch.cos(theta).to(torch.complex64)
    s = torch.sin(theta).to(torch.complex64)
    ep = torch.complex(torch.cos(phi), torch.sin(phi))
    em = torch.complex(torch.cos(phi), -torch.sin(phi))
    row0 = torch.stack([ep * c, ep * s], dim=-1)
    row1 = torch.stack([-em * s, em * c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


MZI_VARIANTS = {
    "confirmed": _mzi_confirmed,  # <- the real qf.U2mzi convention; should match
    "differential": _mzi_differential,
    "differential_phi_signflip": _mzi_differential_signflip,
    "differential_transpose": _mzi_differential_transpose,
    "single_port": _mzi_single_port,
}

EXPECTED_EXACT = ("confirmed", "forward", "row_reblock")
EXPECTED_POWER = ("confirmed", "forward")


def _broadcast_eta(eta, theta):
    device = theta.device
    batch_shape = theta.shape[:-1]
    if not torch.is_tensor(eta):
        eta = torch.tensor(float(eta), device=device)
    else:
        eta = eta.to(device)
    if eta.dim() == 0:
        eta = eta.expand(*batch_shape, 6, 4)
    return eta


def build_mesh_variant(theta, phi, delta, mzi_fn, block_order="forward", eta=1.0):
    """Same product as mesh_forward.build_mesh_lossy, but with a pluggable
    2x2 core so the sweep can try superseded conventions. eta=1.0 is
    numerically identical to the lossless product (lossy sandwich = I)."""
    D_diag = torch.complex(torch.cos(delta), torch.sin(delta))
    U = torch.diag_embed(D_diag)
    eta = _broadcast_eta(eta, theta)
    order = range(6) if block_order == "reverse_apply" else reversed(range(6))
    for k in order:
        block = lossy_mzi_2x2(
            theta[..., k], phi[..., k], eta[..., k, :], core_fn=mzi_fn,
        )
        G = embed_block(block, BLOCK_PAIRS[k])
        U = G @ U
    return U


# --------------------------------------------------------------------------- #
# Candidate y layouts (must match whatever the real data-generation script wrote)
# --------------------------------------------------------------------------- #

def _layout_col_interleaved(U):
    """Current convention: per-column, [re,im] interleaved per component, columns concatenated."""
    cols = U.transpose(-1, -2)
    re, im = cols.real, cols.imag
    return torch.stack([re, im], dim=-1).reshape(*U.shape[:-2], 32)


def _layout_col_reblock(U):
    """Per-column, but all 4 re's then all 4 im's (still column-major)."""
    cols = U.transpose(-1, -2)
    return torch.cat([cols.real.reshape(*U.shape[:-2], 16),
                       cols.imag.reshape(*U.shape[:-2], 16)], dim=-1)


def _layout_row_interleaved(U):
    """Rows instead of columns, per-component interleaved."""
    rows = U
    re, im = rows.real, rows.imag
    return torch.stack([re, im], dim=-1).reshape(*U.shape[:-2], 32)


def _layout_row_reblock(U):
    """CONFIRMED convention, read directly from datagen.py (not guessed):
    output_field_j = I_j @ U_circuit = ROW j of U_circuit (I_j is a one-hot
    ROW vector, so I_j @ U_circuit picks row j, not column j). Per row: real
    part block (4 numbers) then imag part block (4 numbers), rows
    concatenated j=1..4. This matches train_tandem.U_to_measurement /
    measurement_to_U after the fix -- kept here too so the sweep in
    check_sample/check_dataset below can still catch a *core mzi convention*
    mismatch independent of the layout question, which is now settled."""
    re, im = U.real, U.imag
    per_row = torch.cat([re, im], dim=-1)  # (..., 4, 8)
    return per_row.reshape(*U.shape[:-2], 32)


def _layout_full_matrix_reblock(U):
    """Flatten the whole 4x4 real part row-major, then the whole 4x4 imag part row-major."""
    re = U.real.reshape(*U.shape[:-2], 16)
    im = U.imag.reshape(*U.shape[:-2], 16)
    return torch.cat([re, im], dim=-1)


LAYOUT_VARIANTS = {
    "row_reblock": _layout_row_reblock,   # <- confirmed from datagen.py source
    "col_interleaved": _layout_col_interleaved,
    "col_reblock": _layout_col_reblock,
    "row_interleaved": _layout_row_interleaved,
    "full_matrix_reblock": _layout_full_matrix_reblock,
}


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #

def check_sample(theta, phi, delta, y_real, eta=1.0, tol=1e-4, verbose=True):
    """theta:(6,) phi:(6,) delta:(4,) y_real:(32,) -- all real, single sample
    (no batch dim). eta is the known loss used at generation (1.0 = lossless).
    Returns sorted list of (error, mzi_name, block_order, layout_name)."""
    theta, phi, delta = theta.unsqueeze(0), phi.unsqueeze(0), delta.unsqueeze(0)
    results = []
    for mzi_name, mzi_fn in MZI_VARIANTS.items():
        for block_order in ("forward", "reverse_apply"):
            U_pred = build_mesh_variant(theta, phi, delta, mzi_fn, block_order, eta=eta)[0]
            for layout_name, layout_fn in LAYOUT_VARIANTS.items():
                y_pred = layout_fn(U_pred.unsqueeze(0))[0]
                err = (y_pred - y_real).abs().max().item()
                results.append((err, mzi_name, block_order, layout_name))
    results.sort(key=lambda r: r[0])
    if verbose:
        print(f"{'max|err|':>12}  {'mzi variant':<28}{'block order':<16}{'y layout'}")
        for err, mzi_name, block_order, layout_name in results[:6]:
            flag = "  <-- MATCH" if err < tol else ""
            print(f"{err:12.6f}  {mzi_name:<28}{block_order:<16}{layout_name}{flag}")
    return results


def check_dataset(theta, phi, delta, y, n_check=5, tol=1e-4, eta=1.0):
    """Batched convenience wrapper: runs check_sample on n_check rows and
    reports whether a single combo matches consistently across all of them."""
    n = min(n_check, theta.shape[0])
    winner_counts = {}
    for i in range(n):
        print(f"\n=== sample {i} ===")
        results = check_sample(theta[i], phi[i], delta[i], y[i], eta=eta, tol=tol)
        best_err, *combo = results[0]
        if best_err < tol:
            winner_counts[tuple(combo)] = winner_counts.get(tuple(combo), 0) + 1
    print("\n=== summary ===")
    if not winner_counts:
        print("No candidate convention matched any sample to tol="
              f"{tol}. If this is lossy exact-field data, pass eta=<the generation "
              "loss>. Otherwise extend MZI_VARIANTS / LAYOUT_VARIANTS.")
    else:
        for combo, count in sorted(winner_counts.items(), key=lambda kv: -kv[1]):
            print(f"{combo}: matched {count}/{n} samples")
        best_combo = max(winner_counts, key=winner_counts.get)
        if winner_counts[best_combo] == n:
            print(f"\n-> Consistent match across all {n} samples: {best_combo}")
            if best_combo != EXPECTED_EXACT:
                print(f"   Expected {EXPECTED_EXACT}. Swap mesh_forward.mzi_2x2 / "
                      "U_to_measurement to match before trusting predicted params.")
    return winner_counts


def check_power_sample(theta, phi, delta, power, eta=1.0, probes=None, tol=1e-4, verbose=True):
    """Power-mode analogue of check_sample: no y-layout variants (powers are
    concatenated |probe @ U|^2 over DEFAULT_PROBES). Sweeps mzi core + block
    order. Returns sorted list of (error, mzi_name, block_order)."""
    probes = DEFAULT_PROBES if probes is None else probes
    theta, phi, delta = theta.unsqueeze(0), phi.unsqueeze(0), delta.unsqueeze(0)
    results = []
    for mzi_name, mzi_fn in MZI_VARIANTS.items():
        for block_order in ("forward", "reverse_apply"):
            U_pred = build_mesh_variant(theta, phi, delta, mzi_fn, block_order, eta=eta)
            y_pred = probe_powers(U_pred, probes.to(U_pred.device)).reshape(-1)
            err = (y_pred - power).abs().max().item()
            results.append((err, mzi_name, block_order))
    results.sort(key=lambda r: r[0])
    if verbose:
        print(f"{'max|err|':>12}  {'mzi variant':<28}{'block order'}")
        for err, mzi_name, block_order in results[:6]:
            flag = "  <-- MATCH" if err < tol else ""
            print(f"{err:12.6f}  {mzi_name:<28}{block_order}{flag}")
    return results


def check_power_dataset(theta, phi, delta, power, n_check=5, tol=1e-4, eta=1.0,
                        probes=None):
    n = min(n_check, theta.shape[0])
    winner_counts = {}
    for i in range(n):
        print(f"\n=== sample {i} ===")
        results = check_power_sample(
            theta[i], phi[i], delta[i], power[i], eta=eta, tol=tol, probes=probes,
        )
        best_err, *combo = results[0]
        if best_err < tol:
            winner_counts[tuple(combo)] = winner_counts.get(tuple(combo), 0) + 1
    print("\n=== summary ===")
    if not winner_counts:
        print("No candidate convention matched any power sample to tol="
              f"{tol}. Check --eta, probe set / normalization, and that this "
              "file is actually powers (not Re/Im fields).")
    else:
        for combo, count in sorted(winner_counts.items(), key=lambda kv: -kv[1]):
            print(f"{combo}: matched {count}/{n} samples")
        best_combo = max(winner_counts, key=winner_counts.get)
        if winner_counts[best_combo] == n:
            print(f"\n-> Consistent match across all {n} samples: {best_combo}")
            if best_combo != EXPECTED_POWER:
                print(f"   Expected {EXPECTED_POWER}. Swap mesh_forward.mzi_2x2 "
                      "to match before trusting predicted params.")
    return winner_counts


if __name__ == "__main__":
    import data_pipeline as dp
    from regime import Regime

    raw = dp.synthetic_dataset(5, seed=1)
    print("Self-test on synthetic exact-field data (confirmed convention).")
    print(f"Expect the top match to be {EXPECTED_EXACT} with max|err| ~ 0.\n")
    check_dataset(raw["theta"], raw["phi"], raw["delta"], raw["y"], n_check=5)

    print("\n" + "=" * 72)
    print("Self-test on synthetic power data (eta=0.85).")
    print(f"Expect the top match to be {EXPECTED_POWER} with max|err| ~ 0.\n")
    regime = Regime(observation="power", eta=0.85)
    praw = dp.synthetic_regime_dataset(5, regime, seed=1)
    check_power_dataset(
        praw["theta"], praw["phi"], praw["delta"], praw["obs"],
        n_check=5, eta=0.85, probes=regime.probes,
    )

    print("\n" + "=" * 72)
    print("Self-test on synthetic power data (eta=0.85, quadrature=True).")
    print(f"Expect the top match to be {EXPECTED_POWER} with max|err| ~ 0.\n")
    regime_q = Regime(observation="power", eta=0.85, quadrature=True)
    praw_q = dp.synthetic_regime_dataset(5, regime_q, seed=2)
    check_power_dataset(
        praw_q["theta"], praw_q["phi"], praw_q["delta"], praw_q["obs"],
        n_check=5, eta=0.85, probes=regime_q.probes,
    )