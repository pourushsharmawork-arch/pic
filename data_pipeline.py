"""
Turns raw data (either your real 48-row-per-sample dataset, or synthetically
sampled params) into the {"y", "U", "sincos"} dict format that
train_tandem.train_model() expects.

Two data sources, two different trust levels:

1. `load_real_dataset(...)`   -- YOUR data. First 16 = (theta[6], phi[6],
   delta[4]), remaining 32 = the measurement y. This is ground truth for
   accuracy, but the *meaning* of those 48 numbers (row order, block order,
   which mzi convention produced them) is only as reliable as your
   generation script -- verify_convention.py checks the part of this that's
   checkable from data alone (see the big warning at the bottom of this
   docstring).

2. `synthetic_dataset(...)`   -- self-consistent by construction: samples
   params, runs them through mesh_forward.build_mesh to get U, and derives y
   from U_to_measurement. Useful for (a) smoke-testing the whole pipeline
   before you trust it on real data, (b) pretraining / augmentation, (c) as
   the reference generator inside verify_convention.py. It is NOT a
   substitute for real data if your actual hardware/simulator uses a
   different mzi convention than mesh_forward.mzi_2x2 -- see
   verify_convention.py.

*** UPDATE: the y-LAYOUT question is now resolved (from reading datagen.py's
actual source, not guessed): y is ROWS of U_circuit, block [Re,Im] per row,
NOT columns with interleaved re/im. U_to_measurement/measurement_to_U in
train_tandem.py were corrected accordingly -- if anything was trained
against the old versions, retrain from scratch, don't fine-tune. ***

*** MZI CONVENTION: settled. *** mesh_forward.mzi_2x2 matches qf.U2mzi
called as U2mzi(theta, alpha=-phi, beta=pi/2, chi=0). y layout is rows of
U_circuit, [Re(4), Im(4)] per row (exact field) or concatenated |probe @ U|^2
over I1..I4 + I12,I23,I34,I14 (power). Run verify_convention.py against a
handful of real rows (pass eta= generation loss) before a long training run.

*** FOUR REGIMES: *** observation ∈ {exact, power} × eta = 1.0 (lossless) or
< 1 (lossy). `regime.py` is the unified training path for all four.
`train_tandem.py` / `run_pipeline.py` cover exact-field only (use
mode="physics_lossy" + eta for lossy exact). Power data must go through
`load_real_power_dataset` + `run_regime.py --observation power` -- the 32
trailing columns are the same width as exact-field y but are NOT Re/Im.
"""
from __future__ import annotations

import numpy as np
import torch

from mesh_forward import build_mesh, build_mesh_lossy, N_MODES, power_obs_dim, power_row_width
from train_tandem import U_to_measurement, measurement_to_U, angles_to_sincos

N_PARAMS = 16   # 6 theta + 6 phi + 4 delta
N_MEAS = 32     # 4 columns x (re,im) x 4 components


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #

def _load_array(path: str, npz_key: str | None = None) -> np.ndarray:
    if path.endswith(".npz"):
        d = np.load(path)
        if npz_key is not None:
            key = npz_key
        elif len(d.keys()) == 1:
            key = list(d.keys())[0]
        else:
            key = "data" if "data" in d.files else list(d.keys())[0]
        return np.asarray(d[key])
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".csv") or path.endswith(".txt"):
        return np.loadtxt(path, delimiter="," if path.endswith(".csv") else None)
    if path.endswith(".pt"):
        return torch.load(path, map_location="cpu").numpy()
    raise ValueError(f"Don't know how to load {path!r} -- add a branch for its format.")


def load_real_dataset(path: str, orientation: str = "auto") -> dict:
    """
    path: file with your 48-row(-or-column)-per-sample data.
    orientation: "rows" if array is (48, N_samples), "cols" if (N_samples, 48),
                 "auto" tries to infer from shape (assumes N_samples > 48).

    Returns dict with keys "theta" (N,6), "phi" (N,6), "delta" (N,4),
    "y" (N,32), "U" (N,4,4) complex -- ready to hand to build_dataset_dict.
    """
    arr = _load_array(path)
    arr = np.asarray(arr, dtype=np.float64)

    if orientation == "auto":
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D array, got shape {arr.shape}")
        if arr.shape[0] == N_PARAMS + N_MEAS:
            orientation = "rows"
        elif arr.shape[1] == N_PARAMS + N_MEAS:
            orientation = "cols"
        else:
            raise ValueError(
                f"Neither dim of shape {arr.shape} is {N_PARAMS + N_MEAS} "
                f"(={N_PARAMS}+{N_MEAS}). Pass orientation explicitly."
            )

    if orientation == "rows":
        arr = arr.T  # -> (N_samples, 48)

    if arr.shape[1] != N_PARAMS + N_MEAS:
        raise ValueError(f"After orientation fix, expected (N,48), got {arr.shape}")

    params = torch.as_tensor(arr[:, :N_PARAMS], dtype=torch.float32)
    y = torch.as_tensor(arr[:, N_PARAMS:], dtype=torch.float32)

    # ASSUMED param sub-order within the 16 -- flag loudly if wrong.
    theta, phi, delta = params[:, :6], params[:, 6:12], params[:, 12:16]
    U = measurement_to_U(y)

    return {"theta": theta, "phi": phi, "delta": delta, "y": y, "U": U}


# --------------------------------------------------------------------------- #
# Synthetic (self-consistent) data
# --------------------------------------------------------------------------- #

def synthetic_dataset(n_samples: int, seed: int = 0, theta_range=(0.0, 2 * torch.pi)) -> dict:
    """Samples params, builds U via build_mesh, derives y. Self-consistent
    with mesh_forward's *assumed* convention by construction -- see module
    docstring before trusting this as a stand-in for real data."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = theta_range
    theta = torch.rand(n_samples, 6, generator=g) * (hi - lo) + lo
    phi = torch.rand(n_samples, 6, generator=g) * 2 * torch.pi
    delta = torch.rand(n_samples, 4, generator=g) * 2 * torch.pi
    U = build_mesh(theta, phi, delta)
    y = U_to_measurement(U)
    return {"theta": theta, "phi": phi, "delta": delta, "y": y, "U": U}


def synthetic_lossy_dataset(n_samples: int, eta: float, seed: int = 0,
                             theta_range=(0.0, 2 * torch.pi)) -> dict:
    """Same as synthetic_dataset but through build_mesh_lossy -- U is now
    genuinely non-unitary. Use this to smoke-test mode='physics_lossy' before
    running on real lossy data. `eta` matches datagen.py's `loss` kwarg
    (single global scalar efficiency applied to every port of every MZI)."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = theta_range
    theta = torch.rand(n_samples, 6, generator=g) * (hi - lo) + lo
    phi = torch.rand(n_samples, 6, generator=g) * 2 * torch.pi
    delta = torch.rand(n_samples, 4, generator=g) * 2 * torch.pi
    U = build_mesh_lossy(theta, phi, delta, eta)
    y = U_to_measurement(U)
    return {"theta": theta, "phi": phi, "delta": delta, "y": y, "U": U}


# --------------------------------------------------------------------------- #
# Common: dict -> {"y","U","sincos"} + split
# --------------------------------------------------------------------------- #

def build_dataset_dict(raw: dict) -> dict:
    sincos = angles_to_sincos(raw["theta"], raw["phi"], raw["delta"])
    return {"y": raw["y"], "U": raw["U"], "sincos": sincos,
            "theta": raw["theta"], "phi": raw["phi"], "delta": raw["delta"]}


def train_val_split(data: dict, val_frac: float = 0.1, seed: int = 0) -> tuple[dict, dict]:
    n = data["y"].shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train = {k: v[train_idx] for k, v in data.items()}
    val = {k: v[val_idx] for k, v in data.items()}
    return train, val


# --------------------------------------------------------------------------- #
# Regime-aware dataset builder (see regime.py)
# --------------------------------------------------------------------------- #

def synthetic_regime_dataset(n_samples: int, regime, seed: int = 0,
                              theta_range=(0.0, 2 * torch.pi)) -> dict:
    """Samples params, simulates the observation for the given Regime
    (exact field, or probe powers). Returns {"theta","phi","delta","obs"} --
    the format train_model_regime expects directly, no further processing."""
    from regime import simulate
    g = torch.Generator().manual_seed(seed)
    lo, hi = theta_range
    theta = torch.rand(n_samples, 6, generator=g) * (hi - lo) + lo
    phi = torch.rand(n_samples, 6, generator=g) * 2 * torch.pi
    delta = torch.rand(n_samples, 4, generator=g) * 2 * torch.pi
    obs = simulate(theta, phi, delta, regime)
    return {"theta": theta, "phi": phi, "delta": delta, "obs": obs}


def regime_train_val_split(data: dict, val_frac: float = 0.1, seed: int = 0):
    n = data["obs"].shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train = {k: v[train_idx] for k, v in data.items()}
    val = {k: v[val_idx] for k, v in data.items()}
    return train, val


def _read_npz_probe_meta(path: str) -> dict:
    """Optional metadata written by datagen.py (may be absent on old files)."""
    if not path.endswith(".npz"):
        return {}
    d = np.load(path)
    meta = {}
    for key in ("basis_only", "quadrature", "observation", "loss"):
        if key in d.files:
            v = d[key]
            meta[key] = bool(v.item()) if v.shape == () else v
    return meta


def load_real_power_dataset(path: str, orientation: str = "auto",
                             quadrature: bool = False,
                             basis_only: bool = False) -> dict:
    """
    Power-mode rows: [theta(6), phi(6), delta(4), powers].

    Row widths (auto-detected when orientation='auto'):
      32 → basis_only (4 probes × 4 ports = 16 powers)
      48 → 8-probe superposition (same width as exact-field — pass --observation!)
      64 → quadrature (12 probes × 4 ports = 48 powers)

    NPZ metadata keys (from datagen.py) override flags when present.

    Returns {"theta","phi","delta","obs","quadrature","basis_only"}.
    """
    meta = _read_npz_probe_meta(path)
    if "basis_only" in meta:
        basis_only = bool(meta["basis_only"])
    if "quadrature" in meta:
        quadrature = bool(meta["quadrature"])

    arr = _load_array(path)
    arr = np.asarray(arr, dtype=np.float64)
    obs_dim = power_obs_dim(quadrature, basis_only)
    row_width = N_PARAMS + obs_dim

    if orientation == "auto":
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D array, got shape {arr.shape}")
        width_map = {
            N_PARAMS + power_obs_dim(False, True): (True, False),
            N_PARAMS + power_obs_dim(False, False): (False, False),
            N_PARAMS + power_obs_dim(True, False): (False, True),
        }
        for dim in (0, 1):
            w = arr.shape[dim]
            if w in width_map:
                if dim == 0:
                    orientation = "rows"
                else:
                    orientation = "cols"
                basis_only, quadrature = width_map[w]
                obs_dim = power_obs_dim(quadrature, basis_only)
                row_width = N_PARAMS + obs_dim
                break
        else:
            raise ValueError(
                f"Shape {arr.shape} is not a recognized power row width "
                f"(32=basis-only, 48=8-probe, 64=quadrature). Pass flags explicitly."
            )
    if orientation == "rows":
        arr = arr.T

    row_width = N_PARAMS + obs_dim
    if arr.shape[1] != row_width:
        raise ValueError(
            f"Expected (N,{row_width}) for basis_only={basis_only} "
            f"quadrature={quadrature}, got {arr.shape}"
        )

    theta = torch.as_tensor(arr[:, 0:6], dtype=torch.float32)
    phi = torch.as_tensor(arr[:, 6:12], dtype=torch.float32)
    delta = torch.as_tensor(arr[:, 12:16], dtype=torch.float32)
    power = torch.as_tensor(arr[:, 16:16 + obs_dim], dtype=torch.float32)

    if (power < -1e-4).any():
        raise ValueError(
            "Some 'power' values are negative — not a power dataset."
        )

    return {
        "theta": theta, "phi": phi, "delta": delta, "obs": power,
        "quadrature": quadrature, "basis_only": basis_only,
    }


def real_data_to_regime_dict(raw: dict) -> dict:
    """Adapter: load_real_dataset() returns {"theta","phi","delta","y","U"};
    regime.py's train_model_regime wants {"theta","phi","delta","obs"}. For
    data generated by datagen.py --observation exact (I1..I4 basis, Re/Im),
    that's observation="exact" and obs is just y, already in the row-block
    layout -- no transformation needed, just a rename. For power files use
    load_real_power_dataset instead; do not pass those through this adapter."""
    return {"theta": raw["theta"], "phi": raw["phi"], "delta": raw["delta"], "obs": raw["y"]}


if __name__ == "__main__":
    raw = synthetic_dataset(2000, seed=0)
    data = build_dataset_dict(raw)
    train_d, val_d = train_val_split(data, val_frac=0.1)
    print("train y:", train_d["y"].shape, " val y:", val_d["y"].shape)
    print("sincos:", train_d["sincos"].shape, " U:", train_d["U"].dtype, train_d["U"].shape)
    # sanity: U reconstructed from y round-trips exactly (layout self-consistent)
    err = (measurement_to_U(train_d["y"]) - train_d["U"]).abs().max().item()
    print("measurement_to_U round-trip max abs err (should be ~0):", err)