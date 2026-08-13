# PARAMS — Inverse MZI-mesh calibration

Recover the 16 (or 12) physical angles of a 4-mode rectangular MZI mesh
from a measurement of how the mesh acts on light.

This directory is a **tandem inverse model**: a neural net proposes angles,
a differentiable copy of the photonic circuit checks them, and a short
gradient refinement polishes the result. It does **not** call the analytic
`qfsim` decomposer at train or inference time.

The same code path covers four measurement regimes:

| | **exact field** (Re/Im of output amplitudes) | **power only** (`\|amplitude\|²` on photodetectors) |
|---|---|---|
| **lossless** (`η = 1`) | 16 angles, locally surjective onto U(4) | 12 angles; `delta` is invisible |
| **lossy** (`η < 1`) | 16 angles; `U` is non-unitary | 12 angles; hardest identifiability |

---

## 1. The device

A 4-mode rectangular ("1212") mesh implements

```
M = Ga23 @ Ga12 @ Ga34 @ Gb23 @ Gb12 @ Gb34 @ D
```

Each `G_xy` is a 2×2 Mach–Zehnder interferometer on modes `x,y` (1-indexed),
with two angles `(θ, φ)`. `D = diag(e^{iδ₀}, …, e^{iδ₃})` is an output-phase
screen. That is **16 real numbers**: `theta[6]`, `phi[6]`, `delta[4]`.

Light is a **row vector**: `E_out = E_in @ M`. Datagen and `mesh_forward`
share this convention. (The parent `qf/` repo's SU(4) recovery network uses
the column-action `y = U @ x`; that is a different pipeline.)

The 2×2 MZI core, confirmed from `qf.U2mzi(θ, α=−φ, β=π/2, χ=0)`:

```
T(θ, φ) = [[ e^{-iφ} cos θ,  −sin θ      ],
           [  sin θ,          e^{iφ}  cos θ ]]
```

This is `mesh_forward.mzi_2x2`. It matches `qfsim.misc.T_local`. An older
literature placeholder (phase on both off-diagonals) is kept in the source
for diffing only — do not train against it.

### Loss

When `η < 1`, each MZI is sandwiched between real diagonal amplitude
matrices, matching `datagen.lossy_U2mzi` / `qfsim.core.lossy_U2mzi`:

```
L_in @ T(θ, φ) @ L_out,    L = diag(η_port)
```

`η` is applied **as written** (not `√η`). The name "intensity efficiency" in
`qfsim` is therefore a bit misleading: `--loss 0.85` / `--eta 0.85` means
amplitude transmission 0.85 on every port of every MZI. `η = 1` is
numerically identical to the strictly unitary `build_mesh`.

`D` is always lossless. Powers therefore cannot see `delta`: a unit-modulus
phase on output port `j` cancels in `|E_out,j|²`. That is physics, not a
pipeline limitation.

---

## 2. The inverse problem

**Forward (easy):** `(θ, φ, δ, η) → M → measurement y`.

**Inverse (ill-posed):** `y → (θ, φ[, δ])`.

Ill-posed for two independent reasons:

1. **Gauge.** Several angle sets build the same `M`. Analytic decomposition
   (`decomp.py` / `qfsim`) picks one; a network supervised on those labels
   is punished for finding a different, equally valid set.
2. **Incomplete observations.** Powers throw away phase. Real superposition
   probes recover `cos(Δφ)` and leave a **sign ambiguity** on relative
   phases. **Quadrature probes** `(e_i + i e_j)/√2` add the missing
   `Im(U[a,j]*conj(U[b,j]))` term and resolve that sign — enable with
   `--quadrature` when your hardware can prepare them. `delta` is still
   invisible to any power measurement.

The fix is a **tandem / physics loss**: decode the net's angles, run the
differentiable forward model, and compare in *observation space*. Any
parameter set that reproduces `y` is a success. Labels are used only as a
weak auxiliary pull toward a canonical branch.

At inference, the net is a warm start. A short Adam loop (optionally
multi-start) refines the angles on the same physics loss.

---

## 3. Four regimes — can they all be implemented?

**Yes. They all run today**, through one object:

```python
from regime import Regime
regime = Regime(observation="exact" | "power", eta=1.0 | 0.85, quadrature=False | True)
```

| Config | `--observation` | `--eta` / `--loss` | Net I/O | Predicts `delta`? |
|---|---|---|---|---|
| lossless + exact field | `exact` | `1.0` | 32 → 32 (16 angles as sin/cos) | yes |
| lossy + exact field | `exact` | e.g. `0.85` | 32 → 32 | yes |
| lossless + powers (basis only) | `power` + `--basis_only` | `1.0` | **16 → 24** | **no** |
| lossy + powers (basis only) | `power` + `--basis_only` | e.g. `0.85` | **16 → 24** | **no** |
| lossless + powers + superposition | `power` | `1.0` | 32 → 24 | **no** |
| lossy + powers | `power` | e.g. `0.85` | 32 → 24 | **no** |
| lossless + powers + **quadrature** | `power` + `--quadrature` | `1.0` | **48 → 24** | **no** |
| lossy + powers + **quadrature** | `power` + `--quadrature` | e.g. `0.85` | **48 → 24** | **no** |

What is shared: `InverseNet` residual MLP, sin/cos encoding, tandem loss,
unit-circle regularizer, multi-start refine, `build_mesh_lossy`.

What changes per regime, and only this:

1. **Readout** — `U_to_measurement` (Re/Im of four basis outputs) vs
   `probe_powers` (8 or 12 probes × 4 ports when `--quadrature`).
2. **Whether `delta` is a free variable** — forced off for power.
3. **The known `η`** — a lab setting, not inferred.

### Identifiability (what "success" means)

| Config | What you can hope to recover | What "high fidelity" means |
|---|---|---|
| lossless exact | some 16-angle preimage of `M ∈ U(4)` (Jacobian rank 16/16 locally) | reconstructed `M` matches the measured unitary |
| lossy exact | some 16-angle preimage of the *non-unitary* `M(η)` | same, without assuming `M†M = I` |
| lossless power | some `(θ, φ)` matching 32 powers; `δ` arbitrary | reconstructed powers match |
| lossless power + quadrature | tighter phase constraints; less multi-modal refine | same; multi-start less critical |
| lossy power | same, weaker | same; multi-start still helps |

Do **not** score these models by RMSE against stored `(θ, φ, δ)` labels.
That metric mixes gauge, sign ambiguities, and (for power) a true null
space along `δ`. Score **observation fidelity**: do the predicted angles
reproduce the measurement?

### What was not true until this revision

- `datagen.py` only emitted powers, and ran 10k samples on import.
- `verify_convention.py` never tested the confirmed MZI, so the pre-flight
  check could not succeed.
- Power data had no convention check, and lossy exact data was checked
  against a lossless mesh (it would always fail).
- `run_regime.py` required a file and did not smoke-test synthetic data.

Those are fixed. All four configs generate, convention-check, train, and
refine. Details in [§10](#10-code-review-notes).

---

## 4. Data layout

Row width = **16 params + obs_dim**:

| Mode | Row width | `obs` dim | Probes |
|---|---:|---:|---|
| exact field | 48 | 32 | I1…I4 (Re/Im each) |
| power + **basis_only** | **32** | **16** | I1…I4 powers only |
| power + superposition | 48 | 32 | I1…I4 + I12,I23,I34,I14 |
| power + superposition + quadrature | **64** | **48** | above + I12q,… |

NPZ files also store `observation`, `loss`, `basis_only`, and `quadrature`
metadata keys (from `datagen.py`). Auto-detect: **32** cols → power basis-only;
**64** cols → quadrature; **48** cols → exact *or* 8-probe power (pass
`--observation` explicitly).

| Columns | Contents |
|--------:|----------|
| `0:6` | `theta` |
| `6:12` | `phi` |
| `12:16` | `delta` (stored even in power mode, for diagnostics only) |
| `16:` | observation vector (32 or 48 numbers — see table) |

**Exact field** (`--observation exact`) — four computational-basis probes
`I1…I4`. Probe `j` is a one-hot *row*, so `I_j @ M` is **row** `j` of `M`.
Packed as `[Re(row), Im(row)]` per probe, probes concatenated. This is
`U_to_measurement` / `measurement_to_U` (a reshape; valid for lossy `M`).

**Power** (`--observation power`) — eight unit-norm probes by default:

```
I1, I2, I3, I4, I12, I23, I34, I14     where I12 = (e1+e2)/√2, etc.
```

Packed as `|output|²` per port, 8×4 = 32 nonnegative numbers.

**Power + basis only** (`--basis_only`) — four computational-basis probes
only, same inputs as exact-field mode but **powers** not Re/Im. 4×4 = **16**
nonnegative numbers. Row width **32** (unique — not confusable with 48-col
modes). Many relative-φ directions have **zero first-order effect** on
these measurements; see [§13](#13-identifiability--gauge).

**Power + quadrature** (`--quadrature`) — adds four more probes on the same
adjacent pairs, with a 90° relative phase:

```
I12q = (e1 + i e2)/√2,  I23q = (e2 + i e3)/√2,  I34q = (e3 + i e4)/√2,  I14q = (e1 + i e4)/√2
```

12 probes × 4 ports = **48** power numbers. Row width **64**. Use the same
`--quadrature` flag at train and inference time; checkpoint weights are **not**
interchangeable between 32-D and 48-D power models.

**Disambiguation:** 48-column files are either exact field *or* power without
quadrature — the loader cannot tell. Always pass `--observation` to match
generation. 64-column files imply power + quadrature.

Existing `data/data.npz` is **power + lossy, no quadrature** (48 columns,
all `y ≥ 0`). Do not feed it to the exact-field loader.

---

## 5. Files

| File | Role |
|---|---|
| `datagen.py` | Generate any of the four configs from Haar SU(4) + `qfsim` decompose + lossy mesh |
| `mesh_forward.py` | Differentiable `build_mesh` / `build_mesh_lossy`, probes, powers |
| `regime.py` | Unified `Regime`, tandem loss, train, refine — **use this** |
| `data_pipeline.py` | Load exact or power files; synthetic generators; splits |
| `verify_convention.py` | Pre-flight: do stored params + forward model reproduce `y`? |
| `run_regime.py` | CLI for all regimes (+ `--basis_only`, `--quadrature`) |
| `identifiability.py` | Jacobian rank / null space / global gauge demos — **answers Q1 & Q2** |
| `train_tandem.py` | `InverseNet`, sin/cos helpers, and the **exact-field-only** original loop |
| `run_pipeline.py` | CLI for exact-field only (`physics` / `physics_lossy` / `param_only`) |
| `infer_interpret.py` | Per-sample inference + Jacobian sensitivity (all regimes) |
| `decomp.py` | Analytic decomposer wrapper; **never called** during train/infer |
| `qfsim/` | Local copy of the analytic mesh / `U2mzi` |

You do not need to import `mesh_forward` / `regime` yourself unless you are
writing a new driver — `run_regime.py` wires them.

---

## 6. Setup

```bash
cd PARAMS
pip install torch numpy scipy
mkdir -p data
```

---

## Quick end-to-end demo (~2 minutes)

Copy-paste this once after setup. Uses 500 synthetic samples and 15 epochs —
enough to see convention check → train → refine → infer work; bump
`--n_samples` / `--epochs` for real results.

```bash
# 0. Sanity (expect three 5/5 confirmed lines)
python verify_convention.py

# 1. Generate lossless exact-field data
python datagen.py --observation exact --loss 1.0 --n_samples 500 --out data/demo_exact.npz

# 2. Train + eval (convention check runs automatically)
python run_regime.py --data data/demo_exact.npz --observation exact --eta 1.0 \
    --epochs 15 --n_refine_eval 50 --out demo_model.pt

# 3. Infer on one held-out row
python infer_interpret.py --checkpoint demo_model.pt --data data/demo_exact.npz \
    --observation exact --eta 1.0 --index 0
```

**What good output looks like:** convention summary
`('confirmed', 'forward', 'row_reblock'): matched 5/5`; after training,
`+ 8-start refine` fidelity ≳ 0.95 (network-only will be much lower at
15 epochs — that is normal). `infer_interpret` refined obs fidelity ≳ 0.95.

**Power demo** (matches existing `data/data.npz` if present):

```bash
python run_regime.py --data data/data.npz --observation power --eta 0.85 \
    --epochs 15 --out demo_power.pt
python infer_interpret.py --checkpoint demo_power.pt --data data/data.npz \
    --observation power --eta 0.85 --index 0
```

**Quadrature** — add `--quadrature` to datagen, `run_regime.py`, and
`infer_interpret.py` consistently (see §7–8).

---

## 7. Generate data

```bash
python datagen.py --observation exact --loss 1.0  --out data/exact_lossless.npz
python datagen.py --observation exact --loss 0.85 --out data/exact_lossy.npz
python datagen.py --observation power --loss 1.0  --out data/power_lossless.npz
python datagen.py --observation power --basis_only --loss 1.0 --out data/power_basis_lossless.npz
python datagen.py --observation power --loss 0.85 --out data/power_lossy.npz
python datagen.py --observation power --loss 1.0 --quadrature --out data/power_quad_lossless.npz
python datagen.py --observation power --loss 0.85 --quadrature --out data/power_quad_lossy.npz
```

`--loss` must later be passed unchanged as `--eta`. Default `--n_samples` is
10 000. Importing `datagen` no longer writes a file; only `python datagen.py`
does.

Sanity checks:

```bash
python verify_convention.py          # synthetic exact + power self-tests; expect 5/5 confirmed
python data_pipeline.py              # round-trip reshape ~ 0
python mesh_forward.py               # Jacobian rank 16/16, unitarity ~ 0
```

On a real file, `run_regime.py` runs the convention check for you (exact
*and* power, lossless *and* lossy). Expect

- exact: `('confirmed', 'forward', 'row_reblock')`
- power: `('confirmed', 'forward')`

---

## 8. Train

**Unified path (all four regimes):**

```bash
# synthetic smoke test (no file needed)
python run_regime.py --observation exact --eta 1.0 --epochs 30
python run_regime.py --observation power --eta 0.85 --epochs 30

# real files — observation flag MUST match the file
python run_regime.py --data data/exact_lossless.npz --observation exact --eta 1.0 --epochs 60
python run_regime.py --data data/exact_lossy.npz    --observation exact --eta 0.85 --epochs 60
python run_regime.py --data data/power_lossless.npz --observation power --eta 1.0 --epochs 60
python run_regime.py --data data/power_basis_lossless.npz --observation power --basis_only --eta 1.0 --epochs 60
python run_regime.py --data data/power_lossy.npz    --observation power --eta 0.85 --epochs 60
python run_regime.py --data data/power_quad_lossy.npz --observation power --eta 0.85 --quadrature --epochs 60
```

Useful flags: `--n_starts 8`, `--n_refine_eval 200`, `--basis_only` (power:
I1..I4 only), `--quadrature` (power: +quadrature probes), `--mode param_only`
(ablation), `--skip_convention_check`, `--out my_model.pt`.

Bump `--epochs` to 100+ for a serious run. 15–30 is only for wiring tests.

**Exact-field-only legacy path** (`run_pipeline.py`): still valid for
`--mode physics` (lossless) and `--mode physics_lossy --eta …`. It cannot
train on powers.

### Reading the numbers

```
[train] best val_obs_mse: 0.0696

[eval] observation fidelity (found params reproduce measured data):
  network only          : 0.517  (min 0.098)
  + single-start refine  : 0.987  (min 0.778)
  + 8-start refine        : 0.992  (min 0.808)
```

- **`val_obs_mse`** — training metric, not very interpretable alone.
- **`network only`** — amortized guess. Mediocre is normal.
- **`+ N-start refine`** — the number that matters. ≳ 0.98 means the
  refined angles reproduce the measurement. If `min` lags `mean`, some
  samples landed in the wrong basin: raise `--n_starts`.
- This is **observation** fidelity, not parameter-space accuracy.

### Using a trained model

```python
import torch
from regime import Regime, make_model, decode, refine_multistart_regime

regime = Regime(observation="power", eta=0.85, quadrature=True)  # must match training
model = make_model(regime)
model.load_state_dict(torch.load("inverse_net.pt", map_location="cpu"))
model.eval()

y = torch.tensor(your_48_power_numbers, dtype=torch.float32).unsqueeze(0)
theta, phi, delta, err = refine_multistart_regime(model, y, regime, n_starts=8)
```

For power models, `delta` is a dummy zero tensor. Ignore it.

**CLI inference** (all regimes):

```bash
# exact field
python infer_interpret.py --checkpoint inverse_net.pt --data data/exact_lossless.npz

# power + quadrature
python infer_interpret.py --observation power --eta 0.85 --quadrature \\
    --checkpoint inverse_net.pt --data data/power_quad_lossy.npz --index 0
```

Pass `--observation`, `--eta`, and `--quadrature` to match how the checkpoint
was trained. `infer_interpret.py` reports observation fidelity, optional matrix
fidelity (exact mode), predicted angles, label RMSE (diagnostic), and a
Jacobian sensitivity breakdown by probe.

---

## 9. How the training actually works

`InverseNet`: `obs_dim → Linear(256) → [LayerNorm → Linear(512) → GELU →
Linear(256)] × 4 → LayerNorm → Linear(sincos_dim)`.

Angles are stored as `(sin, cos)` pairs and decoded with `atan2`, so the
net never sees the `0/2π` cut.

```
L = physics_MSE(simulate(decode(pred)), y)
  + 0.15 * MSE(pred_sincos, label_sincos)
  + 0.01 * (sin²+cos² − 1)²
```

AdamW + OneCycleLR, batch 256, grad clip 5.

Refinement: 150 Adam steps at lr 0.03 on the raw angles, physics loss only.
Multi-start: net prediction + `n_starts−1` Gaussian perturbations in angle
space; keep the best observation error per sample. Power landscapes are
more multi-modal without quadrature probes (cos-only interference sign
ambiguity); `--quadrature` tightens identifiability and makes multi-start
less critical, but it remains cheap insurance.

---

## 10. Code review notes

Fixes applied in the same pass as this README:

1. **`verify_convention.py`** — `_mzi_confirmed` is now in the sweep
   (it was defined and then never registered, so the checker could not
   report the real convention). Expected combo is
   `('confirmed', 'forward', 'row_reblock')`. Lossy exact data is checked
   through `build_mesh_lossy`. Power data has `check_power_dataset`.
2. **`datagen.py`** — `--observation {exact,power}` and `--loss`; no
   import-time 10k write. Superposition probes stay unit-norm
   (`/ np.linalg.norm`).
3. **`run_regime.py`** — synthetic smoke test if `--data` is omitted;
   convention check for both observation types, with `--eta`.
4. **`infer_interpret.py` / `data_pipeline._load_array`** — `npz_key` is
   actually accepted now (it used to `TypeError`).
5. Stale headers in `mesh_forward.py` / `data_pipeline.py` still claimed
   the MZI convention was unknown, and that datagen divided by `√‖v‖`.

Left as-is, on purpose:

- **`η` vs `√η`.** Matches `qfsim.core.loss_matrix`. Changing it would
  desynchronize the torch forward model from the generator. Documented
  in §1.
- **Two CLIs.** `run_pipeline.py` is the older exact-field driver;
  `run_regime.py` covers all regimes including `--quadrature`.
- **`infer_interpret.py`** now takes `--observation`, `--eta`, `--quadrature`.
- **No automatic exact vs power sniffing** for 48-column files. 64 columns
  imply power + quadrature only.
- **`η` is global and known.** Per-MZI or per-sample loss would need
  `eta` as a `(6, 4)` tensor (already accepted by `build_mesh_lossy`)
  and a `Regime` extension; it is not in `datagen` today.

Convention check failed on real hardware data?

- A *different* combo matches 5/5 — the checker told you which piece
  (core, block order, layout) is off. Change `mzi_2x2` or the layout,
  do not fine-tune a net trained on the wrong one.
- Nothing matches — add a variant, or transcribe the hardware 2×2
  into `mzi_2x2`. Training will still converge (physics loss only needs
  *some* preimage of `y`); predicted angles will not be dial-in settings.

Do not mix `--eta` with a file generated at a different `loss`. There is
no automatic check. Fidelity can still look excellent against the *wrong*
forward model.

---

## 11. Invariants ("no label leakage")

1. `y` is always compared to `simulate(predicted angles)`, never to an
   analytic decompose of `y`.
2. Stored `(θ, φ, δ)` enter only the auxiliary MSE and optional RMSE
   reports.
3. `decompose_U4_rectangular` is not called in `regime.py` /
   `train_tandem.py` / refinement.
4. For exact field, `U_target` reconstructed from `y` is a reshape, not
   a polar decomposition.

---

## 12. Relation to the parent `qf/` repo

The parent recovers `U ∈ SU(4)` from complex fields with a DeepSets
encoder + `su(4)` head, then optionally decomposes `U.T` into mesh
params. This directory learns the direct map `y → angles` with a tandem
physics loss, including lossy and power-only measurements that the parent
pipeline does not handle.

---

## 13. Identifiability & gauge

High **observation fidelity** means the predicted angles reproduce the
measurement. It does **not** mean the angles equal the stored labels or are
unique. Two separate questions:

### Q1 — Basis probes only: what is undetermined?

With `--basis_only`, you measure only `|I_j @ M|²` for `j = 1..4`. Diagonal
entries of `M` (up to loss) are constrained; **relative phases between
off-diagonal entries are largely invisible** to first order.

**How to see it:**

```bash
# Jacobian rank table (all probe configs side by side)
python identifiability.py --compare

# Detail for basis-only: null-space vectors = unobservable directions
python identifiability.py --observation power --basis_only --probe-null 0
```

Expect **rank ≈ 9/12**, **null dim ≈ 3** — three independent directions in
`(θ, φ)` that change the measurement by zero to first order (mostly φ
combinations). The numerical probe along null #0 should show `max|Δobs| ≪ 1`.

Train on basis-only data anyway (`--basis_only`); refine can still reach high
obs fidelity, but **label RMSE** (`infer_interpret.py`) stays large — the net
picks an arbitrary branch among equivalent solutions.

### Q2 — Superposition probes: what is gauge-only?

With default 8-probe power (or exact field), **local** identifiability is
better: the Jacobian at a generic point often has **full rank** on the 12
active `(θ, φ)` (or 16 with exact). That means every *infinitesimal*
parameter direction affects the measurement.

**Global** non-uniqueness remains: many angle sets produce the **same**
observation (mesh decomposition branches, φ combinations that enter only as
sums like `φ_a + φ_b + φ_c = const` at the level of the full circuit map).

**How to see it:**

```bash
# Local: often rank 12/12 for 8-probe power
python identifiability.py --observation power

# Global: two independent cold-start solves, same obs → different (θ, φ)
python identifiability.py --observation power --global-demo

# After training: high obs fidelity, high label RMSE (diagnostic)
python infer_interpret.py --observation power --checkpoint inverse_net.pt \
    --data data/power_lossy.npz --index 0
```

Interpretation guide:

| Metric | Basis-only (Q1) | Superposition (Q2) |
|---|---|---|
| Jacobian null dim | **> 0** — parameters undetermined locally | **0** locally — all small moves change obs |
| Obs fidelity after refine | Can be ≈ 1 | Can be ≈ 1 |
| Label RMSE vs stored `(θ,φ,δ)` | Poor (expected) | Poor (expected) — gauge / branch |
| `--global-demo` param spread | N/A (null space already shows it) | Large `Δφ` at equal obs quality |

**Do not** use stored decomposition labels as ground truth for power models.
Use `identifiability.py` before a study to know which directions are null, and
obs fidelity (not label RMSE) as the success metric.

---

## References

- Ardizzone et al. 2019 — invertible nets and ill-posed inverses
- Rausell-Campo et al. 2026, arXiv:2607.09301 — tandem nets for MZI meshes
- Clements et al. 2016 / Reck et al. 1994 — rectangular / triangular meshes
- Mezzadri 2007 — Haar unitaries via QR of a Ginibre ensemble
