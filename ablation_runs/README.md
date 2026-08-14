# PARAMS Ablation Report

- Generated: `2026-08-13 13:49:32`
- Samples per dataset: `5000`
- Epochs: `60`
- Refinement starts: `200`
- Refinement evaluation samples: `200`

## Experiment matrix

| Experiment | Observation | η | Basis only | Quadrature | Data | Checkpoint |
|---|---|---:|---:|---:|---|---|
| `exact_lossless` | `exact` | `1.0` | False | False | `/home/pourushsharma/projects/pic/qf/PARAMS/data/exact_lossless.npz` | `/home/pourushsharma/projects/pic/qf/PARAMS/checkpoints/exact_lossless.pt` |
| `exact_lossy` | `exact` | `0.85` | False | False | `/home/pourushsharma/projects/pic/qf/PARAMS/data/exact_lossy.npz` | `/home/pourushsharma/projects/pic/qf/PARAMS/checkpoints/exact_lossy.pt` |
| `power_lossless` | `power` | `1.0` | False | False | `/home/pourushsharma/projects/pic/qf/PARAMS/data/power_lossless.npz` | `/home/pourushsharma/projects/pic/qf/PARAMS/checkpoints/power_lossless.pt` |
| `power_basis_lossless` | `power` | `1.0` | True | False | `/home/pourushsharma/projects/pic/qf/PARAMS/data/power_basis_lossless.npz` | `/home/pourushsharma/projects/pic/qf/PARAMS/checkpoints/power_basis_lossless.pt` |
| `power_lossy` | `power` | `0.85` | False | False | `/home/pourushsharma/projects/pic/qf/PARAMS/data/power_lossy.npz` | `/home/pourushsharma/projects/pic/qf/PARAMS/checkpoints/power_lossy.pt` |
| `power_quad_lossless` | `power` | `1.0` | False | True | `/home/pourushsharma/projects/pic/qf/PARAMS/data/power_quad_lossless.npz` | `/home/pourushsharma/projects/pic/qf/PARAMS/checkpoints/power_quad_lossless.pt` |
| `power_quad_lossy` | `power` | `0.85` | False | True | `/home/pourushsharma/projects/pic/qf/PARAMS/data/power_quad_lossy.npz` | `/home/pourushsharma/projects/pic/qf/PARAMS/checkpoints/power_quad_lossy.pt` |

## Command status

| Name | Status | Time (s) | stdout | stderr |
|---|---|---:|---|---|
| `verify_convention` | **PASS** | 2.1 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/sanity/verify_convention.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/sanity/verify_convention.stderr.log` |
| `data_pipeline_selftest` | **PASS** | 2.0 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/sanity/data_pipeline_selftest.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/sanity/data_pipeline_selftest.stderr.log` |
| `mesh_forward_selftest` | **PASS** | 1.7 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/sanity/mesh_forward_selftest.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/sanity/mesh_forward_selftest.stderr.log` |
| `data_exact_lossless` | **PASS** | 9.7 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/data/data_exact_lossless.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/data/data_exact_lossless.stderr.log` |
| `data_exact_lossy` | **PASS** | 10.0 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/data/data_exact_lossy.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/data/data_exact_lossy.stderr.log` |
| `data_power_lossless` | **PASS** | 8.9 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/data/data_power_lossless.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/data/data_power_lossless.stderr.log` |
| `data_power_basis_lossless` | **PASS** | 9.7 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/data/data_power_basis_lossless.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/data/data_power_basis_lossless.stderr.log` |
| `data_power_lossy` | **PASS** | 8.7 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/data/data_power_lossy.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/data/data_power_lossy.stderr.log` |
| `data_power_quad_lossless` | **PASS** | 9.3 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/data/data_power_quad_lossless.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/data/data_power_quad_lossless.stderr.log` |
| `data_power_quad_lossy` | **PASS** | 9.6 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/data/data_power_quad_lossy.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/data/data_power_quad_lossy.stderr.log` |
| `train_exact_lossless` | **PASS** | 202.5 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/training/train_exact_lossless.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/training/train_exact_lossless.stderr.log` |
| `train_exact_lossy` | **PASS** | 199.1 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/training/train_exact_lossy.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/training/train_exact_lossy.stderr.log` |
| `train_power_lossless` | **PASS** | 213.0 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/training/train_power_lossless.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/training/train_power_lossless.stderr.log` |
| `train_power_basis_lossless` | **PASS** | 217.1 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/training/train_power_basis_lossless.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/training/train_power_basis_lossless.stderr.log` |
| `train_power_lossy` | **PASS** | 220.8 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/training/train_power_lossy.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/training/train_power_lossy.stderr.log` |
| `train_power_quad_lossless` | **PASS** | 226.8 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/training/train_power_quad_lossless.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/training/train_power_quad_lossless.stderr.log` |
| `train_power_quad_lossy` | **PASS** | 222.0 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/training/train_power_quad_lossy.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/training/train_power_quad_lossy.stderr.log` |
| `infer_exact_lossless` | **PASS** | 7.3 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/inference/infer_exact_lossless.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/inference/infer_exact_lossless.stderr.log` |
| `infer_exact_lossy` | **PASS** | 9.0 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/inference/infer_exact_lossy.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/inference/infer_exact_lossy.stderr.log` |
| `infer_power_lossless` | **PASS** | 7.6 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/inference/infer_power_lossless.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/inference/infer_power_lossless.stderr.log` |
| `infer_power_basis_lossless` | **PASS** | 7.1 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/inference/infer_power_basis_lossless.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/inference/infer_power_basis_lossless.stderr.log` |
| `infer_power_lossy` | **PASS** | 7.4 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/inference/infer_power_lossy.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/inference/infer_power_lossy.stderr.log` |
| `infer_power_quad_lossless` | **PASS** | 7.7 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/inference/infer_power_quad_lossless.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/inference/infer_power_quad_lossless.stderr.log` |
| `infer_power_quad_lossy` | **PASS** | 7.7 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/inference/infer_power_quad_lossy.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/inference/infer_power_quad_lossy.stderr.log` |
| `ident_compare` | **PASS** | 2.0 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/identifiability/ident_compare.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/identifiability/ident_compare.stderr.log` |
| `ident_q1_basis_null` | **PASS** | 1.9 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/identifiability/ident_q1_basis_null.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/identifiability/ident_q1_basis_null.stderr.log` |
| `ident_q2_local_power` | **PASS** | 2.0 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/identifiability/ident_q2_local_power.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/identifiability/ident_q2_local_power.stderr.log` |
| `ident_q2_global_demo` | **PASS** | 13.2 | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/identifiability/ident_q2_global_demo.stdout.log` | `/home/pourushsharma/projects/pic/qf/PARAMS/ablation_runs/identifiability/ident_q2_global_demo.stderr.log` |

## Parsed training / inference metrics

| Run | val_obs_mse | network | single refine | multi-start | refined fidelity | label RMSE | matrix fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|
| `train_exact_lossless` | 0.07133 | 0.52589 | 0.98839 | 0.99948 |  |  |  |
| `train_exact_lossy` | 0.01016 | 0.56739 | 0.99341 | 0.99986 |  |  |  |
| `train_power_lossless` | 0.00612 | 0.93838 | 0.992 | 0.9999 |  |  |  |
| `train_power_basis_lossless` | 0.00113 | 0.98779 | 0.99981 | 1 |  |  |  |
| `train_power_lossy` | 0.00024 | 0.90946 | 0.99559 | 0.99997 |  |  |  |
| `train_power_quad_lossless` | 0.01242 | 0.88254 | 0.99739 | 0.99993 |  |  |  |
| `train_power_quad_lossy` | 0.00038 | 0.85296 | 0.99774 | 0.99999 |  |  |  |
| `infer_exact_lossless` |  |  |  |  | 0.999998 |  | 0.999998 |
| `infer_exact_lossy` |  |  |  |  | 1 |  | 1 |
| `infer_power_lossless` |  |  |  |  | 1 |  |  |
| `infer_power_basis_lossless` |  |  |  |  | 1 |  |  |
| `infer_power_lossy` |  |  |  |  | 1 |  |  |
| `infer_power_quad_lossless` |  |  |  |  | 1 |  |  |
| `infer_power_quad_lossy` |  |  |  |  | 1 |  |  |

## Identifiability conclusions

### Q1 — Basis probes only

The README predicts that basis-only power measurements leave approximately three locally unobservable directions in the 12 active `(θ, φ)` parameters, with expected Jacobian rank around 9/12. The `ident_q1_basis_null` run is the direct numerical test: its null-space vector should produce negligible first-order observation change. Label RMSE is not a valid uniqueness metric here.

### Q2 — Superposition probes

For default 8-probe power data, the README predicts generic local rank 12/12, but global non-uniqueness remains. The `ident_q2_local_power` run tests local rank; `ident_q2_global_demo` performs two independent cold-start solves and should show distinct parameter sets with essentially the same observation quality.

### Quadrature

Quadrature probes add the missing imaginary interference term, removing the relative-phase sign ambiguity in the measurement model. The expected experimental signature is tighter identifiability and less dependence on multi-start refinement, not necessarily exact recovery of the stored decomposition labels.

## README success criteria

- Convention check: expect `('confirmed', 'forward', 'row_reblock')` and `matched 5/5`.
- Exact-field and power models should be judged primarily by observation fidelity after refinement.
- For power models, `delta` is a true null direction and should not be scored as a recovered label.
- A high refined observation fidelity with a large label RMSE is evidence of a valid alternate preimage, not necessarily model failure.

## Raw logs

Every command's complete stdout/stderr is stored under `ablation_runs/`; `ablation_results.json` contains the machine-readable manifest and parsed metrics.
