"""
PARAMS project ablation / reproduction driver.

Runs the experiments described in README.md:
  - sanity / convention checks
  - all documented data-generation configurations
  - unified tandem training across all generated datasets
  - optional legacy exact-field baselines (physics / physics_lossy / param_only)
  - per-checkpoint inference / refinement
  - identifiability Q1/Q2 demonstrations
  - one consolidated CLI report + JSON manifest
  - raw stdout/stderr for every command

Designed to be run from the PARAMS project root:
    python ablation.py
or:
    python ablation.py --epochs 100 --n-samples 10000 --n-refine-eval 200
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RUN_DIR = ROOT / "ablation_runs"


@dataclass
class CmdResult:
    name: str
    command: list[str]
    returncode: int
    elapsed_s: float
    stdout_file: str
    stderr_file: str
    stdout_tail: str
    stderr_tail: str
    skipped: bool = False


@dataclass
class Experiment:
    name: str
    observation: str
    eta: float
    basis_only: bool = False
    quadrature: bool = False
    data_file: Optional[str] = None
    checkpoint: Optional[str] = None
    kind: str = "unified"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the complete PARAMS README ablation.")
    p.add_argument("--python", default=sys.executable,
                   help="Python executable used for child commands.")
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--n-refine-eval", type=int, default=200)
    p.add_argument("--n-starts", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--skip-data", action="store_true",
                   help="Do not regenerate data; reuse the configured data paths.")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--skip-legacy", action="store_true")
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--skip-identifiability", action="store_true")
    p.add_argument("--skip-sanity", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Skip a command when its expected output already exists.")
    p.add_argument("--only", action="append",
                   help="Run only matching experiment names/substrings. Repeatable.")
    p.add_argument("--device", default=None,
                   help="Passed through as an environment hint; the project CLIs decide usage.")
    return p.parse_args()


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)


def run_cmd(
    name: str,
    cmd: list[str],
    log_dir: Path,
    *,
    env_extra: Optional[dict[str, str]] = None,
    expected_output: Optional[Path] = None,
    resume: bool = False,
    stop_on_error: bool = False,
) -> CmdResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"

    if resume and expected_output is not None and expected_output.exists():
        return CmdResult(
            name=name,
            command=cmd,
            returncode=0,
            elapsed_s=0.0,
            stdout_file=str(stdout_path),
            stderr_file=str(stderr_path),
            stdout_tail="[RESUMED: expected output exists]",
            stderr_tail="",
            skipped=True,
        )

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    t0 = time.perf_counter()
    print(f"\n>>> {name}")
    print("$ " + shlex.join(cmd))

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.perf_counter() - t0

    stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")

    tail_lines = 80
    stdout_tail = "\n".join(proc.stdout.splitlines()[-tail_lines:])
    stderr_tail = "\n".join(proc.stderr.splitlines()[-tail_lines:])

    print(f"[exit={proc.returncode}] [{elapsed:.1f}s]")
    if proc.returncode != 0:
        print(stderr_tail or stdout_tail)
        if stop_on_error:
            raise RuntimeError(f"{name} failed with exit code {proc.returncode}")

    return CmdResult(
        name=name,
        command=cmd,
        returncode=proc.returncode,
        elapsed_s=elapsed,
        stdout_file=str(stdout_path),
        stderr_file=str(stderr_path),
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


def select(name: str, only: Optional[list[str]]) -> bool:
    if not only:
        return True
    return any(x in name for x in only)


def dataset_specs() -> list[dict[str, Any]]:
    return [
        dict(
            name="exact_lossless",
            observation="exact",
            loss=1.0,
            flags=[],
            filename="exact_lossless.npz",
        ),
        dict(
            name="exact_lossy",
            observation="exact",
            loss=0.85,
            flags=[],
            filename="exact_lossy.npz",
        ),
        dict(
            name="power_lossless",
            observation="power",
            loss=1.0,
            flags=[],
            filename="power_lossless.npz",
        ),
        dict(
            name="power_basis_lossless",
            observation="power",
            loss=1.0,
            flags=["--basis_only"],
            filename="power_basis_lossless.npz",
        ),
        dict(
            name="power_lossy",
            observation="power",
            loss=0.85,
            flags=[],
            filename="power_lossy.npz",
        ),
        dict(
            name="power_quad_lossless",
            observation="power",
            loss=1.0,
            flags=["--quadrature"],
            filename="power_quad_lossless.npz",
        ),
        dict(
            name="power_quad_lossy",
            observation="power",
            loss=0.85,
            flags=["--quadrature"],
            filename="power_quad_lossy.npz",
        ),
    ]


def build_train_cmd(args: argparse.Namespace, exp: Experiment) -> list[str]:
    cmd = [
        args.python, "run_regime.py",
        "--data", str(exp.data_file),
        "--observation", exp.observation,
        "--eta", str(exp.eta),
        "--epochs", str(args.epochs),
        "--n_refine_eval", str(args.n_refine_eval),
        "--n_starts", str(args.n_starts),
        "--out", str(exp.checkpoint),
    ]
    if exp.basis_only:
        cmd.append("--basis_only")
    if exp.quadrature:
        cmd.append("--quadrature")
    return cmd


def build_infer_cmd(args: argparse.Namespace, exp: Experiment) -> list[str]:
    cmd = [
        args.python, "infer_interpret.py",
        "--checkpoint", str(exp.checkpoint),
        "--data", str(exp.data_file),
        "--observation", exp.observation,
        "--eta", str(exp.eta),
        "--index", str(args.index),
    ]
    if exp.basis_only:
        cmd.append("--basis_only")
    if exp.quadrature:
        cmd.append("--quadrature")
    return cmd


def parse_numbers(pattern: str, text: str) -> list[float]:
    vals = []
    for m in re.finditer(pattern, text, flags=re.I):
        try:
            vals.append(float(m.group(1)))
        except Exception:
            pass
    return vals


def parse_metrics(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    patterns = {
        "best_val_obs_mse": r"best\s+val_obs_mse\s*:\s*([0-9.eE+-]+)",
        "network_only_fidelity": r"network\s+only\s*:\s*([0-9.eE+-]+)",
        "network_only_min": r"network\s+only\s*:.*?\(min\s*([0-9.eE+-]+)\)",
        "single_refine_fidelity": r"\+\s*single-start\s+refine\s*:\s*([0-9.eE+-]+)",
        "single_refine_min": r"\+\s*single-start\s+refine\s*:.*?\(min\s*([0-9.eE+-]+)\)",
        "multistart_fidelity": r"\+\s*[0-9]+-start\s+refine\s*:?\s*([0-9.eE+-]+)",
        "multistart_min": r"\+\s*[0-9]+-start\s+refine\s*:.*?\(min\s*([0-9.eE+-]+)\)",
        "refined_obs_fidelity": r"refined.*?(?:observation\s+)?fidelity\s*[:=]\s*([0-9.eE+-]+)",
        "label_rmse": r"label\s+RMSE\s*[:=]\s*([0-9.eE+-]+)",
        "matrix_fidelity": r"matrix\s+fidelity\s*[:=]\s*([0-9.eE+-]+)",
        "rank": r"\brank\s*[:=]\s*([0-9]+)",
        "null_dim": r"null(?:[_\s-]*dim|ity)?\s*[:=]\s*([0-9]+)",
    }

    for key, pat in patterns.items():
        ms = parse_numbers(pat, text)
        if ms:
            metrics[key] = ms[-1]

    # Convenience flags based on documented output.
    metrics["convention_confirmed_5_5"] = "matched 5/5" in text
    metrics["looks_successful"] = any([
        metrics.get("multistart_fidelity", -1) >= 0.98,
        metrics.get("refined_obs_fidelity", -1) >= 0.98,
        metrics.get("matrix_fidelity", -1) >= 0.98,
    ])
    return metrics


def main() -> int:
    args = parse_args()
    ensure_dirs()

    all_results: list[CmdResult] = []
    all_metrics: dict[str, Any] = {}
    experiments: list[Experiment] = []

    # ------------------------------------------------------------------
    # 1. Sanity / README reproducibility checks
    # ------------------------------------------------------------------
    if not args.skip_sanity:
        sanity = [
            ("verify_convention", [args.python, "verify_convention.py"]),
            ("data_pipeline_selftest", [args.python, "data_pipeline.py"]),
            ("mesh_forward_selftest", [args.python, "mesh_forward.py"]),
        ]
        for name, cmd in sanity:
            if not select(name, args.only):
                continue
            res = run_cmd(
                name, cmd, RUN_DIR / "sanity",
                resume=False, stop_on_error=args.stop_on_error,
            )
            all_results.append(res)
            all_metrics[name] = parse_metrics(res.stdout_tail + "\n" + res.stderr_tail)

    # ------------------------------------------------------------------
    # 2. Generate every documented dataset configuration
    # ------------------------------------------------------------------
    for spec in dataset_specs():
        exp_name = "data_" + spec["name"]
        data_path = DATA_DIR / spec["filename"]

        if args.only and not select(exp_name, args.only):
            continue

        if not args.skip_data:
            cmd = [
                args.python, "datagen.py",
                "--observation", spec["observation"],
                "--loss", str(spec["loss"]),
                "--n_samples", str(args.n_samples),
                "--out", str(data_path),
            ]
            cmd.extend(spec["flags"])
            res = run_cmd(
                exp_name, cmd, RUN_DIR / "data",
                expected_output=data_path,
                resume=args.resume,
                stop_on_error=args.stop_on_error,
            )
            all_results.append(res)
            all_metrics[exp_name] = parse_metrics(res.stdout_tail + "\n" + res.stderr_tail)

        experiments.append(
            Experiment(
                name=spec["name"],
                observation=spec["observation"],
                eta=spec["loss"],
                basis_only="--basis_only" in spec["flags"],
                quadrature="--quadrature" in spec["flags"],
                data_file=str(data_path),
            )
        )

    # ------------------------------------------------------------------
    # 3. Train the unified tandem model for every dataset
    # ------------------------------------------------------------------
    trained: list[Experiment] = []
    if not args.skip_training:
        for exp in experiments:
            train_name = "train_" + exp.name
            exp.checkpoint = str(CKPT_DIR / f"{exp.name}.pt")
            if not select(train_name, args.only):
                continue

            cmd = build_train_cmd(args, exp)
            res = run_cmd(
                train_name, cmd, RUN_DIR / "training",
                expected_output=Path(exp.checkpoint),
                resume=args.resume,
                stop_on_error=args.stop_on_error,
            )
            all_results.append(res)
            all_metrics[train_name] = parse_metrics(res.stdout_tail + "\n" + res.stderr_tail)
            trained.append(exp)

    # ------------------------------------------------------------------
    # 4. Inference on one held-out sample for every trained model
    # ------------------------------------------------------------------
    if not args.skip_inference:
        for exp in trained:
            name = "infer_" + exp.name
            if not select(name, args.only):
                continue
            res = run_cmd(
                name, build_infer_cmd(args, exp), RUN_DIR / "inference",
                stop_on_error=args.stop_on_error,
            )
            all_results.append(res)
            all_metrics[name] = parse_metrics(res.stdout_tail + "\n" + res.stderr_tail)

    # ------------------------------------------------------------------
    # 5. Legacy exact-field ablations documented in README.
    # ------------------------------------------------------------------
    if not args.skip_legacy:
        legacy = [
            (
                "legacy_physics",
                [
                    args.python, "run_pipeline.py",
                    "--mode", "physics",
                    "--epochs", str(args.epochs),
                ],
            ),
            (
                "legacy_physics_lossy",
                [
                    args.python, "run_pipeline.py",
                    "--mode", "physics_lossy",
                    "--eta", "0.85",
                    "--epochs", str(args.epochs),
                ],
            ),
            (
                "legacy_param_only",
                [
                    args.python, "run_pipeline.py",
                    "--mode", "param_only",
                    "--epochs", str(args.epochs),
                ],
            ),
        ]
        for name, cmd in legacy:
            if not select(name, args.only):
                continue
            res = run_cmd(
                name, cmd, RUN_DIR / "legacy",
                stop_on_error=args.stop_on_error,
            )
            all_results.append(res)
            all_metrics[name] = parse_metrics(res.stdout_tail + "\n" + res.stderr_tail)

    # ------------------------------------------------------------------
    # 6. Identifiability experiments: Q1 + Q2
    # ------------------------------------------------------------------
    if not args.skip_identifiability:
        ident_cmds = [
            (
                "ident_compare",
                [args.python, "identifiability.py", "--compare"],
            ),
            (
                "ident_q1_basis_null",
                [
                    args.python, "identifiability.py",
                    "--observation", "power",
                    "--basis_only",
                    "--probe-null", "0",
                ],
            ),
            (
                "ident_q2_local_power",
                [
                    args.python, "identifiability.py",
                    "--observation", "power",
                ],
            ),
            (
                "ident_q2_global_demo",
                [
                    args.python, "identifiability.py",
                    "--observation", "power",
                    "--global-demo",
                ],
            ),
        ]
        for name, cmd in ident_cmds:
            if not select(name, args.only):
                continue
            res = run_cmd(
                name, cmd, RUN_DIR / "identifiability",
                stop_on_error=args.stop_on_error,
            )
            all_results.append(res)
            all_metrics[name] = parse_metrics(res.stdout_tail + "\n" + res.stderr_tail)

    # ------------------------------------------------------------------
    # 7. Build report
    # ------------------------------------------------------------------
    results_json = {
        "project_root": str(ROOT),
        "config": vars(args),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": [asdict(x) for x in experiments],
        "results": [asdict(x) for x in all_results],
        "metrics": all_metrics,
    }
    (RUN_DIR / "ablation_results.json").write_text(
        json.dumps(results_json, indent=2),
        encoding="utf-8",
    )

    report = make_report(args, experiments, all_results, all_metrics)
    (RUN_DIR / "ABLATION_REPORT.md").write_text(report, encoding="utf-8")

    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)
    print(f"\nJSON:   {RUN_DIR / 'ablation_results.json'}")
    print(f"REPORT: {RUN_DIR / 'ABLATION_REPORT.md'}")
    return 0 if all(r.returncode == 0 for r in all_results) else 2


def status_symbol(code: int, skipped: bool) -> str:
    if skipped:
        return "SKIP"
    return "PASS" if code == 0 else "FAIL"


def make_report(
    args: argparse.Namespace,
    experiments: list[Experiment],
    results: list[CmdResult],
    metrics: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# PARAMS Ablation Report")
    lines.append("")
    lines.append(f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- Samples per dataset: `{args.n_samples}`")
    lines.append(f"- Epochs: `{args.epochs}`")
    lines.append(f"- Refinement starts: `{args.n_starts}`")
    lines.append(f"- Refinement evaluation samples: `{args.n_refine_eval}`")
    lines.append("")

    lines.append("## Experiment matrix")
    lines.append("")
    lines.append("| Experiment | Observation | η | Basis only | Quadrature | Data | Checkpoint |")
    lines.append("|---|---|---:|---:|---:|---|---|")
    for e in experiments:
        lines.append(
            f"| `{e.name}` | `{e.observation}` | `{e.eta}` | "
            f"{e.basis_only} | {e.quadrature} | `{e.data_file}` | `{e.checkpoint or ''}` |"
        )
    lines.append("")

    lines.append("## Command status")
    lines.append("")
    lines.append("| Name | Status | Time (s) | stdout | stderr |")
    lines.append("|---|---|---:|---|---|")
    for r in results:
        lines.append(
            f"| `{r.name}` | **{status_symbol(r.returncode, r.skipped)}** | "
            f"{r.elapsed_s:.1f} | `{r.stdout_file}` | `{r.stderr_file}` |"
        )
    lines.append("")

    lines.append("## Parsed training / inference metrics")
    lines.append("")
    lines.append("| Run | val_obs_mse | network | single refine | multi-start | refined fidelity | label RMSE | matrix fidelity |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in metrics.items():
        if not any(k in m for k in [
            "best_val_obs_mse", "network_only_fidelity", "single_refine_fidelity",
            "multistart_fidelity", "refined_obs_fidelity", "label_rmse", "matrix_fidelity",
        ]):
            continue
        def f(k: str) -> str:
            v = m.get(k)
            return "" if v is None else f"{v:.6g}"
        lines.append(
            f"| `{name}` | {f('best_val_obs_mse')} | {f('network_only_fidelity')} | "
            f"{f('single_refine_fidelity')} | {f('multistart_fidelity')} | "
            f"{f('refined_obs_fidelity')} | {f('label_rmse')} | {f('matrix_fidelity')} |"
        )
    lines.append("")

    lines.append("## Identifiability conclusions")
    lines.append("")
    lines.append("### Q1 — Basis probes only")
    lines.append("")
    lines.append(
        "The README predicts that basis-only power measurements leave approximately "
        "three locally unobservable directions in the 12 active `(θ, φ)` parameters, "
        "with expected Jacobian rank around 9/12. The `ident_q1_basis_null` run is the "
        "direct numerical test: its null-space vector should produce negligible first-order "
        "observation change. Label RMSE is not a valid uniqueness metric here."
    )
    lines.append("")
    lines.append("### Q2 — Superposition probes")
    lines.append("")
    lines.append(
        "For default 8-probe power data, the README predicts generic local rank 12/12, "
        "but global non-uniqueness remains. The `ident_q2_local_power` run tests local rank; "
        "`ident_q2_global_demo` performs two independent cold-start solves and should show "
        "distinct parameter sets with essentially the same observation quality."
    )
    lines.append("")
    lines.append("### Quadrature")
    lines.append("")
    lines.append(
        "Quadrature probes add the missing imaginary interference term, removing the "
        "relative-phase sign ambiguity in the measurement model. The expected experimental "
        "signature is tighter identifiability and less dependence on multi-start refinement, "
        "not necessarily exact recovery of the stored decomposition labels."
    )
    lines.append("")

    lines.append("## README success criteria")
    lines.append("")
    lines.append(
        "- Convention check: expect `('confirmed', 'forward', 'row_reblock')` and `matched 5/5`."
    )
    lines.append(
        "- Exact-field and power models should be judged primarily by observation fidelity after refinement."
    )
    lines.append(
        "- For power models, `delta` is a true null direction and should not be scored as a recovered label."
    )
    lines.append(
        "- A high refined observation fidelity with a large label RMSE is evidence of a valid alternate preimage, not necessarily model failure."
    )
    lines.append("")

    lines.append("## Raw logs")
    lines.append("")
    lines.append(
        "Every command's complete stdout/stderr is stored under `ablation_runs/`; "
        "`ablation_results.json` contains the machine-readable manifest and parsed metrics."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())