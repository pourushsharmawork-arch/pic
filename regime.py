"""
ONE unified pipeline for both of your regimes, instead of two. The insight
that makes this possible: both "exact field" and "power only" observations
are real-valued vectors derived from the same forward model (build_mesh_lossy),
just through a different final readout step -- so the tandem loss, sin/cos
encoding, and refine machinery don't need to know or care which regime
they're in. What differs between regimes is exactly three things, captured
in `Regime` below:
  1. which readout function turns U_circuit into an observation vector
     (`U_to_measurement` -- full complex field, vs `probe_powers` -- powers
     only for a probe set)
  2. whether the network predicts delta at all (physically meaningless to
     predict it when observation="power" -- see mesh_forward.probe_powers'
     docstring for the proof; there's no point spending network capacity or
     loss gradient on a quantity that's provably unconstrained by the data)
  3. the known loss level eta (same lossy forward model either way -- eta=1.0
     recovers the lossless case exactly, verified numerically, so there's no
     separate lossless code path either)
  4. optional quadrature probes in power mode (Regime.quadrature=True adds
     (e_i + i e_j)/√2 probes, 48-D obs instead of 32)
  5. optional basis_only in power mode (I1..I4 powers only, 16-D obs) —
     use identifiability.py to quantify which parameter directions are
     unobservable vs gauge-only

This does NOT require a different model architecture. InverseNet's in_dim/
out_dim were already parameters; here they're just set from the regime
(regime.obs_dim, regime.sincos_dim). Same ResBlock trunk, same optimizer,
same OneCycleLR schedule, same tandem-loss idea, same multi-start refine.
If you ever hit a case where this genuinely isn't enough (e.g. probes that
can't be modeled as `probe @ U`, per-sample-varying probe sets, or needing
to predict eta itself), that's a reason to extend `Regime`/`simulate` -- not
to reach for a different training paradigm.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mesh_forward import (
    BLOCK_LABELS, build_mesh_lossy, make_power_probes, power_obs_dim, probe_powers, N_MODES,
)
from train_tandem import (
    U_to_measurement, angles_to_sincos, sincos_to_angles,
    InverseNet, DEVICE,
)


# --------------------------------------------------------------------------- #
# theta/phi-only sincos encode/decode (12 angles, no delta)
# --------------------------------------------------------------------------- #

def angles_to_sincos_tp(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """(...,6)/(...,6) -> (...,24). Same idea as angles_to_sincos but for the
    power regime, where delta isn't part of what the network predicts."""
    all_angles = torch.cat([theta, phi], dim=-1)  # (..., 12)
    s, c = torch.sin(all_angles), torch.cos(all_angles)
    return torch.stack([s, c], dim=-1).reshape(*all_angles.shape[:-1], 24)


def sincos_to_angles_tp(sc: torch.Tensor):
    pairs = sc.reshape(*sc.shape[:-1], 12, 2)
    s, c = pairs[..., 0], pairs[..., 1]
    angles = torch.atan2(s, c)
    return angles[..., :6], angles[..., 6:12]


# --------------------------------------------------------------------------- #
# Regime
# --------------------------------------------------------------------------- #

@dataclass
class Regime:
    observation: str = "exact"      # "exact" | "power"
    eta: float | torch.Tensor = 1.0  # known loss level(s); 1.0 = lossless
    basis_only: bool = False         # power only: I1..I4 powers, no superpositions
    quadrature: bool = False         # power only: add (e_i + i e_j)/√2 probes
    probes: torch.Tensor | None = None  # override probe set; default from flags
    predict_delta: bool = None       # default: True for "exact", False for "power"

    def __post_init__(self):
        assert self.observation in ("exact", "power")
        if self.basis_only and self.observation != "power":
            raise ValueError("basis_only applies only to observation='power'")
        if self.basis_only and self.quadrature:
            raise ValueError("basis_only and quadrature are incompatible")
        if self.predict_delta is None:
            self.predict_delta = (self.observation == "exact")
        if self.observation == "power" and self.predict_delta:
            raise ValueError(
                "predict_delta=True with observation='power' is a physically "
                "meaningless combination -- power measurements carry zero "
                "gradient information about delta (see mesh_forward.probe_powers). "
                "The network would just learn to output an arbitrary constant "
                "for it. Set predict_delta=False (the default for this regime)."
            )
        if self.observation == "power":
            if self.probes is None:
                self.probes = make_power_probes(self.quadrature, self.basis_only)
        else:
            self.probes = None

    @property
    def probe_config(self) -> str:
        """Short label for logging / identifiability tables."""
        if self.observation == "exact":
            return "exact_basis"
        if self.basis_only:
            return "power_basis_only"
        if self.quadrature:
            return "power_superposition_quadrature"
        return "power_superposition"

    @property
    def n_angles(self) -> int:
        return 16 if self.predict_delta else 12

    @property
    def sincos_dim(self) -> int:
        return self.n_angles * 2

    @property
    def obs_dim(self) -> int:
        if self.observation == "exact":
            return 32
        return self.probes.shape[0] * 4


def encode(theta: torch.Tensor, phi: torch.Tensor, delta: torch.Tensor, regime: Regime) -> torch.Tensor:
    if regime.predict_delta:
        return angles_to_sincos(theta, phi, delta)
    return angles_to_sincos_tp(theta, phi)


def decode(sc: torch.Tensor, regime: Regime):
    """Always returns (theta, phi, delta) -- delta is a zero placeholder (not
    a prediction) when regime.predict_delta is False, since simulate() needs
    *some* delta value to hand build_mesh_lossy even though the result won't
    depend on it (see mesh_forward.probe_powers)."""
    if regime.predict_delta:
        return sincos_to_angles(sc)
    theta, phi = sincos_to_angles_tp(sc)
    delta = torch.zeros(*theta.shape[:-1], 4, device=theta.device, dtype=theta.dtype)
    return theta, phi, delta


def simulate(theta: torch.Tensor, phi: torch.Tensor, delta: torch.Tensor, regime: Regime) -> torch.Tensor:
    """theta,phi,delta -> observation vector, exactly what the physical
    setup (simulated or real) would report for this regime."""
    U = build_mesh_lossy(theta, phi, delta, regime.eta)
    if regime.observation == "exact":
        return U_to_measurement(U)
    P = probe_powers(U, regime.probes.to(U.device))  # (..., n_probes, 4)
    return P.reshape(*P.shape[:-2], -1)


def make_model(regime: Regime, **kwargs) -> InverseNet:
    return InverseNet(in_dim=regime.obs_dim, out_dim=regime.sincos_dim, **kwargs).to(DEVICE)


# --------------------------------------------------------------------------- #
# Losses (regime-generic: both observation types are already real-valued
# vectors, so a single MSE-style loss form covers both -- no .abs()**2
# complex-diff special-casing needed here)
# --------------------------------------------------------------------------- #

def physics_loss_regime(pred_sc: torch.Tensor, target_obs: torch.Tensor, regime: Regime) -> torch.Tensor:
    theta, phi, delta = decode(pred_sc, regime)
    pred_obs = simulate(theta, phi, delta, regime)
    return ((pred_obs - target_obs) ** 2).sum(dim=-1).mean()


def param_aux_loss_regime(pred_sc: torch.Tensor, target_sc: torch.Tensor) -> torch.Tensor:
    return ((pred_sc - target_sc) ** 2).mean()


def unit_circle_reg_regime(pred_sc: torch.Tensor, regime: Regime) -> torch.Tensor:
    pairs = pred_sc.reshape(*pred_sc.shape[:-1], regime.n_angles, 2)
    norm_sq = (pairs ** 2).sum(dim=-1)
    return ((norm_sq - 1.0) ** 2).mean()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_model_regime(
    train_data: dict,   # needs "obs" (regime.obs_dim), "theta","phi","delta" (raw angles)
    val_data: dict,
    regime: Regime,
    mode: str = "physics",   # "physics" (tandem) | "param_only"
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 2e-3,
    aux_weight: float = 0.15,
    reg_weight: float = 0.01,
    seed: int = 0,
    verbose=True,
    model_kwargs: dict | None = None,
):
    torch.manual_seed(seed)
    model = make_model(regime, **(model_kwargs or {}))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n_train = train_data["obs"].shape[0]
    steps_per_epoch = max(1, n_train // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.1
    )

    def target_sincos(idx, data):
        return encode(data["theta"][idx], data["phi"][idx], data["delta"][idx], regime)

    history = {"train_loss": [], "val_obs_mse": []}
    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            obs = train_data["obs"][idx].to(DEVICE)
            target_sc = target_sincos(idx, train_data).to(DEVICE)

            pred_sc = model(obs)

            if mode == "physics":
                loss = physics_loss_regime(pred_sc, obs, regime)
                loss = loss + aux_weight * param_aux_loss_regime(pred_sc, target_sc)
            elif mode == "param_only":
                loss = param_aux_loss_regime(pred_sc, target_sc)
            else:
                raise ValueError(mode)
            loss = loss + reg_weight * unit_circle_reg_regime(pred_sc, regime)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            epoch_loss += loss.item()

        epoch_loss /= steps_per_epoch
        history["train_loss"].append(epoch_loss)

        model.eval()
        with torch.no_grad():
            obs_val = val_data["obs"].to(DEVICE)
            pred_sc = model(obs_val)
            theta, phi, delta = decode(pred_sc, regime)
            pred_obs = simulate(theta, phi, delta, regime)
            val_mse = ((pred_obs - obs_val) ** 2).mean().item()
        history["val_obs_mse"].append(val_mse)

        if val_mse < best_loss:
            best_loss = val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"[{regime.observation}/eta={regime.eta}] epoch {epoch:3d}  "
                  f"train_loss={epoch_loss:.4f}  val_obs_mse={val_mse:.5f}"
                  f"{'  *best*' if val_mse == best_loss else ''}")

    model.load_state_dict(best_state)
    history["best_val_obs_mse"] = best_loss
    if verbose:
        print(f"restored best checkpoint (val_obs_mse={best_loss:.5f})")
    return model, history


# --------------------------------------------------------------------------- #
# Refinement (single- and multi-start), regime-generic
# --------------------------------------------------------------------------- #

def refine_regime(model, obs, regime: Regime, n_steps=150, lr=0.03):
    model.eval()
    with torch.no_grad():
        pred_sc = model(obs.to(DEVICE))
        theta0, phi0, delta0 = decode(pred_sc, regime)

    theta = theta0.clone().requires_grad_(True)
    phi = phi0.clone().requires_grad_(True)
    params = [theta, phi]
    if regime.predict_delta:
        delta = delta0.clone().requires_grad_(True)
        params.append(delta)
    else:
        delta = delta0  # fixed zeros, not optimized -- nothing to gain, no gradient anyway

    opt = torch.optim.Adam(params, lr=lr)
    obs = obs.to(DEVICE)

    for _ in range(n_steps):
        opt.zero_grad()
        pred_obs = simulate(theta, phi, delta, regime)
        loss = ((pred_obs - obs) ** 2).sum(dim=-1).sum()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred_obs = simulate(theta, phi, delta, regime)
        obs_err = ((pred_obs - obs) ** 2).sum(dim=-1)
    return theta.detach(), phi.detach(), delta.detach(), obs_err


def refine_multistart_regime(model, obs, regime: Regime, n_starts=8, n_steps=150, lr=0.03,
                              perturb_std=0.4, seed=0):
    """Multi-start matters MORE for power measurements, especially without
    quadrature probes (sign ambiguities from cos-only interference, on top
    of the usual decomposition ambiguity). With quadrature=True the landscape
    is less multi-modal, but multi-start is still cheap insurance."""
    model.eval()
    obs = obs.to(DEVICE)
    with torch.no_grad():
        pred_sc = model(obs)
        theta0, phi0, delta0 = decode(pred_sc, regime)

    g = torch.Generator(device="cpu").manual_seed(seed)
    best_theta, best_phi, best_delta = theta0.clone(), phi0.clone(), delta0.clone()
    with torch.no_grad():
        best_err = ((simulate(theta0, phi0, delta0, regime) - obs) ** 2).sum(dim=-1)

    for s in range(n_starts):
        if s == 0:
            theta_s, phi_s = theta0.clone(), phi0.clone()
        else:
            noise = lambda shape: (torch.randn(shape, generator=g) * perturb_std).to(DEVICE)
            theta_s = theta0 + noise(theta0.shape)
            phi_s = phi0 + noise(phi0.shape)
        theta_s = theta_s.clone().requires_grad_(True)
        phi_s = phi_s.clone().requires_grad_(True)
        params = [theta_s, phi_s]
        if regime.predict_delta:
            delta_s = (delta0 if s == 0 else delta0 + noise(delta0.shape)).clone().requires_grad_(True)
            params.append(delta_s)
        else:
            delta_s = delta0

        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(n_steps):
            opt.zero_grad()
            pred_obs = simulate(theta_s, phi_s, delta_s, regime)
            loss = ((pred_obs - obs) ** 2).sum(dim=-1).sum()
            loss.backward()
            opt.step()

        with torch.no_grad():
            err_s = ((simulate(theta_s, phi_s, delta_s, regime) - obs) ** 2).sum(dim=-1)
            improved = err_s < best_err
            best_theta = torch.where(improved.unsqueeze(-1), theta_s.detach(), best_theta)
            best_phi = torch.where(improved.unsqueeze(-1), phi_s.detach(), best_phi)
            best_delta = torch.where(improved.unsqueeze(-1), delta_s.detach(), best_delta)
            best_err = torch.where(improved, err_s, best_err)

    return best_theta, best_phi, best_delta, best_err
