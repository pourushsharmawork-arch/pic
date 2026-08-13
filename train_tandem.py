"""
Tandem-network training pipeline for predicting mesh decomposition parameters
(theta, phi, delta) from the 32-real-number measurement vector y (= the 4x4
unitary's action on the 4 computational basis vectors, i.e. its columns).

Key ideas (grounded in the literature -- see README.md for citations):

1. TANDEM / PHYSICS LOSS (primary). Rather than supervising the raw angles
   directly, decode the network's output into angles, run them through the
   differentiable `build_mesh`, and compare the *reconstructed* U to the
   target. This is the standard fix for the many-to-one ambiguity inherent
   to *any* mesh parametrization (multiple angle sets -> same U): Ardizzone
   et al. 2019 frame this as the generic issue with ill-posed inverse
   problems, and Rausell-Campo et al. 2026 (arXiv:2607.09301) show it
   concretely for MZI meshes with "tandem neural networks". Optimizing in
   *matrix* space sidesteps having to pick a canonical parameter label.

2. sin/cos ANGLE ENCODING (secondary/auxiliary loss + decoding). All 16
   angles are regressed as (sin, cos) pairs and decoded via atan2, so the
   network never has to learn the 0/2*pi discontinuity.

3. MULTI-START + LOCAL REFINEMENT (inference-time). The network gives a fast
   amortized initial guess; a short (~100-200 step) Adam refinement directly
   on the physics loss squeezes out the residual error at ~1000x less cost
   than optimizing from a random start (this is the trick from the user's
   own idea list, and is exactly the "secondary optimization stage" that
   Rausell-Campo et al. note self-configuring/clear-box methods still need).
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn

from mesh_forward import build_mesh, build_mesh_lossy, unitarity_error

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Data representation
# --------------------------------------------------------------------------- #

def U_to_measurement(U: torch.Tensor) -> torch.Tensor:
    """
    U: (..., 4, 4) complex. Returns y: (..., 32) real.

    CORRECTED to match datagen.py exactly (this was guessed wrong in the
    first pass -- see the note below). datagen.py does:

        output_field_j = (I_j @ U_circuit)[0]     # I_j = one-hot ROW vector
        row = concat(..., output_field_j.real, output_field_j.imag, ...)

    I_j @ U_circuit for a one-hot row vector I_j picks out ROW j of
    U_circuit (not column j -- that would be U_circuit @ e_j). And within
    each basis vector it's a re-BLOCK then im-BLOCK (4 reals then 4
    imaginaries), not interleaved per-component. So the true layout is:
    for j in 0..3: [Re(U[j,:]) (4 numbers), Im(U[j,:]) (4 numbers)],
    concatenated over j -- 8 x 4 = 32.

    (Earlier version of this function used columns + per-component
    interleaving, i.e. a completely different linear map. If you trained
    anything against that version, the "U" and "sincos" your model actually
    learned to match do not correspond to your real measurement -- retrain
    from scratch after this fix, not just fine-tune.)
    """
    re = U.real  # (..., 4, 4): re[..., j, :] = Re(row j)
    im = U.imag
    per_row = torch.cat([re, im], dim=-1)  # (..., 4, 8): row j -> [re_j(4), im_j(4)]
    return per_row.reshape(*U.shape[:-2], 32)


def measurement_to_U(y: torch.Tensor) -> torch.Tensor:
    """Exact inverse of U_to_measurement. Purely a reshape -- makes no
    assumption about U being unitary, so it's valid for lossy (sub-unitary)
    circuits too."""
    per_row = y.reshape(*y.shape[:-1], 4, 8)
    re, im = per_row[..., :4], per_row[..., 4:]
    return torch.complex(re, im)


def angles_to_sincos(theta, phi, delta):
    """(...,6)/(...,6)/(...,4) -> (...,32) [sin,cos pairs for all 16 angles]."""
    all_angles = torch.cat([theta, phi, delta], dim=-1)  # (..., 16)
    s = torch.sin(all_angles)
    c = torch.cos(all_angles)
    return torch.stack([s, c], dim=-1).reshape(*all_angles.shape[:-1], 32)


def sincos_to_angles(sc: torch.Tensor):
    """(...,32) -> theta(...,6), phi(...,6), delta(...,4), decoded via atan2."""
    pairs = sc.reshape(*sc.shape[:-1], 16, 2)
    s, c = pairs[..., 0], pairs[..., 1]
    angles = torch.atan2(s, c)
    return angles[..., :6], angles[..., 6:12], angles[..., 12:16]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class ResBlock(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class InverseNet(nn.Module):
    """32-dim measurement -> 32-dim sin/cos angle encoding."""

    def __init__(self, in_dim=32, out_dim=32, width=256, n_blocks=4):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([ResBlock(width, width * 2) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, out_dim)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.out(self.out_norm(h))


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

def physics_loss(pred_sincos: torch.Tensor, U_target: torch.Tensor) -> torch.Tensor:
    theta, phi, delta = sincos_to_angles(pred_sincos)
    U_pred = build_mesh(theta, phi, delta)
    diff = U_pred - U_target
    return (diff.abs() ** 2).sum(dim=(-1, -2)).mean()


def param_aux_loss(pred_sincos: torch.Tensor, target_sincos: torch.Tensor) -> torch.Tensor:
    # MSE directly in sin/cos space -- periodic-safe, no wraparound discontinuity.
    return ((pred_sincos - target_sincos) ** 2).mean()


def unit_circle_reg(pred_sincos: torch.Tensor) -> torch.Tensor:
    pairs = pred_sincos.reshape(*pred_sincos.shape[:-1], 16, 2)
    norm_sq = (pairs ** 2).sum(dim=-1)
    return ((norm_sq - 1.0) ** 2).mean()


def reconstruction_fidelity(U_pred: torch.Tensor, U_target: torch.Tensor) -> torch.Tensor:
    """Average process fidelity |<U_pred, U_target>_F|^2 / N^2 in [0,1].
    ONLY valid when both U_pred and U_target are (near-)unitary -- the /n^2
    normalization assumes ||U||_F^2 = n for both. Use
    `normalized_overlap_fidelity` for lossy targets."""
    n = U_target.shape[-1]
    overlap = (U_pred.conj() * U_target).sum(dim=(-1, -2))
    return ((overlap.abs() ** 2) / (n ** 2))


def normalized_overlap_fidelity(U_pred: torch.Tensor, U_target: torch.Tensor) -> torch.Tensor:
    """Cosine-similarity-squared in Frobenius inner-product space:
    |<U_pred,U_target>_F|^2 / (||U_pred||_F^2 ||U_target||_F^2), in [0,1].
    Reduces exactly to `reconstruction_fidelity` when both matrices are
    unitary (||.||_F^2 = n for each, so the denominator is n^2) -- use this
    one instead whenever U_target may be lossy/non-unitary, since it
    normalizes by each matrix's *actual* norm rather than assuming n."""
    overlap = (U_pred.conj() * U_target).sum(dim=(-1, -2))
    norm_pred_sq = (U_pred.abs() ** 2).sum(dim=(-1, -2))
    norm_target_sq = (U_target.abs() ** 2).sum(dim=(-1, -2))
    return (overlap.abs() ** 2) / (norm_pred_sq * norm_target_sq).clamp_min(1e-12)


def physics_loss_lossy(pred_sincos: torch.Tensor, U_target: torch.Tensor,
                        eta: torch.Tensor) -> torch.Tensor:
    """Tandem loss for lossy data: reconstructs the predicted params through
    the LOSSY forward model (matching datagen.py's actual physical circuit)
    instead of the ideal unitary one, so both sides of the comparison live on
    the same (non-unitary) manifold. `eta` must be the same loss level(s)
    used to generate U_target -- it's a known experimental setting, not
    something the network predicts."""
    theta, phi, delta = sincos_to_angles(pred_sincos)
    U_pred = build_mesh_lossy(theta, phi, delta, eta)
    diff = U_pred - U_target
    return (diff.abs() ** 2).sum(dim=(-1, -2)).mean()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_model(
    train_data,
    val_data,
    mode: str = "physics",   # "physics" (tandem, lossless) | "physics_lossy" | "param_only"
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 2e-3,
    aux_weight: float = 0.15,
    reg_weight: float = 0.01,
    eta=None,   # required for mode="physics_lossy": scalar or (...,6,4) loss level(s)
    seed: int = 0,
    verbose=True,
):
    if mode == "physics_lossy" and eta is None:
        raise ValueError(
            "mode='physics_lossy' needs `eta` (the loss level(s) used to generate "
            "this data, e.g. the same `loss=` value passed to datagen.py's "
            "sample_programmed_unitary_statistics). It's a known experimental "
            "setting, not something inferred from the data."
        )
    fid_fn = normalized_overlap_fidelity if mode == "physics_lossy" else reconstruction_fidelity
    torch.manual_seed(seed)
    model = InverseNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n_train = train_data["y"].shape[0]
    steps_per_epoch = max(1, n_train // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.1
    )

    history = {"train_loss": [], "val_fidelity": [], "val_recon_err": []}
    best_fid = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            y = train_data["y"][idx].to(DEVICE)
            U_t = train_data["U"][idx].to(DEVICE)
            target_sc = train_data["sincos"][idx].to(DEVICE)

            pred_sc = model(y)

            if mode == "physics":
                loss = physics_loss(pred_sc, U_t)
                loss = loss + aux_weight * param_aux_loss(pred_sc, target_sc)
            elif mode == "physics_lossy":
                loss = physics_loss_lossy(pred_sc, U_t, eta)
                loss = loss + aux_weight * param_aux_loss(pred_sc, target_sc)
            elif mode == "param_only":
                loss = param_aux_loss(pred_sc, target_sc)
            else:
                raise ValueError(mode)
            loss = loss + reg_weight * unit_circle_reg(pred_sc)

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
            y_val = val_data["y"].to(DEVICE)
            U_val = val_data["U"].to(DEVICE)
            pred_sc = model(y_val)
            th, ph, de = sincos_to_angles(pred_sc)
            U_pred = build_mesh_lossy(th, ph, de, eta) if mode == "physics_lossy" else build_mesh(th, ph, de)
            fid = fid_fn(U_pred, U_val).mean().item()
            recon_err = torch.linalg.matrix_norm(U_pred - U_val, ord="fro").mean().item()
        history["val_fidelity"].append(fid)
        history["val_recon_err"].append(recon_err)

        if fid > best_fid:
            best_fid = fid
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"[{mode}] epoch {epoch:3d}  train_loss={epoch_loss:.4f}  "
                  f"val_fidelity={fid:.4f}  val_recon_frob={recon_err:.4f}"
                  f"{'  *best*' if fid == best_fid else ''}")

    model.load_state_dict(best_state)
    history["best_val_fidelity"] = best_fid
    if verbose:
        print(f"[{mode}] restored best checkpoint (val_fidelity={best_fid:.4f})")
    return model, history


# --------------------------------------------------------------------------- #
# Post-hoc local refinement (network prediction as warm start)
# --------------------------------------------------------------------------- #

def refine(model, y, U_target, n_steps=150, lr=0.03, eta=None):
    """eta=None -> lossless build_mesh (original behavior). Pass eta (the known
    loss level(s) used to generate U_target) to refine against the correct
    lossy forward model instead -- refining a lossy target against the
    lossless model has the same structural bias problem as training against
    it does (see mesh_forward.build_mesh_lossy's module comment), so this
    matters just as much at inference time as it does during training."""
    model.eval()
    with torch.no_grad():
        pred_sc = model(y.to(DEVICE))
        theta0, phi0, delta0 = sincos_to_angles(pred_sc)

    theta = theta0.clone().requires_grad_(True)
    phi = phi0.clone().requires_grad_(True)
    delta = delta0.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta, phi, delta], lr=lr)
    U_target = U_target.to(DEVICE)
    fid_fn = normalized_overlap_fidelity if eta is not None else reconstruction_fidelity
    forward = (lambda th, ph, de: build_mesh_lossy(th, ph, de, eta)) if eta is not None else build_mesh

    for _ in range(n_steps):
        opt.zero_grad()
        U_pred = forward(theta, phi, delta)
        loss = ((U_pred - U_target).abs() ** 2).sum(dim=(-1, -2)).sum()
        loss.backward()
        opt.step()

    with torch.no_grad():
        U_pred = forward(theta, phi, delta)
        fid = fid_fn(U_pred, U_target)
        err = torch.linalg.matrix_norm(U_pred - U_target, ord="fro")
    return theta.detach(), phi.detach(), delta.detach(), fid, err


def refine_multistart(model, y, U_target, n_starts=8, n_steps=150, lr=0.03,
                       perturb_std=0.4, seed=0, eta=None):
    """
    True multi-start version of `refine`: the network's prediction is used as
    ONE of the starts (perturb_std=0 for that one), plus (n_starts-1) more
    starts obtained by perturbing it in angle-space, each independently
    refined by gradient descent on the physics loss. Per-sample, the best
    final fidelity across starts is kept.

    Why this is worth the extra compute over plain `refine`: the tandem loss
    is non-convex in angle-space (many-to-one param->U map means the loss
    surface has multiple basins), so if the network's single point-estimate
    lands near a bad local basin for some samples, single-start gradient
    descent gets stuck there. A handful of nearby random restarts is cheap
    (this whole step is still ~1000x cheaper than a fully random cold start,
    per the docstring at the top of this file) and catches most of those
    stragglers.

    Returns theta, phi, delta, fid, err each shaped (n_starts_kept=1, B, ...)
    i.e. already reduced to the best start per sample -- same shapes as
    plain `refine`'s output.
    """
    model.eval()
    B = y.shape[0]
    device = DEVICE
    U_target = U_target.to(device)
    fid_fn = normalized_overlap_fidelity if eta is not None else reconstruction_fidelity
    forward = (lambda th, ph, de: build_mesh_lossy(th, ph, de, eta)) if eta is not None else build_mesh

    with torch.no_grad():
        pred_sc = model(y.to(device))
        theta0, phi0, delta0 = sincos_to_angles(pred_sc)  # each (B, k)

    g = torch.Generator(device="cpu").manual_seed(seed)

    best_theta = theta0.clone()
    best_phi = phi0.clone()
    best_delta = delta0.clone()
    with torch.no_grad():
        best_fid = fid_fn(forward(theta0, phi0, delta0), U_target)

    for s in range(n_starts):
        if s == 0:
            theta_s, phi_s, delta_s = theta0.clone(), phi0.clone(), delta0.clone()
        else:
            noise = lambda shape: torch.randn(shape, generator=g).to(device) * perturb_std
            theta_s = theta0 + noise(theta0.shape)
            phi_s = phi0 + noise(phi0.shape)
            delta_s = delta0 + noise(delta0.shape)

        theta_s = theta_s.clone().requires_grad_(True)
        phi_s = phi_s.clone().requires_grad_(True)
        delta_s = delta_s.clone().requires_grad_(True)
        opt = torch.optim.Adam([theta_s, phi_s, delta_s], lr=lr)

        for _ in range(n_steps):
            opt.zero_grad()
            U_pred = forward(theta_s, phi_s, delta_s)
            # per-sample sum, but summed loss still gives correct per-sample
            # grads since samples don't interact -- same trick as `refine`.
            loss = ((U_pred - U_target).abs() ** 2).sum(dim=(-1, -2)).sum()
            loss.backward()
            opt.step()

        with torch.no_grad():
            U_pred = forward(theta_s, phi_s, delta_s)
            fid_s = fid_fn(U_pred, U_target)
            improved = fid_s > best_fid
            best_theta = torch.where(improved.unsqueeze(-1), theta_s.detach(), best_theta)
            best_phi = torch.where(improved.unsqueeze(-1), phi_s.detach(), best_phi)
            best_delta = torch.where(improved.unsqueeze(-1), delta_s.detach(), best_delta)
            best_fid = torch.where(improved, fid_s, best_fid)

    with torch.no_grad():
        U_pred = forward(best_theta, best_phi, best_delta)
        best_err = torch.linalg.matrix_norm(U_pred - U_target, ord="fro")

    return best_theta, best_phi, best_delta, best_fid, best_err