#!/usr/bin/env python3
"""Joint dual-latent IRT (ability theta^A + overconfidence theta^C + per-model length slope gamma_m).

Implements the model form that S0 + an independent V0 double-verified in
`derivation_joint_irt.md` (Joint Dual-Latent IRT: Identification, Constraints,
Gradients, 2026-06-14). The equations, constraints, and gradient structure below
are copied from that derivation; section/equation citations are inline.

Model (derivation §1, §4):
  Eq1 (correctness c in {0,1}, Bernoulli 2PL):
      P(c_{im}=1) = sigmoid( a_i * (theta^A_m - b_i) )                 [derivation Eq1, line 141]
  Eq2 (confidence y in (0,1), Beta with phi fixed = 10):
      y_{im} ~ Beta(mu*phi, (1-mu)*phi)
      mu_{im} = sigmoid( alpha_i * (theta^A_m + theta^C_m + z_i) + gamma_m * L̃_i )   [Eq2, line 144]

  Model-side latents:
      theta^A_m  ability        -> enters BOTH equations (the only shared parameter; §1 note)
      theta^C_m  overconfidence -> Eq2 only
      gamma_m    per-model length slope (verbosity) -> Eq2 only  (§4: per-model, NOT global)
  Item-side params:
      a_i, b_i   Eq1 discrimination / difficulty
      alpha_i    Eq2 discrimination
      z_i        Eq2 item base (easiness); already soaks item-level length (§2.2)

Constraints (derivation §3, §4 tables):
  C1  mean_m theta^A_m = 0     (theta^A location)
  C2  mean_m theta^C_m = 0     (theta^C location + A<->C swap jointly with C1)
  C3  mean_i z_i       = 0     (uniform z DOF)
  C5  phi fixed = 10           (Beta precision)
  C6  mean_m gamma_m   = 0     (constant component of gamma_m)
  -- theta^A SCALE: no constraint. Pinned by Eq2's bare theta^A (§2.3). Do NOT
     z-score model main effects; only CENTER them (per task spec).
  Implemented by projecting each centered block onto its mean-zero subspace
  after every optimizer step (derivation §6, line 221: "subtract the block mean").

beta_nll is reused verbatim from beta_irt.py (which transplants it from the IRSL repo).
"""
import numpy as np
import torch

from beta_irt import beta_nll

# Pure CPU per task spec.
torch.set_num_threads(4)


def _center_(t: torch.Tensor) -> None:
    """In-place mean-zero projection of a 1-D parameter block (constraints C1/C2/C3/C6).

    Done under no_grad after each optimizer step. Subtracting the block mean is
    the projection onto the mean-zero subspace described in derivation §6 (line 221).
    """
    with torch.no_grad():
        t -= t.mean()


def fit_joint(
    C: np.ndarray,
    Y: np.ndarray,
    Lt: np.ndarray,
    *,
    n_epochs: int = 400,
    lr: float = 0.05,
    grad_clip=None,
    phi: float = 10.0,
    clamp_eps: float = 1e-6,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = False,
):
    """Fit the joint dual-latent IRT.

    Parameters
    ----------
    C : (M, I) array of correctness in {0,1}, may contain NaN (masked out of Eq1).
    Y : (M, I) array of confidence in (0,1) (Beta channel), may contain NaN (masked out of Eq2).
    Lt : (I,) array of per-item centered length L̃_i (same across models; derivation §1).
    n_epochs, lr : AdamW full-batch optimization (single optimizer over all params;
                   derivation §6 says alternating OR projection are both fine -- we use a
                   single joint AdamW with post-step centering, the autograd path S0 validated).
    phi : Beta precision, fixed = 10 (C5).
    clamp_eps : Y clamp range, matches beta_irt.calibrate_2pl default.

    Returns
    -------
    dict with keys:
      theta_A (M,), theta_C (M,), gamma_m (M,),
      a (I,), b (I,), alpha (I,), z (I,),
      loss_history (list[float]).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    C = np.asarray(C, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Lt = np.asarray(Lt, dtype=np.float64)
    M, I = C.shape
    assert Y.shape == (M, I), f"Y shape {Y.shape} != C shape {(M, I)}"
    assert Lt.shape == (I,), f"Lt shape {Lt.shape} != ({I},)"

    Ct = torch.tensor(C, device=device, dtype=torch.float64)
    Yt = torch.tensor(Y, device=device, dtype=torch.float64).clamp(clamp_eps, 1 - clamp_eps)
    Lti = torch.tensor(Lt, device=device, dtype=torch.float64)
    phi_t = torch.tensor(phi, device=device, dtype=torch.float64)

    # Masks: only score observed cells in each channel (derivation §1 line 52, missing masked).
    mask1 = ~torch.isnan(Ct)  # Eq1 (correctness)
    mask2 = ~torch.isnan(Yt)  # Eq2 (confidence)
    Ct0 = torch.nan_to_num(Ct, nan=0.0)  # zeroed where masked; never read under mask
    Yt0 = torch.nan_to_num(Yt, nan=0.5)

    # --- parameter init (style follows beta_irt.calibrate_2pl) ---
    theta_A = (torch.randn(M, device=device, dtype=torch.float64) * 0.1).requires_grad_()
    theta_C = (torch.randn(M, device=device, dtype=torch.float64) * 0.1).requires_grad_()
    gamma_m = (torch.randn(M, device=device, dtype=torch.float64) * 0.1).requires_grad_()
    a = (torch.ones(I, device=device, dtype=torch.float64)
         + torch.randn(I, device=device, dtype=torch.float64) * 0.1).requires_grad_()
    b = (torch.randn(I, device=device, dtype=torch.float64) * 0.1).requires_grad_()
    alpha = (torch.ones(I, device=device, dtype=torch.float64)
             + torch.randn(I, device=device, dtype=torch.float64) * 0.1).requires_grad_()
    z = (torch.randn(I, device=device, dtype=torch.float64) * 0.1).requires_grad_()

    opt = torch.optim.AdamW([theta_A, theta_C, gamma_m, a, b, alpha, z], lr=lr)

    n1 = float(mask1.sum().item())
    n2 = float(mask2.sum().item())

    def compute_loss():
        # Eq1: Bernoulli 2PL logit  a_i*(theta^A_m - b_i)   (derivation line 28)
        eta1 = a[None, :] * (theta_A[:, None] - b[None, :])
        p = torch.sigmoid(eta1)
        # Bernoulli NLL on observed cells
        ll1 = Ct0 * torch.log(p.clamp_min(1e-12)) + (1.0 - Ct0) * torch.log((1.0 - p).clamp_min(1e-12))
        nll1 = -(ll1 * mask1).sum() / max(n1, 1.0)

        # Eq2: Beta logit alpha_i*(theta^A+theta^C+z) + gamma_m*L̃_i  (derivation line 29/144)
        eta2 = alpha[None, :] * (theta_A[:, None] + theta_C[:, None] + z[None, :]) + gamma_m[:, None] * Lti[None, :]
        # NaN guard: at large I (HS 9586 items) eta2 can saturate -> sigmoid hits {0,1} in float64
        # -> beta_nll's lgamma(mu*phi)/lgamma((1-mu)*phi) = lgamma(0) = +inf -> NaN grad.
        # Clamp mirrors the Yt clamp (same clamp_eps); inert for well-behaved fits (ARC verified unchanged).
        mu = torch.sigmoid(eta2).clamp(clamp_eps, 1.0 - clamp_eps)
        # beta_nll reused verbatim from beta_irt.py (IRSL transplant)
        nll2_cells = beta_nll(Yt0, mu, phi_t)  # (M, I)
        nll2 = (nll2_cells * mask2).sum() / max(n2, 1.0)

        return nll1 + nll2

    loss_history = []
    for ep in range(n_epochs):
        opt.zero_grad()
        loss = compute_loss()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([theta_A, theta_C, gamma_m, a, b, alpha, z], grad_clip)
        opt.step()
        # Post-step constraint projection (C1/C2/C3/C6): center each block to mean zero.
        # theta^A SCALE intentionally left free (pinned by Eq2; §2.3) -- center only.
        _center_(theta_A)   # C1
        _center_(theta_C)   # C2
        _center_(z)         # C3
        _center_(gamma_m)   # C6
        loss_history.append(float(loss.detach().cpu()))
        if verbose and (ep % 50 == 0 or ep == n_epochs - 1):
            print(f"  epoch {ep:4d}  loss {loss_history[-1]:.6f}")

    return {
        "theta_A": theta_A.detach().cpu().numpy(),
        "theta_C": theta_C.detach().cpu().numpy(),
        "gamma_m": gamma_m.detach().cpu().numpy(),
        "a": a.detach().cpu().numpy(),
        "b": b.detach().cpu().numpy(),
        "alpha": alpha.detach().cpu().numpy(),
        "z": z.detach().cpu().numpy(),
        "loss_history": loss_history,
    }
