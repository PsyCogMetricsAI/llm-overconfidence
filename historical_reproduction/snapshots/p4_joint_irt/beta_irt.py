#!/usr/bin/env python3
"""Standard Beta-IRT baseline, transplanted verbatim from the IRSL official repo.

This is the IRSL paper's Beta-IRT model applied to logprob data, used as the
"standard Beta-IRT baseline" in the logprob-vs-binary IRT comparison.

Equations are transplanted UNCHANGED from IRSL/utils.py. The only
edits permitted (per task spec) are: device forced to 'cpu', tqdm removed.
Each transplanted function is annotated with its source line numbers so the
mapping back to the repo is verifiable (file = truth).

Model semantics (additive convention theta + z):
  mu_{m,i} = sigmoid( alpha_i * ( theta_m + z_i ) )
    theta_m = model ability
    z_i     = item easiness  (= -difficulty)
    alpha_i = item discrimination
  responses y ~ Beta(mu*phi, (1-mu)*phi), precision phi fixed = 10.
"""
import torch

# Pure CPU per task spec.
torch.set_num_threads(4)


# ------------------------------------------------------------------
# Transplanted verbatim from IRSL/utils.py:540-543
#   def beta_nll(y, mu, phi)
# (unchanged)
# ------------------------------------------------------------------
def beta_nll(y, mu, phi):
    a = mu * phi
    b = (1.0 - mu) * phi
    return -((a - 1) * torch.log(y) + (b - 1) * torch.log1p(-y) - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)))


# ------------------------------------------------------------------
# Transplanted verbatim from IRSL/utils.py:202-260
#   def calibrate_2pl(...)
# Edits from original (only those permitted by task spec):
#   - device default 'cpu'
#   - tqdm progress bar removed (plain range)
#   - loss history (initial/final) returned for convergence diagnostics
# Equations (mu, beta_nll loss, optimizer setup, E/M alternation) UNCHANGED.
# ------------------------------------------------------------------
def calibrate_2pl(
    resmat: torch.Tensor,
    device: str = "cpu",
    loss_kind: str = "beta",
    max_epochs: int = 50,
    max_iter_per_epoch: int = 100,
    lr_theta: float = 0.1,
    lr_items: float = 0.01,
    phi: float = 10,
    clamp_eps: float = 1e-6,
):
    resmat = resmat.to(device)
    n_test_takers, n_items = resmat.shape
    thetas = torch.randn(n_test_takers, device=device, requires_grad=True)
    zs = torch.randn(n_items, device=device) * 0.1
    zs.requires_grad_()
    alphas = torch.ones(n_items, device=device) + torch.randn(n_items, device=device) * 0.1
    alphas.requires_grad_()
    phi_tensor = torch.tensor(phi, device=device)

    if loss_kind == "beta":
        def compute_loss(y, mu, mask):
            y = y.clamp(clamp_eps, 1 - clamp_eps)
            return beta_nll(y[mask], mu[mask], phi_tensor).mean()
    elif loss_kind == "binary":
        from torch.distributions import Bernoulli

        def compute_loss(y, mu, mask):
            return -Bernoulli(probs=mu[mask]).log_prob(y[mask]).mean()
    else:
        raise ValueError(f"Unknown loss_kind: {loss_kind}")

    theta_optimizer = torch.optim.AdamW([thetas], lr=lr_theta)
    item_optimizer = torch.optim.AdamW([alphas, zs], lr=lr_items)

    loss_history = []
    for epoch in range(max_epochs):
        # E-step: Update theta
        for iteration in range(max_iter_per_epoch):
            theta_optimizer.zero_grad()
            mask = ~torch.isnan(resmat)
            mu = torch.sigmoid(alphas[None, :] * (thetas[:, None] + zs[None, :]))
            theta_loss = compute_loss(resmat, mu, mask)
            theta_loss.backward()
            theta_optimizer.step()

        # M-step: Update item parameters
        for iteration in range(max_iter_per_epoch):
            item_optimizer.zero_grad()
            mask = ~torch.isnan(resmat)
            mu = torch.sigmoid(alphas[None, :] * (thetas[:, None] + zs[None, :]))
            item_loss = compute_loss(resmat, mu, mask)
            item_loss.backward()
            item_optimizer.step()

        loss_history.append(float((theta_loss + item_loss).detach().cpu()))

    return {
        'theta': thetas.detach().cpu().numpy(),
        'alpha': alphas.detach().cpu().numpy(),
        'z': zs.detach().cpu().numpy(),
        'loss_history': loss_history,
    }
