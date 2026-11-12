"""The PyTorch half of adapt_flow.py: the source -> target latent flow f_eta, built on
the official `flow_matching` library (AffineProbPath + CondOTScheduler for the path,
ODESolver for integration).  Works on plain latent arrays, so it never touches JAX.

The library does not cover the two terms specific to this method, so they live here:
  * the safety-weighted OT coupling, c(z0, z1) = ||z0 - z1||^2 + lambda_c |h_s - h_t|,
    which picks the (x_0, x_1) pairs the path is then sampled on;
  * the alignment loss through the frozen regressor h_hat_t, (h_s - h_hat_t(z1_hat))^2.

Same objective as flow_transfer.FlowTransfer.update_flow, except that Sinkhorn runs in
log space: exp(-cost / ot_reg) underflows at these cost scales.

Not imported from models/agents/__init__.py, so only adapt_flow.py needs torch.
"""

import os
import sys

import numpy as np
import torch
from torch import nn
from tqdm.auto import trange

# the library is used from its checkout rather than pip-installed
FM_ROOT = os.environ.get("FLOW_MATCHING_ROOT", "/home2/ayanroy/FM/flow_matching")
if FM_ROOT not in sys.path:
    sys.path.insert(0, FM_ROOT)

from flow_matching.path import AffineProbPath  # noqa: E402
from flow_matching.path.scheduler import CondOTScheduler  # noqa: E402
from flow_matching.solver import ODESolver  # noqa: E402
from flow_matching.utils import ModelWrapper  # noqa: E402

ADAPTIVE_ODE_METHODS = ("dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun")


def pick_device():
    """CUDA if it can really run: is_available() is also True when the torch build has
    no kernels for the node's GPU (e.g. cu128 on a GTX 1080 Ti), and the first op then fails."""
    if torch.cuda.is_available():
        try:
            torch.randint(2, (1,), device='cuda')
            return torch.device('cuda')
        except RuntimeError as e:
            print(f"[adapt] torch cannot run on {torch.cuda.get_device_name(0)}, using CPU for the flow: {str(e).splitlines()[0]}")
    return torch.device('cpu')


def _mlp(in_dim, hidden_dims, out_dim):
    layers, d = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, out_dim))


class VelocityField(ModelWrapper):
    """v_eta : Z x [0, 1] -> Z.  ODESolver passes t as a scalar, the path as a batch."""

    def __init__(self, latent_dim, hidden_dims):
        super().__init__(_mlp(latent_dim + 1, hidden_dims, latent_dim))

    def forward(self, x, t, **extras):
        t = t.expand(x.shape[0]) if t.dim() == 0 else t
        return self.model(torch.cat([x, t[:, None]], dim=-1))


def sinkhorn_log_coupling(cost, reg, num_iters):
    """log A* for the entropic OT problem with uniform marginals."""
    k0, k1 = cost.shape
    log_K = -cost / reg
    log_a, log_b = -np.log(k0), -np.log(k1)
    log_u, log_v = torch.zeros(k0, device=cost.device), torch.zeros(k1, device=cost.device)
    for _ in range(num_iters):
        log_u = log_a - torch.logsumexp(log_K + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u[:, None], dim=0)
    return log_u[:, None] + log_K + log_v[None, :]


def train_flow(z_s, h_s, z_t, h_t, tk, schedule, seed, device, log=None):
    """Fits h_hat_t on the target, then v_eta on pooled source -> target.
    Numpy in, the trained VelocityField out.  `tk`/`schedule` are train_config.py's
    transfer_kwargs/transfer_schedule; `log(phase, step, info)` is called every 1000 steps."""
    log = log or (lambda *_: None)
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    z_s, h_s = torch.as_tensor(z_s, device=device), torch.as_tensor(h_s, device=device)
    z_t, h_t = torch.as_tensor(z_t, device=device), torch.as_tensor(h_t, device=device)
    latent_dim, bs = z_s.shape[-1], schedule['ot_batch_size']

    def batch(n):
        return torch.randint(n, (bs,), device=device, generator=gen)

    regressor = _mlp(latent_dim, tk['hidden_dims'], 1).to(device)
    opt = torch.optim.Adam(regressor.parameters(), lr=tk['regressor_lr'])
    for i in trange(schedule['reg_steps'], desc='adapt: safety regressor'):
        idx = batch(len(z_t))
        loss = ((regressor(z_t[idx]).squeeze(-1) - h_t[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 1000 == 0:
            log('reg', i, {'reg_loss': loss.item()})
    regressor.requires_grad_(False)

    velocity = VelocityField(latent_dim, tk['hidden_dims']).to(device)
    path = AffineProbPath(scheduler=CondOTScheduler())
    opt = torch.optim.Adam(velocity.parameters(), lr=tk['velocity_lr'])
    for i in trange(schedule['flow_train_steps'], desc='adapt: flow map'):
        i_s, i_t = batch(len(z_s)), batch(len(z_t))
        z0, z1, hs, ht = z_s[i_s], z_t[i_t], h_s[i_s], h_t[i_t]

        cost = torch.cdist(z0, z1) ** 2 + tk['lambda_c'] * torch.abs(hs[:, None] - ht[None, :])
        log_A = sinkhorn_log_coupling(cost, tk['ot_reg'], tk['ot_iters']).reshape(-1)
        pair = torch.multinomial(torch.softmax(log_A, dim=0), bs, replacement=True, generator=gen)
        z0, hs = z0[pair // bs], hs[pair // bs]
        z1 = z1[pair % bs]

        t = torch.rand(bs, device=device, generator=gen)
        sample = path.sample(x_0=z0, x_1=z1, t=t)
        fm_loss = ((velocity(sample.x_t, sample.t) - sample.dx_t) ** 2).sum(-1).mean()

        z1_hat = z0 + velocity(z0, torch.zeros(bs, device=device))
        align_loss = ((hs - regressor(z1_hat).squeeze(-1)) ** 2).mean()

        loss = fm_loss + tk['lambda_h'] * align_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 1000 == 0:
            log('flow', i, {'fm_loss': fm_loss.item(), 'align_loss': align_loss.item(), 'loss': loss.item()})
    return velocity


@torch.no_grad()
def project(velocity, z, method, flow_steps, device, chunk=65536):
    """f_eta(z) = psi_eta(z, 1), integrated with the library's ODESolver."""
    solver = ODESolver(velocity_model=velocity)
    step_size = None if method in ADAPTIVE_ODE_METHODS else 1.0 / flow_steps
    time_grid = torch.tensor([0.0, 1.0])
    z = torch.as_tensor(z)
    out = [solver.sample(x_init=b.to(device), step_size=step_size, method=method, time_grid=time_grid).cpu().numpy()
           for b in z.split(chunk)]
    return np.concatenate(out, axis=0)
