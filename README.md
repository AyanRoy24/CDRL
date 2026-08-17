# CBF Transfer Pipeline: Ms → Mt Persistent Safety Transfer

This document explains how the codebase implements the transfer formulation
(`10 Aug.pdf`): moving a certified persistent-safety barrier from a source
CMDP `Ms = ⟨S, A, T, r, cs, γ, ρ0, Clim⟩` to a sparse-data target CMDP
`Mt = ⟨S, A, T, r, ct, γ, ρ0, Clim⟩` that shares dynamics/reward and differs
only in the cost function, **without retraining the barrier from scratch on
target data**.

Entry point: `train_offline.py --transfer=True`, which calls
`call_main_transfer()`. That function is the pipeline in miniature — it runs
Steps 1–4 below in order, each backed by one module.

```
Ds, Dt  ──▶  FlowTransfer (flow_transfer.py)     Steps 1, 2
                 │  autoencoder + velocity field
                 ▼
            CBFTransfer   (cbf_transfer.py)      Step 3
                 │  safety critic Q_h^phi fit on Zs AND f_eta(Zs)
                 │  reward critic/value/policy fit on Ds only
                 ▼
            evaluate_pr   (evaluation.py)        Step 4
                 │  eval on Mt, filtering pi_theta's actions via Q_h^phi
```

Everything hangs off two base classes that are unmodified: `CBF` (`cbf.py`)
is the single-domain IJCAI `CBFsmooth` actor-critic; `CBFTransfer` subclasses
it and only swaps the safety-critic input space from raw `(s, a)` to the
shared latent `Z`.

---

## Step 1 — Shared coordinate system: the encoder/decoder (`networks/transfer_nets.py`, `flow_transfer.py`)

`Encoder` (`phi_enc : S x A -> Z`) and `Decoder` (`d_dec : Z -> S x A`) are
bundled in `AutoEncoder` and trained jointly by
`FlowTransfer.update_reconstruction`, minimizing

```
L_reconstruction = E_(s,a)~Ds∪Dt [ || d_dec(phi_enc(s,a)) - (s,a) ||^2 ]
```

In `call_main_transfer`, this is the `transfer: encoder/decoder` loop: each
step draws a batch from `Ds` and one from `Dt`, concatenates them, and takes
one gradient step. This is the *only* place source and target observations
are pooled directly — everywhere downstream, `Z` is the shared space and the
two domains are only ever related through it. The encoder is later frozen
(`jax.lax.stop_gradient` wraps every `encode()` call in `update_flow`,
`update_h`, `pretrain_safety_regressor`) so the coordinate system doesn't
drift while later losses are fit on top of it.

## Step 2 — Safety-consistent flow map (`flow_transfer.py`)

This is the bulk of `FlowTransfer` and implements 2(a)–(h) of the PDF.

**2(g) — target safety regressor, pretrained first.**
`pretrain_safety_regressor` fits `h_hat_t^xi : Z -> R` on `Dt` alone:

```
xi <- argmin_xi E_(s,a)~Dt [ (h_hat_t^xi(phi_enc(s,a)) - h_t(s,a))^2 ]
```

`h_t(s, a) = c(s, a)` if `c(s,a) > 0` else `-1` — this indicator is already
computed upstream in `DSRLDataset.__init__` (`dsrl_datasets.py:54`):
`costs = where(cost > 0, cost_scale, -1)`, so `batch["costs"]` *is* `h(s,a)`
everywhere in this codebase, not the raw violation flag. This regressor is
trained to convergence and then only ever *read* (never updated) inside
`update_flow` — it is the fixed target that `L_align` measures against.

**2(a)–(b) — safety-weighted OT coupling.** `update_flow` encodes a source
and a target minibatch (`z0`, `z1`), then builds the cost matrix

```
c(z0_i, z1_j) = || z0_i - z1_j ||^2 + lambda_c * | hs_i - ht_j |
```

(`flow_transfer.py:175-177`) and solves for `A* = argmin_{A in Bk} sum A(i,j) c(z0_i,z1_j)`
via entropic-regularized Sinkhorn (`_sinkhorn`, uniform marginals over the
minibatch — the `Bk` of the PDF). `lambda_c` biases the transport plan
toward pairs that agree in immediate safety status, on top of raw latent
distance. As the PDF notes, `Bk` only constrains marginals, not injectivity,
so many-to-one coupling under `|Dt| << |Ds|` is expected, not a bug.

**2(c)–(d) — flow matching on sampled coupled pairs.** `_sample_coupling`
draws index pairs `(i_l, j_l) ~ A*` by treating the coupling matrix as a
joint distribution over `(i, j)` and sampling categorically — *only* coupled
pairs are used, never arbitrary cross-batch pairs. From each pair:

```
z_tau = (1-tau) z0 + tau z1,   u = z1 - z0
L_FM(eta) = mean || v_eta(z_tau, tau) - u ||^2
```

`_integrate_flow` (used elsewhere, at eval/projection time) traces
`psi_eta(z0, tau)` by fixed-step Euler integration of `v_eta`; `f_eta(z0) =
psi_eta(z0, 1)` is that trajectory's endpoint — the actual source→target map
used in Step 3. Note the *training* loss `L_FM` never calls `_integrate_flow`
itself; it only needs `v_eta` at the sampled `(z_tau, tau)`, per the flow-
matching objective. `_integrate_flow` is the multi-step version used once
the field is trained (or, via the single-Euler-step shortcut below, to
approximate it during training for the alignment term).

**2(e) — no noise prior.** The flow's source and target marginals are the
pushforward of `Ds` and `Dt` through the frozen encoder, i.e. exactly `z0`
and `z1` from the sampled minibatches above — never Gaussian noise. This is
implicit in the code: `update_flow` only ever samples `z0`/`z1` from encoded
dataset batches.

**2(h) — trajectory-level alignment loss.** Inside the same loss function
(`flow_transfer.py:190-200`), a single Euler half-step estimates where the
*current* `v_eta` sends `z0_pair`:

```
z1_hat = z0_pair + v_eta(z0_pair, 0)      # single-step Euler estimate of z1
L_align(eta) = mean( (hs_pair - h_hat_t(z1_hat))^2 )
loss = L_FM + lambda_h * L_align
```

Both terms are combined into one `loss_fn` and backpropagated through
`v_eta` in a single `jax.grad` call — `L_FM` shapes the coupling-level
endpoints, `L_align` continuously penalizes the *live* model's output
against the frozen `h_hat_t^xi`, so safety is preserved along the whole
trajectory, not just at the two dataset endpoints. `h_hat_t`'s own
parameters are held fixed here (no gradient flows into `safety_regressor` —
it was already trained and frozen in Step 2(g)).

## Step 3 — Barrier trained simultaneously on `Zs` and `f_eta(Zs)` (`cbf_transfer.py`)

`CBFTransfer` subclasses `CBF`, keeping the reward critic (`Q_r`, `V_r`) and
diffusion policy (`pi_theta`) *exactly* as in `cbf.py` — they train only on
raw `(s, a, r, s')` from `Ds`, which is what makes them domain-invariant
(the reward Bellman target only ever depends on shared `r` and shared
successor `s'`). Only the safety critic and the branch condition of the
reward critic route through `phi_enc` + `f_eta`.

`update_h` implements 3(a)–(c):

1. Encode the batch (`z = phi_enc(s,a)`, stop-gradient — the encoder stays
   frozen) and forward-project it: `z_proj = f_eta(z)` via `_integrate_flow`
   (also stop-gradient — Step 3 never backpropagates into the flow).
2. `V_h` is updated exactly as base `CBF`, via the expectile loss against
   `Q_h^target`'s max over the ensemble.
3. The same discrete-time CBF Bellman target
   `y_h = (1-gamma) h(s,a) + gamma * max(h(s,a), V_h(s'))` is fit at *both*
   points:

```python
qhs      = safe_critic(z)
qhs_proj = safe_critic(z_proj)
qh_loss  = mean( |qhs - y_h| + |qhs_proj - y_h| )
```

This is exactly 3(b)-(c): one network `Q_h^phi` is trained to be simultaneously
correct on `Zs` and `Zt` because the flow is assumed safety-consistent
(`h_t(f_eta(z)) = h_s(z)`), so both points share the same Bellman target by
construction — there's no separate "project then refit" stage; the single
`jax.grad` call in `Qh_loss` does both at once, which is the "simultaneous,
not post-hoc" contribution claimed in the PDF's Novelty section.

`update_r` mirrors `CBF.update_r` with one change: the branch condition
`Q_h^phi(z) <= 0` reads the *latent-valued* safety critic (`agent._encode`
then `safe_critic`) instead of the raw-`(s,a)` one — everything else
(`r_min`, the unsafe-branch target `r_min/(1-gamma) - Q_h`) is unchanged.

`update_actor` is byte-for-byte the base `CBF` actor update: advantage-
weighted regression of the diffusion policy against `Q_r`/`V_r`, touching
neither the encoder nor the flow.

## Step 4 — Evaluation on `Mt`, no target-time projection (`cbf_transfer.py:eval_actions`, `evaluation.py`)

At each target state `s`, `N` candidate actions are sampled from the
(source-trained, domain-invariant) diffusion policy `pi_theta`, each is
encoded to a target latent `z_i = phi_enc(s, a_i)`, and the one minimizing
`Q_h^phi(z_i)` is executed:

```python
a* = argmin_i Q_h^phi(phi_enc(s, a_i))
```

Critically, `Q_h^phi` is called *directly* on target latents — no
`f_eta`/projection step at eval time — because Step 3 already made the
critic valid on `Zt` by construction. This is what "no inverse-flow lookup
on the deployment path" means in the Novelty section, and it's the direct
code analogue of the PDF's Step 4.

`call_main_transfer` runs this via `evaluate_pr(agent, env_target,
eval_episodes)` (`evaluation.py`), which rolls out `agent.eval_actions` in
the target `PointRobot` env (`env_target`, built with
`target_hazard_position_list` / `target_hazard_size` from
`dataset_kwargs`) and reports return, cost, and the barrier
`coverage`/`validity` diagnostics (`check_coverage`, `check_valid` — the
discrete-time CBF condition `h(s') >= (1-alpha) h(s)`).

---

## Where each PDF equation lives

| PDF | Code |
|---|---|
| `L_reconstruction` | `FlowTransfer.update_reconstruction` |
| `A* = argmin_{A in Bk} ...`, `c(z0,z1)` | `FlowTransfer.update_flow` → `_sinkhorn`, cost matrix at `flow_transfer.py:175-178` |
| `psi_eta`, `f_eta(z0) = psi_eta(z0,1)` | `_integrate_flow` (`flow_transfer.py`) |
| `(i_l,j_l) ~ A*`, `z_tau`, `u`, `L_FM` | `_sample_coupling`, `update_flow` loss_fn |
| `h_hat_t^xi` regressor | `FlowTransfer.pretrain_safety_regressor` |
| `L_align` (single-step Euler `z1_hat`) | `update_flow` loss_fn, `flow_transfer.py:194-197` |
| Safety critic `Psi(h,v)` mixer, `y_h` | `CBF.update_h` / `CBFTransfer.update_h`, `target_qh = (1-gamma)*h + gamma*max(h, V_h)` |
| Simultaneous fit at `z` and `f_eta(z)` | `CBFTransfer.update_h`, `Qh_loss` |
| Reward critic `y_r` branch on `Q_h` sign | `CBF.update_r` / `CBFTransfer.update_r` |
| Policy AWR | `CBF.update_actor` (shared, unmodified) |
| Step 4 deployment filter | `CBFTransfer.eval_actions` |

## Config knobs (`train_config.py`)

`transfer_kwargs` — `latent_dim`, `hidden_dims`, `autoencoder_lr`,
`velocity_lr`, `regressor_lr`, `lambda_c` (safety weight inside OT cost),
`lambda_h` (alignment-loss weight), `ot_reg`/`ot_iters` (Sinkhorn), `flow_steps`
(Euler steps for `f_eta`). `transfer_schedule` — step counts for the three
pretraining phases (`recon_steps`, `reg_steps`, `flow_train_steps`) and the
minibatch size used for OT (`ot_batch_size`). `dataset_kwargs.pr_data` /
`pr_data_target` point at `Ds`/`Dt`; `target_hazard_position_list` /
`target_hazard_size` define the target `PointRobot`'s cost geometry (`None`
reuses the source geometry as a same-domain smoke test).

## Assumptions to keep in mind (per the PDF)

The persistent-safety guarantee is conditional on `h_t(f_eta(z)) = h_s(z)`
holding — this is only *approximately* enforced via `lambda_c`/`lambda_h`,
not guaranteed, so its residual error should be checked empirically on
held-out labeled `Dt` pairs, especially since `Clim = 0` tolerates no
violation. Reusing `pi_theta` on the target only holds exactly on the
jointly-safe region `C*_s ∩ C*_t`; outside it, reward efficiency degrades
but Step 4's filter still enforces safety. The alignment loss needs some
labeled target costs in `Dt` to fit `h_hat_t^xi` — if `Dt`'s cost coverage is
too sparse, the flow degrades toward an unregularized (safety-agnostic) one.
