# ===========================================================================
# Standard Stochastic Gradient Descent on the FSC log-likelihood.
#
# Unlike the previous version, this does NOT reuse train_em's E-step
# statistics (gamma, xi) or the Fisher/EM-gradient identity. Instead it:
#
#   1. Computes NLL(theta) directly via the forward algorithm in log-space
#      (no backward pass, no gamma/xi -- we only need the scalar log-
#      likelihood, not the posterior over memory states).
#   2. Gets dL/d(theta) via PyTorch autodiff through that forward pass.
#   3. Takes a step with torch.optim.Adam (default) or SGD(momentum/
#      Nesterov), plus a configurable LR scheduler.
#
# This is mathematically a different route to the same quantity computed
# by the Fisher-identity version (both are exact gradients of the true
# log-likelihood), but the implementation shares no code or intermediate
# statistics with train_em -- there's no Q-function, no E-step, nothing
# EM-shaped in this file. That's the point of this rewrite.
#
# Model (for reference, since forward_backward's semantics were implicit
# in the original E-step code):
#   m_0        ~ rho                      m in [0, M)
#   a_t        ~ pi(. | m_t)               a in [0, A)   (actions are observed)
#   m_{t+1}    ~ g(. | m_t, a_t, y_t)      y in [0, Y)   (obs are observed/exogenous)
#   log p(actions, obs | theta) computed by the forward algorithm below.
# ===========================================================================
import os
# Torch can oversubscribe cores across parallel restart workers the same
# way numba/MKL did -- cap per-process threads before torch initializes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np
import torch
from joblib import Parallel, delayed

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Parameterization (torch analogue of unpack_theta_np / theta_from_params)
# ---------------------------------------------------------------------------

def pack_theta(rho, pi, g):
    return np.concatenate([rho.ravel(), pi.ravel(), g.ravel()])


def init_theta(M, A, Y, scale=0.3, seed=None):
    rng = np.random.default_rng(seed)
    dim = M + A * M + M * M * A * Y
    return scale * rng.standard_normal(dim)


def theta_from_params(rho, pi, g, eps=1e-10):
    """Warm-start helper: probabilities -> logits. Softmax is shift-
    invariant, so log(p) reproduces p exactly on unpack -- faithful warm
    start, not an approximation."""
    rho_logits = np.log(rho + eps)
    pi_logits = np.log(pi + eps)
    g_logits = np.log(g + eps)
    return np.concatenate([rho_logits.ravel(), pi_logits.ravel(), g_logits.ravel()])


def unpack_theta_logits(theta, M, A, Y):
    """Split a flat theta tensor into the three raw logit blocks. Kept
    separate from softmax so the log-space forward algorithm can use
    log_softmax directly (more numerically stable than log(softmax(x)))."""
    idx = 0
    rho_logits = theta[idx:idx + M]; idx += M
    pi_logits = theta[idx:idx + A * M].reshape(A, M); idx += A * M
    g_logits = theta[idx:].reshape(M, M, A, Y)
    return rho_logits, pi_logits, g_logits


def unpack_theta_probs(theta, M, A, Y):
    """Same as unpack_theta_np but for torch tensors -- returns actual
    probabilities (e.g. for eval / warm-starting the next stage), not logits."""
    rho_logits, pi_logits, g_logits = unpack_theta_logits(theta, M, A, Y)
    rho = torch.softmax(rho_logits, dim=0)
    pi = torch.softmax(pi_logits, dim=0)
    g = torch.softmax(g_logits, dim=0)
    return rho, pi, g


# ---------------------------------------------------------------------------
# Cross-mc warm start: embed a converged (rho, pi, g) from mc_prev into the
# larger mc_new problem. Same logic as the MAPSO/EM versions -- shared
# memory nodes keep their learned probabilities, new boundary nodes get a
# fresh random init. Defined here too so this module is usable standalone;
# operates purely in probability space (like the EM version), independent
# of the torch logit machinery above -- theta_from_params converts the
# result to logits for train_sgd's init_rho/init_pi/init_g arguments.
# ---------------------------------------------------------------------------
def embed_fsc_params_across_mc(rho_prev, pi_prev, g_prev, mc_prev, mc_new,
                                A=3, Y=2, seed=None):
    '''Embed a converged (rho, pi, g) from mc_prev into mc_new's dimension.
    Shared nodes (|m| <= mc_prev) keep their learned probabilities; new
    boundary nodes get a fresh Dirichlet-random init. Same contract as the
    EM/MAPSO versions -- used here to build warm starts for train_sgd.'''
    def random_fsc_parameters(M, A, Y, seed=None):
        rng = np.random.default_rng(seed)
        rho = rng.dirichlet(np.ones(M))
        pi = rng.dirichlet(np.ones(A), size=M).T
        g = np.zeros((M, M, A, Y))
        for a in range(A):
            for y in range(Y):
                g[:, :, a, y] = rng.dirichlet(np.ones(M), size=M).T
        return rho, pi, g

    M_prev = 2 * mc_prev + 1
    M_new = 2 * mc_new + 1
    if M_new < M_prev:
        raise ValueError("embed_fsc_params_across_mc only supports mc_new >= mc_prev.")

    ms_prev = list(range(-mc_prev, mc_prev + 1))
    ms_new = list(range(-mc_new, mc_new + 1))
    idx_prev = {m: i for i, m in enumerate(ms_prev)}
    idx_new = {m: i for i, m in enumerate(ms_new)}

    rho_new, pi_new, g_new = random_fsc_parameters(M_new, A, Y, seed=seed)

    for m in ms_prev:
        i_prev, i_new = idx_prev[m], idx_new[m]
        rho_new[i_new] = rho_prev[i_prev]
        pi_new[:, i_new] = pi_prev[:, i_prev]

    for m in ms_prev:
        for mp in ms_prev:
            i_prev, ip_prev = idx_prev[m], idx_prev[mp]
            i_new, ip_new = idx_new[m], idx_new[mp]
            g_new[ip_new, i_new, :, :] = g_prev[ip_prev, i_prev, :, :]

    rho_new = rho_new / rho_new.sum()
    return rho_new, pi_new, g_new


# ---------------------------------------------------------------------------
# Forward algorithm (log-space, differentiable) -- replaces forward_backward
# for training. We only need the scalar log-likelihood; autodiff handles
# the gradient, so there's no beta pass and no gamma/xi to compute.
# ---------------------------------------------------------------------------

def sequence_nll_torch(actions, obs, log_rho, log_pi, log_g):
    """
    log_rho : (M,)
    log_pi  : (A, M)   log_pi[a, m] = log P(action=a | node=m)
    log_g   : (M, M, A, Y)  log_g[mp, m, a, y] = log P(next node=mp | m, a, y)
    actions, obs : 1-D int sequences, length T
    Returns scalar NLL (torch tensor, differentiable).
    """
    T = len(actions)
    a0 = actions[0]
    log_alpha = log_rho + log_pi[a0]  # (M,) over m_0

    for t in range(T - 1):
        a_t, y_t, a_next = actions[t], obs[t], actions[t + 1]
        # log_g_slice[mp, m] = log P(m_{t+1}=mp | m_t=m, a_t, y_t)
        log_g_slice = log_g[:, :, a_t, y_t]                 # (M, M) -> (mp, m)
        # sum over m in log-space: logsumexp_m( log_alpha[m] + log_g_slice[mp, m] )
        log_alpha = torch.logsumexp(log_g_slice + log_alpha.unsqueeze(0), dim=1)
        log_alpha = log_alpha + log_pi[a_next]               # emit a_{t+1} from node mp

    ll = torch.logsumexp(log_alpha, dim=0)
    return -ll


def batch_nll_torch(batch, log_rho, log_pi, log_g, reduction="mean"):
    total = 0.0
    for actions, obs in batch:
        total = total + sequence_nll_torch(actions, obs, log_rho, log_pi, log_g)
    if reduction == "mean":
        return total / max(1, len(batch))
    return total


# ---------------------------------------------------------------------------
# Optimizer factory -- SGD or Adam, since Adam's per-parameter adaptive
# step size handles this problem's very uneven gradient scale (a rarely-
# visited (m,a,y) cell in g_logits gets a tiny, noisy gradient signal
# while common cells get large ones) far better than plain SGD+momentum,
# which uses one global step size for every logit.
# ---------------------------------------------------------------------------

def make_optimizer(theta, kind, lr, momentum=0.9, nesterov=True,
                    betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    kind = kind.lower()
    if kind == "adam":
        return torch.optim.Adam([theta], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if kind == "sgd":
        return torch.optim.SGD([theta], lr=lr, momentum=momentum, nesterov=nesterov)
    raise ValueError(f"Unknown optimizer kind: {kind!r}")


# ---------------------------------------------------------------------------
# Scheduler factory -- configurable rather than hard-coded, per your request
# ---------------------------------------------------------------------------

def make_scheduler(optimizer, kind, **kwargs):
    """
    kind : None | "step" | "exponential" | "cosine"
    kwargs are forwarded to the underlying torch scheduler, e.g.:
      step:        step_size=10, gamma=0.5
      exponential: gamma=0.99
      cosine:      T_max=50, eta_min=0.0
    """
    if kind is None:
        return None
    kind = kind.lower()
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **{"step_size": 10, "gamma": 0.5, **kwargs})
    if kind == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **{"gamma": 0.99, **kwargs})
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **{"T_max": 50, "eta_min": 0.0, **kwargs})
    raise ValueError(f"Unknown scheduler kind: {kind!r}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_sgd(dataset, M, A=3, Y=2, n_epochs=50, batch_size=16, lr=0.01,
              optimizer_kind="adam", momentum=0.9, nesterov=True,
              betas=(0.9, 0.999), adam_eps=1e-8, weight_decay=0.0,
              scheduler_kind="step", scheduler_kwargs=None,
              tol=1e-10, patience=None, restore_best=True,
              verbose=True, seed=None, eval_every=1,
              init_rho=None, init_pi=None, init_g=None, init_jitter=0.0,
              dataset_nll_fn=None):
    """
    Standard mini-batch gradient descent: torch.optim.Adam (default) or
    torch.optim.SGD(momentum=..., nesterov=...), stepping on the true NLL
    gradient obtained via autodiff through the forward algorithm. No
    E-step, no EM machinery.

    optimizer_kind : "adam" (default) or "sgd"
        Adam's per-parameter adaptive step size matters a lot here: many
        (m, a, y) cells in g_logits are visited rarely, so their gradient
        is small and noisy relative to common cells -- plain SGD uses one
        global lr for all of them, so common cells move fast while rare
        ones barely move (or momentum overshoots them). Adam rescales
        each logit's step by its own gradient history, which is usually
        why SGD looks stuck/noisy on a mostly-unvisited mc=1 controller
        while Adam still makes progress. `lr` default is lowered to 0.01
        accordingly -- Adam's effective step size is O(lr), not O(lr *
        raw_gradient) the way SGD's is, so the old lr=1.0 would be far
        too large for Adam.
    betas, adam_eps, weight_decay : passed straight to torch.optim.Adam
        when optimizer_kind="adam"; ignored for "sgd".
    momentum, nesterov : only used when optimizer_kind="sgd".

    init_rho, init_pi, init_g : optional arrays, shapes (M,), (A,M), (M,M,A,Y)
        If given (e.g. rho_mapso, pi_mapso, g_mapso, or a cross-mc embedded
        warm start via embed_fsc_params_across_mc), SGD starts from this
        point in probability space instead of a random logit draw. Since
        softmax is shift-invariant, log(p) reproduces p exactly on unpack,
        so this is a faithful warm start, not an approximation.
    init_jitter : float, default 0.0
        If >0 and init_rho/init_pi/init_g are given, adds
        N(0, init_jitter^2) noise to the warm-started logits (seeded by
        `seed`). Only matters for multi-restart warm starts -- with
        jitter=0, every restart would start at the identical point and
        just re-derive the same local optimum n_restarts times.

    dataset_nll_fn : callable(dataset, rho_np, pi_np, g_np) -> float
        Full-dataset NLL for epoch-level convergence tracking/logging (e.g.
        your existing numpy `dataset_nll`). Torch params are detached to
        numpy before calling it. If None, epoch NLL is estimated as the
        mean of per-batch training NLL over the epoch (cheaper, noisier).

    patience : int or None, default None
        Number of consecutive eval checks (every `eval_every` epochs)
        with no improvement of at least `tol` over the best-seen NLL
        before stopping. None disables patience -- training runs the full
        `n_epochs` (still subject to the old instant-stop `tol` check
        against the previous eval, kept for backward compatibility).
        Mini-batch NLL is noisy, so patience > 1 (e.g. 5-10) is usually
        what you want instead of stopping on the first flat/worse step.
    restore_best : bool, default True
        If True and patience triggered the stop (or just at the end of
        training), return the params from the best eval checkpoint rather
        than the last one -- guards against the last few epochs having
        drifted worse before patience ran out.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed if seed is not None else 0)

    if init_rho is not None:
        theta_np = theta_from_params(init_rho, init_pi, init_g)
        if init_jitter > 0:
            theta_np = theta_np + init_jitter * rng.standard_normal(theta_np.shape)
    else:
        theta_np = init_theta(M, A, Y, scale=0.3, seed=seed)

    theta = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)

    optimizer = make_optimizer(theta, optimizer_kind, lr, momentum=momentum, nesterov=nesterov,
                                betas=betas, eps=adam_eps, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, scheduler_kind, **(scheduler_kwargs or {}))

    nll_history = []
    time_history = []
    start_time = time.time()
    prev_nll = np.inf
    n = len(dataset)

    best_nll = np.inf
    best_theta = theta.detach().clone()
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        order = rng.permutation(n)
        epoch_batch_nlls = []
        for start in range(0, n, batch_size):
            batch_idx = order[start:start + batch_size]
            batch = [dataset[i] for i in batch_idx]

            optimizer.zero_grad()
            rho_logits, pi_logits, g_logits = unpack_theta_logits(theta, M, A, Y)
            log_rho = torch.log_softmax(rho_logits, dim=0)
            log_pi = torch.log_softmax(pi_logits, dim=0)
            log_g = torch.log_softmax(g_logits, dim=0)

            loss = batch_nll_torch(batch, log_rho, log_pi, log_g, reduction="mean")
            loss.backward()
            optimizer.step()
            epoch_batch_nlls.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        if (epoch % eval_every) == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                rho, pi, g = unpack_theta_probs(theta, M, A, Y)
                rho_np, pi_np, g_np = rho.numpy(), pi.numpy(), g.numpy()
            if dataset_nll_fn is not None:
                nll = dataset_nll_fn(dataset, rho_np, pi_np, g_np)
            else:
                nll = float(np.mean(epoch_batch_nlls))
            nll_history.append(nll)
            time_history.append(time.time() - start_time)
            if verbose:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(f"SGD epoch {epoch:03d} | NLL = {nll:.6f} | lr = {cur_lr:.4g}")

            if nll < best_nll - tol:
                best_nll = nll
                best_theta = theta.detach().clone()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if patience is not None and epochs_no_improve >= patience:
                if verbose:
                    print(f"SGD early-stopped (no improvement in {patience} eval checks).")
                break
            if abs(prev_nll - nll) < tol:
                if verbose:
                    print("SGD converged.")
                break
            prev_nll = nll

    final_theta = best_theta if restore_best else theta.detach()
    with torch.no_grad():
        rho, pi, g = unpack_theta_probs(final_theta, M, A, Y)
        rho_np, pi_np, g_np = rho.numpy(), pi.numpy(), g.numpy()
    return rho_np, pi_np, g_np, nll_history, time_history


# ---------------------------------------------------------------------------
# Restart drivers (sequential and process-parallel via joblib/loky), with
# warm-start support ported over: warm_params=(rho_warm, pi_warm, g_warm)
# already embedded to this problem's M (see embed_fsc_params_across_mc).
# If given, restarts in `warm_restarts` (or ALL restarts if None) start
# from warm_params via init_rho/init_pi/init_g + init_jitter -- the first
# such restart exactly (jitter=0), the rest jittered by warm_sigma. Other
# restarts are unaffected (normal random init).
# ---------------------------------------------------------------------------

def train_sgd_restarts(dataset, M, A=3, Y=2, n_restarts=5, seed=0,
                        warm_params=None, warm_restarts=None, warm_sigma=0.1,
                        **sgd_kwargs):
    """Sequential version, for debugging a single restart without
    process-pool complications.

    warm_params: optional (rho_warm, pi_warm, g_warm), already embedded to
        this problem's M (see embed_fsc_params_across_mc). If given,
        restarts in `warm_restarts` (or ALL restarts if None) start from
        warm_params, the first such restart exactly (jitter=0), the rest
        jittered by warm_sigma.
    """
    best = None
    first_warm_seen = False
    for r in range(n_restarts):
        seed_r = seed + r
        is_warm = warm_params is not None and ((warm_restarts is None) or (r in warm_restarts))
        if is_warm:
            rho_warm, pi_warm, g_warm = warm_params
            jitter = 0.0 if not first_warm_seen else warm_sigma
            first_warm_seen = True
            rho, pi, g, nll_history, time_history = train_sgd(
                dataset, M, A, Y, seed=seed_r, verbose=False,
                init_rho=rho_warm, init_pi=pi_warm, init_g=g_warm, init_jitter=jitter,
                **sgd_kwargs)
        else:
            rho, pi, g, nll_history, time_history = train_sgd(
                dataset, M, A, Y, seed=seed_r, verbose=False, **sgd_kwargs)
        final_nll = nll_history[-1]
        if best is None or final_nll < best[0]:
            best = (final_nll, rho, pi, g, nll_history, time_history, seed_r)
    final_nll, rho, pi, g, nll_history, time_history, best_seed = best
    return rho, pi, g, nll_history, time_history, best_seed


def _run_one_restart(seed_r, dataset, M, A, Y, sgd_kwargs,
                      init_rho=None, init_pi=None, init_g=None, init_jitter=0.0, is_warm=False):
    rho, pi, g, nll_history, time_history = train_sgd(
        dataset, M, A, Y, seed=seed_r, verbose=False,
        init_rho=init_rho, init_pi=init_pi, init_g=init_g, init_jitter=init_jitter,
        **sgd_kwargs)
    return nll_history[-1], rho, pi, g, nll_history, time_history, seed_r, is_warm


def train_sgd_restarts_parallel(dataset, M, A=3, Y=2, n_restarts=5, seed=0,
                                 n_jobs=-1,
                                 warm_params=None, warm_restarts=None, warm_sigma=0.1,
                                 **sgd_kwargs):
    """Process-parallel restarts via loky. Each worker is a fresh process,
    so torch.set_num_threads(1) and the OMP/MKL thread caps set at module
    import time apply cleanly in every worker -- same oversubscription
    protection as before.

    warm_params: optional (rho_warm, pi_warm, g_warm), already embedded to
        this problem's M (e.g. via embed_fsc_params_across_mc from a
        smaller-mc converged solution). If given, restarts in
        `warm_restarts` (or ALL restarts if warm_restarts=None) start from
        warm_params -- the first such restart exactly (jitter=0), the rest
        jittered by warm_sigma via init_jitter. Other restarts are unaffected
        (normal random init).

    Returns
    -------
    best_rho, best_pi, best_g, best_nll_history, best_time_history, best_seed,
    best_restart, results, all_nll, all_nll_histories
        all_nll           : shape (n_restarts,) -- final NLL per restart
        all_nll_histories : list of length n_restarts (histories may differ
                             in length if a restart early-stops via `tol`/
                             `patience`)
    """
    # --- build per-restart init args up front (jittering needs to happen
    #     deterministically before dispatch to worker processes) ---
    tasks = []
    first_warm_seen = False
    for r in range(n_restarts):
        seed_r = seed + r
        is_warm = warm_params is not None and ((warm_restarts is None) or (r in warm_restarts))
        if is_warm:
            rho_warm, pi_warm, g_warm = warm_params
            jitter = 0.0 if not first_warm_seen else warm_sigma
            first_warm_seen = True
            tasks.append((seed_r, rho_warm, pi_warm, g_warm, jitter, True))
        else:
            tasks.append((seed_r, None, None, None, 0.0, False))

    raw_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_one_restart)(seed_r, dataset, M, A, Y, sgd_kwargs,
                                   init_rho=ir, init_pi=ip, init_g=ig,
                                   init_jitter=ij, is_warm=iw)
        for seed_r, ir, ip, ig, ij, iw in tasks
    )

    all_nll = np.zeros(n_restarts)
    all_nll_histories = []
    results = []

    best_nll = np.inf
    best_restart = -1
    best = None

    for i, (final_nll, rho, pi, g, nll_history, time_history, seed_r, is_warm) in enumerate(raw_results):
        results.append({
            'restart': i,
            'seed': seed_r,
            'warm_started': is_warm,
            'rho': rho, 'pi': pi, 'g': g,
            'nll_value': final_nll,
            'nll_history': nll_history,
            'time_history': time_history,
        })
        all_nll[i] = final_nll
        all_nll_histories.append(nll_history)

        tag = ' [warm]' if is_warm else ''
        print(f'[restart {i}] seed={seed_r}  final NLL={final_nll:.6f}{tag}')

        if final_nll < best_nll:
            best_nll = final_nll
            best_restart = i
            best = (rho, pi, g, nll_history, time_history, seed_r)

    print(f'\nBest restart: {best_restart}  (NLL={best_nll:.6f})')

    best_rho, best_pi, best_g, best_nll_history, best_time_history, best_seed = best

    return (best_rho, best_pi, best_g, best_nll_history, best_time_history, best_seed,
            best_restart, results, all_nll, all_nll_histories)