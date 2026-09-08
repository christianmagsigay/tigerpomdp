import numpy as np
import time

def forward_scaled(actions, observations, rho, pi, g, eps=1e-300):
    M = len(rho)
    T = len(actions)
    alpha = np.zeros((T, M))
    c = np.zeros(T)

    alpha[0] = rho * pi[actions[0]]
    c[0] = max(alpha[0].sum(), eps)   # floor the normalizer, not alpha itself
    alpha[0] /= c[0]

    for t in range(T - 1):
        a = actions[t]
        y = observations[t]
        for mp in range(M):
            alpha[t+1, mp] = pi[actions[t+1], mp] * np.sum(alpha[t] * g[mp, :, a, y])
        c[t+1] = max(alpha[t+1].sum(), eps)
        alpha[t+1] /= c[t+1]

    loglik = np.sum(np.log(c))
    return alpha, c, loglik


def backward_scaled(actions, observations, pi, g, c):
    M = pi.shape[1]
    T = len(actions)
    beta = np.zeros((T, M))
    beta[-1] = 1.0

    for t in range(T-2, -1, -1):
        a = actions[t]
        y = observations[t]
        for m in range(M):
            beta[t, m] = np.sum(g[:, m, a, y] * pi[actions[t+1]] * beta[t+1])
        beta[t] /= c[t+1]
    return beta


def compute_gamma(alpha, beta):
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma


def compute_xi(actions, observations, alpha, beta, pi, g):
    T, M = alpha.shape
    xi = np.zeros((T-1, M, M))
    for t in range(T-1):
        a = actions[t]
        y = observations[t]
        denom = 0.
        for m in range(M):
            for mp in range(M):
                xi[t, m, mp] = alpha[t, m] * g[mp, m, a, y] * pi[actions[t+1], mp] * beta[t+1, mp]
                denom += xi[t, m, mp]
        xi[t] /= denom
    return xi


def trajectory_loglikelihood(actions, observations, rho, pi, g):
    _, _, ll = forward_scaled(actions, observations, rho, pi, g)
    return ll


def dataset_nll(dataset, rho, pi, g):
    total_ll = 0.
    for actions, obs in dataset:
        total_ll += trajectory_loglikelihood(actions, obs, rho, pi, g)
    nll = -total_ll / len(dataset)
    return max(nll, 0.0)

def forward_backward(actions, observations, rho, pi, g):
    alpha, c, loglik = forward_scaled(actions, observations, rho, pi, g)
    beta = backward_scaled(actions, observations, pi, g, c)
    gamma = compute_gamma(alpha, beta)
    xi = compute_xi(actions, observations, alpha, beta, pi, g)
    return alpha, beta, gamma, xi, loglik


def random_fsc_parameters(M, A, Y, seed=None):
    rng = np.random.default_rng(seed)
    rho = rng.dirichlet(np.ones(M))
    pi = rng.dirichlet(np.ones(A), size=M).T
    g = np.zeros((M, M, A, Y))
    for a in range(A):
        for y in range(Y):
            g[:, :, a, y] = rng.dirichlet(np.ones(M), size=M).T
    return rho, pi, g


# ===========================================================================
# Warm-start helpers: embed a converged (rho, pi, g) from mc_prev into the
# larger mc_new problem, and jitter it to seed a diverse swarm of restarts.
# Unlike MAPSO's logit-space embedding, these work directly on probabilities
# (EM's parameters are already valid distributions), so no log/softmax
# round-trip is needed -- just re-normalization after copying/jittering.
# ===========================================================================
def embed_fsc_params_across_mc(rho_prev, pi_prev, g_prev, mc_prev, mc_new,
                                A=3, Y=2, seed=None):
    '''
    Embed a converged (rho, pi, g) from a smaller mc_prev into the M_new =
    2*mc_new+1 dimension needed for mc_new. Shared memory nodes (|m| <=
    mc_prev) keep their exact learned probabilities; the newly-added
    boundary nodes (mc_prev < |m| <= mc_new) get fresh Dirichlet-random
    distributions, since there's no prior information for them.

    Assumes node labels are exactly range(-mc, mc+1), matching
    build_memory_fsc's numbering (same assumption as MAPSO's
    embed_theta_across_mc).

    Returns rho_new, pi_new, g_new -- valid probability arrays, ready to
    pass as init_rho/init_pi/init_g to train_em, or to jitter_fsc_params
    below for building a warm-started restart swarm.
    '''
    rng = np.random.default_rng(seed)

    M_prev = 2 * mc_prev + 1
    M_new = 2 * mc_new + 1
    if M_new < M_prev:
        raise ValueError("embed_fsc_params_across_mc only supports mc_new >= mc_prev.")

    ms_prev = list(range(-mc_prev, mc_prev + 1))
    ms_new = list(range(-mc_new, mc_new + 1))
    idx_prev = {m: i for i, m in enumerate(ms_prev)}
    idx_new = {m: i for i, m in enumerate(ms_new)}
    shared_ms = ms_prev  # all of mc_prev's nodes exist in mc_new

    # --- fresh, valid random distributions at the new size ---
    rho_new, pi_new, g_new = random_fsc_parameters(M_new, A, Y, seed=seed)

    # --- overwrite shared-node entries with the learned mc_prev values ---
    for m in shared_ms:
        i_prev, i_new = idx_prev[m], idx_new[m]
        rho_new[i_new] = rho_prev[i_prev]
        pi_new[:, i_new] = pi_prev[:, i_prev]

    for m in shared_ms:
        for mp in shared_ms:
            i_prev, ip_prev = idx_prev[m], idx_prev[mp]
            i_new, ip_new = idx_new[m], idx_new[mp]
            g_new[ip_new, i_new, :, :] = g_prev[ip_prev, i_prev, :, :]

    # --- renormalize: rho and g's new boundary-node columns were left as
    #     random Dirichlet draws (already normalized), but the copied
    #     shared-node rho entries need rho_new to sum to 1 overall again ---
    rho_new = rho_new / rho_new.sum()

    return rho_new, pi_new, g_new


def jitter_fsc_params(rho, pi, g, sigma=0.05, rng=None):
    '''
    Perturb a valid (rho, pi, g) while keeping every distribution valid
    (non-negative, normalized). Multiplies by exp(gaussian noise) then
    renormalizes -- equivalent to additive noise in log-space, which
    respects the simplex constraint (unlike naive additive noise, which
    can produce negative "probabilities").

    sigma=0 returns an exact copy (used for the one warm restart that
    should start exactly at the warm point, same role as particle[0] in
    MAPSO's make_warm_init_fn).
    '''
    if rng is None:
        rng = np.random.default_rng()

    if sigma == 0:
        return rho.copy(), pi.copy(), g.copy()

    M, A, Y = pi.shape[1], pi.shape[0], g.shape[3]

    rho_j = rho * np.exp(sigma * rng.standard_normal(rho.shape))
    rho_j /= rho_j.sum()

    pi_j = pi * np.exp(sigma * rng.standard_normal(pi.shape))
    pi_j /= pi_j.sum(axis=0, keepdims=True)

    g_j = g * np.exp(sigma * rng.standard_normal(g.shape))
    g_j /= g_j.sum(axis=0, keepdims=True)

    return rho_j, pi_j, g_j


# ===========================================================================
# 5.3 EM training (exactly as given)
# ===========================================================================
def train_em(dataset, M, A=3, Y=2, max_iter=200, tol=1e-10, smoothing=1e-12,
             verbose=True, seed=None,
             init_rho=None, init_pi=None, init_g=None):
    '''
    If init_rho/init_pi/init_g are all provided, EM starts from them
    (e.g. a MAPSO or cross-mc warm start) instead of a random init -- seed
    is then ignored for initialization purposes.
    '''
    if init_rho is not None and init_pi is not None and init_g is not None:
        rho, pi, g = init_rho.copy(), init_pi.copy(), init_g.copy()
    else:
        rho, pi, g = random_fsc_parameters(M, A, Y, seed=seed)

    history = []
    prev_ll = -np.inf

    em_nll_history = []
    em_time_history = []
    start_time = time.time()

    for iteration in range(max_iter):
        rho_count = np.zeros(M)
        pi_num = np.zeros((A, M))
        pi_den = np.zeros(M)
        g_num = np.zeros((M, M, A, Y))
        g_den = np.zeros((M, A, Y))
        total_ll = 0.0

        for actions, obs in dataset:
            alpha, beta, gamma, xi, ll = forward_backward(actions, obs, rho, pi, g)
            total_ll += ll
            T = len(actions)
            rho_count += gamma[0]

            for t in range(T):
                a = actions[t]
                pi_num[a] += gamma[t]
                pi_den += gamma[t]

            for t in range(T - 1):
                a = actions[t]
                y = obs[t]
                g_num[:, :, a, y] += xi[t].T
                g_den[:, a, y] += gamma[t]

        avg_ll = total_ll / len(dataset)
        history.append(avg_ll)

        rho = rho_count + smoothing
        rho /= rho.sum()

        for m in range(M):
            pi[:, m] = pi_num[:, m] + smoothing
            pi[:, m] /= pi[:, m].sum()

        for a in range(A):
            for y in range(Y):
                for m in range(M):
                    if g_den[m, a, y] < 1e-12:
                        g[:, m, a, y] = 1.0 / M
                    else:
                        g[:, m, a, y] = g_num[:, m, a, y] + smoothing
                        g[:, m, a, y] /= g[:, m, a, y].sum()

        current_nll = dataset_nll(dataset, rho, pi, g)
        em_nll_history.append(current_nll)
        em_time_history.append(time.time() - start_time)

        if verbose:
            print(f"EM Iter {iteration:03d} | LL = {avg_ll:.6f} | NLL = {current_nll:.6f}")

        if np.abs(avg_ll - prev_ll) < tol:
            if verbose:
                print("EM converged.")
            break
        prev_ll = avg_ll

    return rho, pi, g, history, em_nll_history, em_time_history


def train_em_with_restarts(dataset_encoded, M, A, Y,
                            max_iter=200, n_restarts=10, seed=None,
                            warm_params=None, warm_restarts=None, warm_sigma=0.05,
                            verbose=False):
    '''
    Run EM n_restarts times with independent, reproducible seeds, and return
    the best fit plus per-restart diagnostics for plotting.

    warm_params: optional (rho_warm, pi_warm, g_warm), already embedded to
        this problem's M (see embed_fsc_params_across_mc). If given:
          - restarts in `warm_restarts` (or ALL restarts if warm_restarts is
            None) initialize from warm_params, jittered by warm_sigma
            (except the FIRST such restart, which starts from an exact
            copy, sigma=0 -- mirrors MAPSO's make_warm_init_fn behavior).
          - all other restarts use the normal random init, unchanged.
    warm_restarts: set of restart indices (0-based) to warm-start. Ignored
        if warm_params is None.

    seed=None -> nondeterministic master seed (still internally consistent
                 per-call, just not reproducible across script runs).
    seed=int  -> fully reproducible: same seed always gives the same
                 restart_seeds, same inits, same trajectories, same result.

    Returns
    -------
    best_rho, best_pi, best_g, best_nll, best_restart, results, all_nll, all_nll_histories
        all_nll           : shape (n_restarts,) — final NLL for each restart
        all_nll_histories : list of length n_restarts (histories may differ in
                             length if EM stops early via convergence, so this
                             is a list rather than a fixed-size array)
    '''
    master_rng = np.random.default_rng(seed)
    restart_seeds = master_rng.integers(0, 2**31 - 1, size=n_restarts)

    best_nll = np.inf
    best_rho = best_pi = best_g = None
    best_restart = -1
    results = []

    all_nll = np.zeros(n_restarts)
    all_nll_histories = []

    first_warm_seen = False

    for i, rseed in enumerate(restart_seeds):
        rseed = int(rseed)

        is_warm = warm_params is not None and ((warm_restarts is None) or (i in warm_restarts))

        if is_warm:
            rho_warm, pi_warm, g_warm = warm_params
            sigma = 0.0 if not first_warm_seen else warm_sigma
            first_warm_seen = True
            rng = np.random.default_rng(rseed)
            init_rho, init_pi, init_g = jitter_fsc_params(rho_warm, pi_warm, g_warm, sigma=sigma, rng=rng)
            rho, pi, g, _, nll_hist, _ = train_em(
                dataset_encoded, M, A, Y,
                max_iter=max_iter, verbose=verbose,
                init_rho=init_rho, init_pi=init_pi, init_g=init_g,
            )
        else:
            rho, pi, g, _, nll_hist, _ = train_em(
                dataset_encoded, M, A, Y,
                max_iter=max_iter, verbose=verbose, seed=rseed,
            )

        final_value = nll_hist[-1]
        results.append({
            'restart': i,
            'seed': rseed,
            'warm_started': is_warm,
            'rho': rho, 'pi': pi, 'g': g,
            'nll_value': final_value,
            'nll_history': nll_hist,
        })

        all_nll[i] = final_value
        all_nll_histories.append(nll_hist)

        tag = ' [warm]' if is_warm else ''
        print(f'[restart {i}] seed={rseed}  final NLL={final_value:.6f}{tag}')

        if final_value < best_nll:
            best_nll = final_value
            best_rho, best_pi, best_g = rho, pi, g
            best_restart = i

    print(f'\nBest restart: {best_restart}  (NLL={best_nll:.6f})')

    return best_rho, best_pi, best_g, best_nll, best_restart, results, all_nll, all_nll_histories


import os

# Must be set before numpy/numba import bindings spin up their thread pools.
# Prevents worker oversubstription in joblib.Parallel (same fix as the
# MAPSO/SGD parallel restarts).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from joblib import Parallel, delayed


def _em_restart_worker(rseed, dataset_encoded, M, A, Y, max_iter, verbose,
                        init_rho=None, init_pi=None, init_g=None, is_warm=False):
    '''
    Single EM restart, run in a worker process. Returns the same per-restart
    dict shape used by train_em_with_restarts, so results/all_nll/
    all_nll_histories can be reassembled identically to the serial version.
    '''
    rho, pi, g, _, nll_hist, _ = train_em(
        dataset_encoded, M, A, Y,
        max_iter=max_iter, verbose=verbose, seed=rseed,
        init_rho=init_rho, init_pi=init_pi, init_g=init_g,
    )
    final_value = nll_hist[-1]
    return {
        'seed': rseed,
        'warm_started': is_warm,
        'rho': rho, 'pi': pi, 'g': g,
        'nll_value': final_value,
        'nll_history': nll_hist,
    }


def train_em_with_restarts_parallel(dataset_encoded, M, A, Y,
                                     max_iter=200, n_restarts=10, seed=None,
                                     warm_params=None, warm_restarts=None, warm_sigma=0.05,
                                     verbose=False, n_jobs=-1, backend='loky'):
    '''
    Parallel version of train_em_with_restarts, with the same warm-start
    contract: warm_params=(rho_warm, pi_warm, g_warm) already embedded to
    this problem's M (see embed_fsc_params_across_mc); warm_restarts picks
    which restart indices use it (None = all); the first warm restart gets
    an exact copy (sigma=0), the rest are jittered by warm_sigma.

    seed=None -> nondeterministic master seed (internally consistent per-call,
                 not reproducible across script runs).
    seed=int  -> fully reproducible: same seed always gives the same
                 restart_seeds, same inits, same trajectories, same result --
                 regardless of n_jobs, since each restart is seeded
                 independently and results are reassembled in restart order.

    Returns
    -------
    Same signature as train_em_with_restarts:
    best_rho, best_pi, best_g, best_nll, best_restart, results, all_nll, all_nll_histories
    '''
    master_rng = np.random.default_rng(seed)
    restart_seeds = [int(s) for s in master_rng.integers(0, 2**31 - 1, size=n_restarts)]

    # --- build per-restart init args up front (jittering needs to happen
    #     on the main process, deterministically, before dispatch) ---
    tasks = []
    first_warm_seen = False
    for i, rseed in enumerate(restart_seeds):
        is_warm = warm_params is not None and ((warm_restarts is None) or (i in warm_restarts))
        if is_warm:
            rho_warm, pi_warm, g_warm = warm_params
            sigma = 0.0 if not first_warm_seen else warm_sigma
            first_warm_seen = True
            rng = np.random.default_rng(rseed)
            init_rho, init_pi, init_g = jitter_fsc_params(rho_warm, pi_warm, g_warm, sigma=sigma, rng=rng)
            tasks.append((rseed, init_rho, init_pi, init_g, True))
        else:
            tasks.append((rseed, None, None, None, False))

    raw_results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_em_restart_worker)(rseed, dataset_encoded, M, A, Y, max_iter, verbose,
                                     init_rho=ir, init_pi=ip, init_g=ig, is_warm=iw)
        for rseed, ir, ip, ig, iw in tasks
    )

    best_nll = np.inf
    best_rho = best_pi = best_g = None
    best_restart = -1
    results = []
    all_nll = np.zeros(n_restarts)
    all_nll_histories = []

    for i, r in enumerate(raw_results):
        r = {'restart': i, **r}
        results.append(r)
        all_nll[i] = r['nll_value']
        all_nll_histories.append(r['nll_history'])

        tag = ' [warm]' if r['warm_started'] else ''
        print(f"[restart {i}] seed={r['seed']}  final NLL={r['nll_value']:.6f}{tag}")

        if r['nll_value'] < best_nll:
            best_nll = r['nll_value']
            best_rho, best_pi, best_g = r['rho'], r['pi'], r['g']
            best_restart = i

    print(f'\nBest restart: {best_restart}  (NLL={best_nll:.6f})')

    return best_rho, best_pi, best_g, best_nll, best_restart, results, all_nll, all_nll_histories