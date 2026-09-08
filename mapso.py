import numpy as np
import numba as nb
import os
import warnings

warnings.filterwarnings('ignore', category=nb.NumbaWarning)

# ===========================================================================
# MAPSO — Modified Adaptive Particle Swarm Optimization
#
# The objective function passed in must be @nb.njit and return (float, bool),
# where the bool is a rejection flag (True = reject the proposed position).
# ===========================================================================

strat_names = ('S1 exploration', 'S2 exploitation', 'S3 convergence', 'S4 jumping out')


@nb.njit
def S1_exploration(f):
    if f <= 0.4 or f > 0.8:
        return 0.
    elif 0.4 < f <= 0.6:
        return 5 * f - 2
    elif 0.6 < f <= 0.7:
        return 1.
    else:
        return -10 * f + 8


@nb.njit
def S2_exploitation(f):
    if f <= 0.2 or f > 0.6:
        return 0.
    elif 0.2 < f <= 0.3:
        return 10 * f - 2
    elif 0.3 < f <= 0.4:
        return 1.
    else:
        return -5 * f + 3


@nb.njit
def S3_convergence(f):
    if f <= 0.1 or f > 0.3:
        return 1.
    else:
        return -5 * f + 1.5


@nb.njit
def S4_jumping_out(f):
    if f <= 0.7:
        return 0.
    elif 0.7 < f <= 0.9:
        return 5 * f - 3.5
    else:
        return 1.


@nb.njit(parallel=True)
def run_optimization(fun_to_min, fun_params,
                     trainable_mask,
                     n_dimensions, n_particles, n_iterations,
                     init_particles, init_velocities,
                     num_neighbors_init, num_neighbors_final, num_neighbors_mid,
                     c1=2.0, c2=2.0, w=0.9,
                     sigma_min=0.1, sigma_max=1.,
                     seed=-1,
                     tol=1e-10, patience=1,
                     verbose=True, verbose_epochs=True):
    '''Run the MAPSO optimizer. Pass seed >= 0 for reproducible runs.
    Stops early once |gbest_value - prev_gbest_value| < tol for `patience`
    consecutive iterations in a row (patience=1 -> stop on first such iter).'''

    if seed >= 0:
        np.random.seed(seed)

    particles = init_particles
    velocities = init_velocities
    pbest = particles.copy()

    gbest_array = np.zeros((n_iterations, n_dimensions), dtype=np.float64)
    gbest_values_array = np.zeros(n_iterations, dtype=np.float64)

    pbest_values = np.zeros(n_particles)
    for j in nb.prange(n_particles):
        pbest_values[j], _ = fun_to_min(particles[j], fun_params)

    idx_min = pbest_values.argmin()
    gbest = pbest[idx_min]
    gbest_value = pbest_values[idx_min]
    prev_gbest_value = gbest_value  # for tol check
    stall_count = 0                 # consecutive iters below tol

    lbest = pbest.copy()
    lbest_values = pbest_values.copy()

    if verbose_epochs:
        print('Initial best value:', gbest_value)

    gbest_array[0] = gbest
    gbest_values_array[0] = gbest_value

    idxs_trainable = np.where(trainable_mask)[0]
    current_strategy = 0

    masks_infs = np.zeros((n_particles, n_dimensions), dtype=np.bool_)
    for i in nb.prange(n_particles):
        masks_infs[i] = ~np.isinf(particles[i])

    actual_iters = n_iterations  # updated if we break early

    for idx_iter in range(n_iterations):

        if idx_iter < n_iterations // 2:
            num_neighbors = int(num_neighbors_init + (num_neighbors_mid - num_neighbors_init) * (idx_iter / n_iterations) ** 4)
        else:
            num_neighbors = int(num_neighbors_mid + (num_neighbors_final - num_neighbors_mid) * ((idx_iter - n_iterations // 2) / (n_iterations // 2)) ** 4)

        if verbose:
            print(f'\t Number of spatial neighbors: {num_neighbors}')

        distances = np.zeros((n_particles, n_particles), dtype=np.float64)
        for i in nb.prange(n_particles):
            for j in range(i + 1, n_particles):
                mask = masks_infs[i] & masks_infs[j]
                diff = particles[i] - particles[j]
                distances[i, j] = np.sqrt(np.sum(diff[mask] ** 2))
                distances[j, i] = distances[i, j]

        for i in nb.prange(n_particles):
            neighbors = np.argsort(distances[i])[:num_neighbors]
            best_neighbor_idx = neighbors[np.argmin(pbest_values[neighbors])]

            if pbest_values[best_neighbor_idx] < lbest_values[i]:
                lbest[i] = pbest[best_neighbor_idx]
                lbest_values[i] = pbest_values[best_neighbor_idx]

            if current_strategy == 2:
                sigma = sigma_max - (sigma_max - sigma_min) * idx_iter / n_iterations
                random_dim = np.random.randint(0, len(idxs_trainable))
                mutated_lbest = lbest[i].copy()
                mutated_lbest[idxs_trainable[random_dim]] += np.random.randn() * sigma

                mutated_fitness_value, flag_trj = fun_to_min(mutated_lbest, fun_params)
                current_lbest_fitness, _ = fun_to_min(lbest[i], fun_params)

                if not flag_trj and mutated_fitness_value < current_lbest_fitness:
                    lbest[i] = mutated_lbest
                    lbest_values[i] = mutated_fitness_value

        for j in nb.prange(n_particles):
            new_velocity = velocities[j].copy()
            new_position = particles[j].copy()

            for idx_dim in idxs_trainable:
                new_velocity[idx_dim] *= w
                new_velocity[idx_dim] += c1 * np.random.rand() * (pbest[j, idx_dim] - particles[j, idx_dim])
                new_velocity[idx_dim] += c2 * np.random.rand() * (lbest[j, idx_dim] - particles[j, idx_dim])
                new_position[idx_dim] += new_velocity[idx_dim]

            fitness_value, flag_trj = fun_to_min(new_position, fun_params)

            if not flag_trj:
                velocities[j] = new_velocity.copy()
                particles[j] = new_position.copy()

                if fitness_value < pbest_values[j]:
                    pbest_values[j] = fitness_value
                    pbest[j] = particles[j]
            else:
                velocities[j] *= w

        if current_strategy == 2:
            sigma = sigma_max - (sigma_max - sigma_min) * idx_iter / n_iterations

            for j in nb.prange(n_particles):
                random_dim = np.random.randint(0, len(idxs_trainable))
                mutated_lbest = lbest[j].copy()
                mutated_lbest[idxs_trainable[random_dim]] += np.random.randn() * sigma
                mutated_fitness_value, flag_trj = fun_to_min(mutated_lbest, fun_params)

                if not flag_trj and mutated_fitness_value < lbest_values[j]:
                    if verbose:
                        print('\t Mutated lbest: ', lbest_values[j], '-> ', mutated_fitness_value)
                    lbest[j] = mutated_lbest
                    lbest_values[j] = mutated_fitness_value

        idx_best = lbest_values.argmin()
        if lbest_values[idx_best] < gbest_value:
            gbest = lbest[idx_best]
            gbest_value = lbest_values[idx_best]

        gbest_array[idx_iter] = gbest
        gbest_values_array[idx_iter] = gbest_value

        D = distances.sum(axis=1) / (n_particles - 1)
        Dmin = D.min()
        Dmax = D.max()
        fval = (D[idx_best] - Dmin) / (Dmax - Dmin + 1e-10)

        strategies_vals = np.zeros(4, dtype=np.float64)
        strategies_vals[0] = S1_exploration(fval)
        strategies_vals[1] = S2_exploitation(fval)
        strategies_vals[2] = S3_convergence(fval)
        strategies_vals[3] = S4_jumping_out(fval)

        non_zero_indices = [i for i in range(4) if strategies_vals[i] != 0]
        count_zero = 4 - len(non_zero_indices)

        if count_zero == 3:
            for i in range(4):
                if strategies_vals[i] != 0:
                    current_strategy = i
                    break
        elif count_zero == 2:
            for i in range(1, 5):
                next_strategy = (current_strategy + i) % 4
                if strategies_vals[next_strategy] != 0:
                    current_strategy = next_strategy
                    break

        delta = 0.05 + (np.random.random() * 0.05)

        if current_strategy == 0:
            c1 += delta
            c2 -= delta
        elif current_strategy == 1:
            c1 += 0.5 * delta
            c2 -= 0.5 * delta
        elif current_strategy == 2:
            c1 += 0.5 * delta
            c2 += 0.5 * delta
        elif current_strategy == 3:
            c1 -= delta
            c2 += delta

        c1 = max(1.5, min(2.5, c1))
        c2 = max(1.5, min(2.5, c2))

        if c1 + c2 > 4.0:
            c1 = 4.0 * (c1 / (c1 + c2))
            c2 = 4.0 * (c2 / (c1 + c2))

        w = 1 / (1 + 1.5 * np.exp(-2.6 * fval))

        if verbose_epochs:
            print(f'Iteration {idx_iter+1}/{n_iterations}, best value:', gbest_value)
        if verbose:
            print('\t f value:', np.round(fval, 3), '- strategy:', strat_names[current_strategy],
                  '- c1:', np.round(c1, 3), '- c2:', np.round(c2, 3), 'w: ', np.round(w, 3))
            print('\t Average local best fitness:', np.round(np.mean(lbest_values), 3),
                  '+-', np.round(np.std(lbest_values), 3))

        # --- tol / patience check ---
        if np.abs(gbest_value - prev_gbest_value) < tol:
            stall_count += 1
        else:
            stall_count = 0
        prev_gbest_value = gbest_value

        if stall_count >= patience:
            actual_iters = idx_iter + 1
            break

    return gbest_array[:actual_iters], gbest_values_array[:actual_iters]


# ===========================================================================
# Restarts driver — plain Python, sits outside numba
# ===========================================================================

def init_fn(seed, n_particles, dimension, restart_idx=None):
    '''Draw initial particle positions/velocities, seeded for reproducibility.
    restart_idx is accepted for API compatibility with warm-start init_fns
    (see make_warm_init_fn) but unused here -- always fully random.'''
    np.random.seed(seed)
    init_particles = 0.3 * np.random.randn(n_particles, dimension)    # smaller init scale
    init_velocities = 0.02 * np.random.randn(n_particles, dimension)
    return init_particles, init_velocities


def make_warm_init_fn(warm_theta, base_init_fn=init_fn, warm_restarts=None,
                       warm_sigma=0.05, warm_velocity_scale=0.02):
    '''
    Returns an init_fn(seed, n_particles, dimension, restart_idx) that:
      - for restarts in `warm_restarts` (or ALL restarts if warm_restarts=None):
          seeds the swarm as warm_theta + small Gaussian jitter, so the swarm
          starts clustered around a known-good point instead of from scratch.
          One particle (index 0) is kept as an EXACT copy of warm_theta, so
          the warm start itself is always a valid candidate even if jittering
          only makes things worse initially.
      - for all other restarts: falls back to base_init_fn (fully random),
          preserving exploration diversity across the restart sweep.

    warm_theta must already be embedded into the FULL dimension of the target
    (larger-mc) problem -- see embed_theta_across_mc below.
    '''
    warm_theta = np.asarray(warm_theta)

    def _init(seed, n_particles, dimension, restart_idx=None):
        if warm_theta.shape[0] != dimension:
            raise ValueError(
                f"warm_theta has dimension {warm_theta.shape[0]}, but this "
                f"problem's dimension is {dimension}. You must embed the "
                f"smaller-mc theta into the full larger-mc dimension before "
                f"passing it to make_warm_init_fn (see embed_theta_across_mc)."
            )
        is_warm = (warm_restarts is None) or (restart_idx in warm_restarts)
        rng = np.random.default_rng(seed)
        if is_warm:
            particles = warm_theta[None, :] + warm_sigma * rng.standard_normal((n_particles, dimension))
            particles[0] = warm_theta   # keep one exact copy of the warm point
            velocities = warm_velocity_scale * rng.standard_normal((n_particles, dimension))
            return particles, velocities
        else:
            return base_init_fn(seed, n_particles, dimension, restart_idx=restart_idx)

    return _init


def embed_theta_across_mc(theta_prev, mc_prev, mc_new, A=3, Y=2,
                           new_node_scale=0.3, seed=None):
    '''
    Embed a converged theta (raw logits, in the unpack_theta layout) from a
    smaller mc_prev into the larger dimension needed for mc_new. Memory nodes
    m shared between the two (i.e. |m| <= mc_prev) keep their exact learned
    logits; the newly-added boundary nodes (mc_prev < |m| <= mc_new) get
    small random logits, since they have no prior information to warm-start
    from.

    Assumes ms_sorted for a given mc is exactly range(-mc, mc+1) -- matching
    build_memory_fsc's node numbering. If your FSC ever uses a different
    node-labeling scheme, the idx_prev/idx_new mapping below needs to match
    that instead.

    Returns theta_new, ready to pass to make_warm_init_fn.
    '''
    rng = np.random.default_rng(seed)

    M_prev = 2 * mc_prev + 1
    M_new = 2 * mc_new + 1
    if M_new < M_prev:
        raise ValueError("embed_theta_across_mc only supports mc_new >= mc_prev "
                          "(growing the node set, not shrinking it).")

    ms_prev = list(range(-mc_prev, mc_prev + 1))
    ms_new = list(range(-mc_new, mc_new + 1))
    idx_prev = {m: i for i, m in enumerate(ms_prev)}
    idx_new = {m: i for i, m in enumerate(ms_new)}
    shared_ms = [m for m in ms_prev if m in idx_new]   # == ms_prev, since mc_new >= mc_prev

    # --- unpack theta_prev into raw logit arrays (no softmax) ---
    idx = 0
    rho_logits_prev = theta_prev[idx:idx + M_prev]; idx += M_prev
    pi_logits_prev = theta_prev[idx:idx + A * M_prev].reshape(A, M_prev); idx += A * M_prev
    g_logits_prev = theta_prev[idx:].reshape(M_prev, M_prev, A, Y)

    # --- build new logit arrays, new nodes get small random init ---
    rho_logits_new = new_node_scale * rng.standard_normal(M_new)
    pi_logits_new = new_node_scale * rng.standard_normal((A, M_new))
    g_logits_new = new_node_scale * rng.standard_normal((M_new, M_new, A, Y))

    # --- copy shared-node logits across, translating indices m -> idx_new[m] ---
    for m in shared_ms:
        i_prev, i_new = idx_prev[m], idx_new[m]
        rho_logits_new[i_new] = rho_logits_prev[i_prev]
        pi_logits_new[:, i_new] = pi_logits_prev[:, i_prev]

    for m in shared_ms:
        for mp in shared_ms:
            i_prev, ip_prev = idx_prev[m], idx_prev[mp]
            i_new, ip_new = idx_new[m], idx_new[mp]
            g_logits_new[ip_new, i_new, :, :] = g_logits_prev[ip_prev, i_prev, :, :]

    theta_new = np.concatenate([rho_logits_new.ravel(),
                                 pi_logits_new.ravel(),
                                 g_logits_new.ravel()])
    return theta_new


def run_optimization_with_restarts(fun_to_min, fun_params, trainable_mask,
                                    n_dimensions, n_particles, n_iterations,
                                    num_neighbors_init, num_neighbors_final, num_neighbors_mid,
                                    n_restarts=5, seed=None,
                                    c1=2.0, c2=2.0, w=0.9,
                                    sigma_min=0.1, sigma_max=1.,
                                    tol=1e-10, patience=1,
                                    init_fn=init_fn,
                                    verbose=False, verbose_epochs=False):
    '''
    Run MAPSO n_restarts times, each with an independent, reproducible seed
    derived from `seed`, and return the best result plus per-restart diagnostics.

    Each restart stops early once |gbest_value - prev_gbest_value| < tol holds
    for `patience` consecutive iterations, so restarts may finish in fewer
    than n_iterations iterations.

    init_fn: called as init_fn(seed, n_particles, n_dimensions, restart_idx=i).
    Pass a make_warm_init_fn(...) result to warm-start some/all restarts from
    a known-good point (e.g. a converged smaller-mc solution, embedded via
    embed_theta_across_mc).

    seed=None -> nondeterministic master seed (still internally consistent
                 per-call, just not reproducible across script runs).
    seed=int  -> fully reproducible: same seed always gives the same
                 restart_seeds, same inits, same trajectories, same result.

    Returns
    -------
    best_gbest, best_gbest_value, best_restart, results, all_nll, all_nll_histories
        all_nll           : shape (n_restarts,) — final NLL for each restart
        all_nll_histories : list of length n_restarts — convergence curve per
                             restart (ragged, since restarts may converge and
                             stop at different iteration counts)
    '''
    master_rng = np.random.default_rng(seed)
    restart_seeds = master_rng.integers(0, 2**31 - 1, size=n_restarts)

    best_gbest_value = np.inf
    best_gbest = None
    best_restart = -1
    results = []

    all_nll = np.zeros(n_restarts)
    all_nll_histories = []   # ragged

    for i, rseed in enumerate(restart_seeds):
        rseed = int(rseed)

        init_particles, init_velocities = init_fn(rseed, n_particles, n_dimensions, restart_idx=i)

        gbest_array, gbest_values_array = run_optimization(
            fun_to_min, fun_params, trainable_mask,
            n_dimensions, n_particles, n_iterations,
            init_particles, init_velocities,
            num_neighbors_init, num_neighbors_final, num_neighbors_mid,
            c1=c1, c2=c2, w=w,
            sigma_min=sigma_min, sigma_max=sigma_max,
            seed=rseed,
            tol=tol, patience=patience,
            verbose=verbose, verbose_epochs=verbose_epochs,
        )

        final_value = gbest_values_array[-1]
        # results.append({
        #     'restart': i,
        #     'seed': rseed,
        #     'gbest': gbest_array[-1],
        #     'gbest_value': final_value,
        #     'gbest_history': gbest_values_array,
        #     'n_iters_run': len(gbest_values_array),
        # })
        results.append({
            'restart': i,
            'seed': rseed,
            'gbest': gbest_array[-1],
            'gbest_value': final_value,
            'gbest_history': gbest_values_array,
            'theta_history': gbest_array.copy(),
            'n_iters_run': len(gbest_values_array),
        })

        all_nll[i] = final_value
        all_nll_histories.append(gbest_values_array)

        print(f'[restart {i}] seed={rseed}  iters={len(gbest_values_array)}/{n_iterations}  '
              f'final value={final_value:.6f}')

        if final_value < best_gbest_value:
            best_gbest_value = final_value
            best_gbest = gbest_array[-1]
            best_restart = i

    print(f'\nBest restart: {best_restart}  (value={best_gbest_value:.6f})')

    return best_gbest, best_gbest_value, best_restart, results, all_nll, all_nll_histories


def run_optimization_with_restarts_checkpointed(fun_to_min, fun_params, trainable_mask,
                                                  n_dimensions, n_particles, n_iterations,
                                                  num_neighbors_init, num_neighbors_final, num_neighbors_mid,
                                                  checkpoint_dir, run_id,
                                                  n_restarts=5, seed=None,
                                                  checkpoint_every=200,
                                                  c1=2.0, c2=2.0, w=0.9,
                                                  sigma_min=0.1, sigma_max=1.,
                                                  tol=1e-10, patience=1,
                                                  init_fn=init_fn,
                                                  verbose=False, verbose_epochs=False):
    '''
    Same contract as run_optimization_with_restarts, but:
      - each restart is run in chunks of `checkpoint_every` iterations
      - state is saved to disk after every chunk
      - if the kernel dies, re-running this call with the same run_id
        resumes from the last completed chunk of the last incomplete
        restart, instead of starting the whole sweep over

    init_fn: called as init_fn(seed, n_particles, n_dimensions, restart_idx=i)
    at the START of each restart only (not on resume from a mid-restart
    checkpoint, since particle state is already checkpointed by then). Pass
    a make_warm_init_fn(...) result to warm-start some/all restarts.

    tol / patience : each restart stops early once |gbest_value -
        prev_gbest_value| < tol holds for `patience` consecutive iterations
        in a row. This is checked both inside run_optimization (within a
        chunk) and at chunk boundaries here (across chunks), and the
        stall counter itself is checkpointed so a crash/resume mid-restart
        doesn't reset patience progress.
    '''
    os.makedirs(checkpoint_dir, exist_ok=True)
    meta_path = os.path.join(checkpoint_dir, f'{run_id}_meta.npz')

    master_rng = np.random.default_rng(seed)
    restart_seeds = master_rng.integers(0, 2**31 - 1, size=n_restarts)

    # --- resume bookkeeping across restarts ---
    if os.path.exists(meta_path):
        meta = np.load(meta_path, allow_pickle=True)
        restart_seeds = meta['restart_seeds']  # use the ORIGINAL seeds, not fresh ones
        completed_restarts = int(meta['completed_restarts'])
        all_nll = meta['all_nll']
        all_nll_histories = list(meta['all_nll_histories'])  # ragged -> object array
        best_gbest_value = float(meta['best_gbest_value'])
        best_gbest = meta['best_gbest'] if meta['best_gbest'].ndim else None
        best_restart = int(meta['best_restart'])
        results = list(meta['results'])
        print(f'[{run_id}] Resuming sweep at restart {completed_restarts}/{n_restarts}')
    else:
        completed_restarts = 0
        all_nll = np.zeros(n_restarts)
        all_nll_histories = []   # ragged: list of arrays, one per restart
        best_gbest_value = np.inf
        best_gbest = None
        best_restart = -1
        results = []
        print(f'[{run_id}] Starting fresh sweep of {n_restarts} restarts')

    def save_meta():
        np.savez(
            meta_path,
            restart_seeds=restart_seeds,
            completed_restarts=completed_restarts,
            all_nll=all_nll,
            all_nll_histories=np.array(all_nll_histories, dtype=object),
            best_gbest_value=best_gbest_value,
            best_gbest=best_gbest if best_gbest is not None else np.array(None),
            best_restart=best_restart,
            results=np.array(results, dtype=object),
        )

    for i in range(completed_restarts, n_restarts):
        rseed = int(restart_seeds[i])
        restart_ckpt_path = os.path.join(checkpoint_dir, f'{run_id}_restart{i}_ckpt.npz')

        # --- resume within-restart chunk progress, if any ---
        if os.path.exists(restart_ckpt_path):
            ckpt = np.load(restart_ckpt_path)
            particles = ckpt['particles']
            velocities = ckpt['velocities']
            done_iters = int(ckpt['done_iters'])
            gbest_values_history = list(ckpt['gbest_values_history'])
            gbest_history = list(ckpt['gbest_history'])
            stall_count = int(ckpt['stall_count'])
            print(f'[{run_id}] Restart {i}: resuming at iteration {done_iters}/{n_iterations} '
                  f'(stall_count={stall_count}/{patience})')
        else:
            particles, velocities = init_fn(rseed, n_particles, n_dimensions, restart_idx=i)
            done_iters = 0
            gbest_values_history = []
            gbest_history = []
            stall_count = 0
            print(f'[{run_id}] Restart {i}: starting fresh (seed={rseed})')

        converged = False

        while done_iters < n_iterations:
            chunk_size = min(checkpoint_every, n_iterations - done_iters)

            gbest_array, gbest_values_array = run_optimization(
                fun_to_min, fun_params, trainable_mask,
                n_dimensions, n_particles, chunk_size,
                particles, velocities,
                num_neighbors_init, num_neighbors_final, num_neighbors_mid,
                c1=c1, c2=c2, w=w,
                sigma_min=sigma_min, sigma_max=sigma_max,
                seed=-1,  # don't reseed mid-restart, only at restart start via init_fn
                tol=tol, patience=patience,  # inner loop can exit this chunk early
                verbose=verbose, verbose_epochs=verbose_epochs,
            )

            gbest_values_history.extend(gbest_values_array.tolist())
            gbest_history.extend(gbest_array.tolist())
            ran_this_chunk = len(gbest_values_array)
            done_iters += ran_this_chunk

            if ran_this_chunk < chunk_size:
                # run_optimization broke early inside this chunk -> patience
                # was already exhausted there.
                stall_count = patience
            else:
                # chunk ran to completion: update the cross-chunk stall
                # counter using the boundary between this chunk's first
                # value and the previous chunk's last value.
                if len(gbest_values_history) >= 2:
                    step_delta = abs(gbest_values_history[-1] - gbest_values_history[-2])
                    if step_delta < tol:
                        stall_count += 1
                    else:
                        stall_count = 0

            np.savez(
                restart_ckpt_path,
                particles=particles,
                velocities=velocities,
                done_iters=done_iters,
                gbest_values_history=np.array(gbest_values_history),
                gbest_history=np.array(gbest_history),
                stall_count=stall_count,
            )

            if stall_count >= patience:
                converged = True
                break

        # --- restart finished (either ran out of iterations or converged) ---
        gbest_values_history = np.array(gbest_values_history)
        gbest_history = np.array(gbest_history)
        final_value = gbest_values_history[-1]

        # results.append({
        #     'restart': i,
        #     'seed': rseed,
        #     'gbest': gbest_history[-1],
        #     'gbest_value': final_value,
        #     'gbest_history': gbest_values_history,
        #     'n_iters_run': len(gbest_values_history),
        #     'converged': converged,
        # })


        results.append({
            'restart': i,
            'seed': rseed,
            'gbest': gbest_history[-1],
            'gbest_value': final_value,
            'gbest_history': gbest_values_history,
            'theta_history': gbest_history.copy(),
            'n_iters_run': len(gbest_values_history),
            'converged': converged,
        })

        all_nll[i] = final_value
        all_nll_histories.append(gbest_values_history)

        status = 'converged' if converged else 'max_iter'
        print(f'[{run_id}] Restart {i} complete ({status}, {len(gbest_values_history)}/{n_iterations} iters). '
              f'seed={rseed}  final value={final_value:.6f}')

        if final_value < best_gbest_value:
            best_gbest_value = final_value
            best_gbest = gbest_history[-1]
            best_restart = i

        completed_restarts = i + 1
        save_meta()

        os.remove(restart_ckpt_path)  # restart done, no longer needed

    print(f'\n[{run_id}] Sweep complete. Best restart: {best_restart} (value={best_gbest_value:.1e})')

    return best_gbest, best_gbest_value, best_restart, results, all_nll, all_nll_histories

import numpy as np
import numba as nb

# ===========================================================================
# Probabilistic finite-state-controller (FSC) model, fit by maximum
# likelihood (via MAPSO) to a dataset of (actions, observations) sequences.
#
#   rho[m]          initial distribution over M memory nodes
#   pi[a, m]        P(action = a | memory = m)          (policy)
#   g[m', m, a, y]  P(memory' = m' | memory = m, a, y)   (belief transition)
#
# trajectory_loglik computes, via a forward filter over the latent memory
# node, the log-likelihood of the *actions actually taken*, treating the
# observations as known/given drivers of the (latent) memory transition.
# ===========================================================================

@nb.njit
def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


@nb.njit
def unpack_theta(theta, M, A, Y):
    idx = 0
    rho_logits = theta[idx:idx+M]
    idx += M
    pi_logits = theta[idx:idx+A*M].reshape(A, M)
    idx += A*M
    g_logits = theta[idx:].reshape(M, M, A, Y)

    rho = softmax(rho_logits)

    pi = np.zeros((A, M))
    for m in range(M):
        pi[:, m] = softmax(pi_logits[:, m])

    g = np.zeros((M, M, A, Y))
    for a in range(A):
        for y in range(Y):
            for m in range(M):
                g[:, m, a, y] = softmax(g_logits[:, m, a, y])

    return rho, pi, g


@nb.njit
def trajectory_loglik(actions, observations, rho, pi, g):
    M = rho.shape[0]
    T = len(actions)

    alpha = rho * pi[actions[0]]
    c = np.sum(alpha)
    if c <= 1e-300:
        return -1e30
    alpha /= c
    loglik = np.log(c)

    for t in range(T - 1):
        new_alpha = np.zeros(M)
        for mp in range(M):
            s = 0.0
            for m in range(M):
                s += alpha[m] * g[mp, m, actions[t], observations[t]]
            new_alpha[mp] = pi[actions[t+1], mp] * s

        c = np.sum(new_alpha)
        if c <= 1e-300:
            return -1e30

        new_alpha /= c
        loglik += np.log(c)
        alpha = new_alpha

    return loglik


@nb.njit
def tiger_negative_loglik(theta, params):
    dataset, M, A, Y = params
    rho, pi, g = unpack_theta(theta, M, A, Y)

    nll = 0.0
    for k in range(len(dataset)):
        actions, observations = dataset[k]
        ll = trajectory_loglik(actions, observations, rho, pi, g)
        if ll < -1e20 or np.isnan(ll):
            return 1e30, True
        nll -= ll

    nll /= len(dataset)
    return nll, False


# ---------------------------------------------------------------------------
# Helpers (plain Python — these prepare inputs for the njit code above,
# they don't need to be jitted themselves).
#
# Encoding convention:
#   action a in {-1, 0, 1}  ->  index a + 1   (0=OpenLeft, 1=Listen, 2=OpenRight)
#   obs    z in {-1, 1}     ->  index (z + 1) // 2  (0=HearLeft, 1=HearRight)
# ---------------------------------------------------------------------------

def encode_dataset(dataset):
    """Convert generate_dataset()'s output (list of {'trajectory': [...]})
    into the (actions_idx, observations_idx) integer-array pairs that
    tiger_negative_loglik expects."""
    encoded = []
    for ep in dataset:
        actions = np.array([step['a'] + 1 for step in ep['trajectory']], dtype=np.int64)
        observations = np.array([(step['z'] + 1) // 2 for step in ep['trajectory']], dtype=np.int64)
        encoded.append((actions, observations))
    return encoded


def build_ground_truth_theta(fsc, m0=0, big=25.0):
    """Construct the theta that reproduces the exact memory-FSC agent:
    one-hot rho at the starting memory node, one-hot pi at each node's
    prescribed action, one-hot g at each listen-node's deterministic
    m -> m+-1 transition. Used only as a sanity-check upper bound on how
    low the NLL can go (it should be ~0, since this model class can
    represent the exact generative policy exactly)."""
    ms_sorted = sorted(fsc.keys())
    M = len(ms_sorted)
    idx_of = {m: i for i, m in enumerate(ms_sorted)}
    A, Y = 3, 2

    rho_logits = np.full(M, -big)
    rho_logits[idx_of[m0]] = big

    pi_logits = np.full((A, M), -big)
    for m in ms_sorted:
        a = fsc[m]['action']
        pi_logits[a + 1, idx_of[m]] = big

    g_logits = np.full((M, M, A, Y), -big)
    for m in ms_sorted:
        node = fsc[m]
        if node['action'] == 0:                      # listen: real m -> m+-1 transition
            for z in (-1, 1):
                mp = node['g'][z]
                y_idx = (z + 1) // 2
                g_logits[idx_of[mp], idx_of[m], 1, y_idx] = big
        else:                                         # terminal node: never used, self-loop
            for y_idx in range(Y):
                g_logits[idx_of[m], idx_of[m], node['action'] + 1, y_idx] = big

    theta = np.concatenate([rho_logits.ravel(), pi_logits.ravel(), g_logits.ravel()])
    return theta, M, A, Y

def build_trainable_mask(M, A, Y, listen_idx=1):
    """g[:, :, a, :] for a != listen_idx is never read by trajectory_loglik
    (the terminal open-door action is always last in a trajectory, so it
    never drives a transition). Mask those entries out so MAPSO doesn't
    waste search budget on parameters the likelihood can't see."""
    dim_rho, dim_pi, dim_g = M, A * M, M * M * A * Y
    total = dim_rho + dim_pi + dim_g
    mask = np.ones(total, dtype=bool)
    idx = dim_rho + dim_pi
    for mp in range(M):
        for m in range(M):
            for a in range(A):
                for y in range(Y):
                    if a != listen_idx:
                        mask[idx] = False
                    idx += 1
    return mask