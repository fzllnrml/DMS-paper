"""Fixed controls, with the original sampler and step geometry."""
import math
import time
import numpy as np
import torch
from scipy.spatial.distance import cdist
from src.mnist_experiments.geometry import step_delta


def ddim_indices(total=1000, steps=105):
    if total != 1000 or steps not in (50, 105):
        raise ValueError('Frozen pilot requires T=1000 and 50/105 DDIM evaluations')
    indices = torch.linspace(total - 1, 0, steps).round().long()
    # Never silently collapse indices and charge evaluations that did not occur.
    if len(torch.unique_consecutive(indices)) != steps:
        raise ValueError('Duplicate DDIM indices: incompatible with frozen NFE budget')
    return indices.tolist()


def ddim_105(model, noise):
    ddim_indices(model.schedule.T, 105)
    return model.ddim(noise, 105)


@torch.no_grad()
def refine(model, initial, steps, alpha):
    x = initial.clone()
    for _ in range(steps):
        x = (x + step_delta(model.score(x, 0), alpha, 'rms')).clamp(-1, 1)
        if not torch.isfinite(x).all():
            raise FloatingPointError('Nonfinite refinement')
    return x


def low_noise_only(model, initial):
    return refine(model, refine(model, initial, 45, .03), 10, .003)


def select_representatives(pool, count=128):
    """Only generated pixels enter selection; no evaluator/reference arguments."""
    started = time.perf_counter()
    x = pool.detach().cpu().flatten(1).double().numpy()
    if not 1 <= count <= len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid generated pool or output count')
    d = cdist(x, x) / math.sqrt(x.shape[1])
    first = int(np.argmin(d.sum(axis=0)))
    selected = [first]
    best = d[:, first].copy()
    available = np.ones(len(x), dtype=bool)
    available[first] = False
    objectives = [float(best.sum())]
    for _ in range(count - 1):
        costs = np.minimum(best[:, None], d).sum(axis=0)
        costs[~available] = np.inf
        j = int(np.argmin(costs))  # numpy argmin chooses first original index.
        selected.append(j)
        available[j] = False
        best = np.minimum(best, d[:, j])
        objectives.append(float(best.sum()))
    indices = torch.tensor(selected, dtype=torch.long)
    return pool[indices.to(pool.device)].clone(), indices, dict(
        selector_neural_nfe=0, selection_wall_seconds=time.perf_counter()-started,
        distance_matrix_bytes=d.nbytes, working_array_bytes_upper=2*d.nbytes+x.nbytes+5*len(x)*8,
        pairwise_entries=len(x)**2, greedy_candidate_cost_entries=(count-1)*len(x)**2,
        objective_history=objectives, arithmetic='CPU float64 pixel RMS',
        memory_note='Explicit array estimate, not allocator/process peak memory')


@torch.no_grad()
def contraction_diagnostic(initial, mapping, seed, scales=(.02, .05), directions=2,
                           minimum_deviation=.01):
    """mapping is the complete ten-step map, shared by both trajectories."""
    if initial.ndim != 2 or not torch.isfinite(initial).all() or (initial.abs() > 1).any():
        raise ValueError('Expected finite feasible [N,D] candidates')
    n, d = initial.shape
    rng = torch.Generator().manual_seed(seed)
    v = torch.randn(n, directions, d, generator=rng)
    v = (v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)).to(initial.device)
    reference = mapping(initial.clone())
    drift = (reference-initial).square().mean(-1).sqrt()
    arrays = dict(initial=initial.cpu(), reference_endpoint=reference.cpu(), directions=v.cpu())
    rows, summaries = [], []
    for scale in scales:
        perturbed = (initial[:, None] + scale*math.sqrt(d)*v).clamp(-1, 1)
        delta0 = (perturbed-initial[:, None]).square().mean(-1).sqrt()
        endpoint = mapping(perturbed.reshape(-1, d)).reshape(n, directions, d)
        separation = (endpoint-reference[:, None]).square().mean(-1).sqrt()
        if not torch.isfinite(endpoint).all() or not torch.isfinite(reference).all():
            raise FloatingPointError('Nonfinite diagnostic map')
        valid = delta0 > minimum_deviation
        rho = separation / delta0.clamp_min(torch.finfo(delta0.dtype).tiny)
        values = rho[valid].cpu().double().numpy()
        summaries.append(dict(scale=scale, valid_fraction=float(valid.float().mean()),
            median_rho=float(np.median(values)) if len(values) else None,
            mean_rho=float(values.mean()) if len(values) else None,
            fraction_rho_lt_1=float((values < 1).mean()) if len(values) else None,
            fraction_rho_le_09=float((values <= .9).mean()) if len(values) else None,
            median_reference_drift=float(np.median(drift.cpu().numpy())),
            valid_count=int(valid.sum()), total_directions=n*directions))
        for i in range(n):
            for j in range(directions):
                rows.append(dict(candidate=i, direction=j, scale=scale, valid=bool(valid[i,j]),
                    delta0=float(delta0[i,j]), final_separation=float(separation[i,j]),
                    rho=float(rho[i,j]) if valid[i,j] else None, reference_drift=float(drift[i])))
        arrays[str(scale)] = dict(perturbed_initial=perturbed.cpu(), perturbed_endpoint=endpoint.cpu())
    return dict(summary=summaries, per_candidate=rows,
                interpretation='Finite-horizon contraction diagnostic of the specified projected map; not proof of a mode'), arrays


def stability_v2(model, initial, seed):
    return contraction_diagnostic(initial, lambda x: refine(model, x, 10, .003), seed+1001)
