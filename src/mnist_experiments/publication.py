"""Frozen publication protocol. Exploratory defaults and PCA stay independent."""
from copy import deepcopy
from .common import fingerprint, merge_dict
from .config import DEFAULT

PROTOCOL = 'mnist_rms_publication_v1'
METHODS = ['ddim', 'ddim_cluster', 'original_v3f2', 'rms_v3f2',
           'rms_v3f2_merge', 'matched_ddim', 'matched_cluster']
ABLATIONS = ['rms_v3f2_mean', 'rms_v3f2_every_level']
STABILITY_METHODS = ['ddim', 'rms_v3f2', 'rms_v3f2_merge', 'matched_ddim']
MATCH_TARGETS = {'matched_ddim': 'rms_v3f2', 'matched_cluster': 'rms_v3f2_merge'}


def definitions():
    search = merge_dict(DEFAULT['search'], {
        'schedule': {'kind': 'explicit', 'timesteps': [800, 500, 300, 200, 100, 50, 20, 5, 0]},
        'step_mode': 'rms', 'alpha': .03, 'steps_per_level': 5,
        'final_steps': 10, 'final_t': 0, 'final_step_scale': .1,
        'projection': 'clamp', 'tolerance': 1e-6, 'save_trajectory': False,
        'merge': {'metric': 'rms_euclidean', 'schedule': 'final_only',
                  'representative': 'medoid', 'radius_schedule': 'constant', 'radius': .65},
    })
    result = {}
    for method in METHODS + ABLATIONS:
        c = deepcopy(search)
        if method in {'ddim', 'ddim_cluster', 'matched_ddim', 'matched_cluster'}:
            c['steps_per_level'] = c['final_steps'] = 0
        if method in {'ddim', 'matched_ddim', 'original_v3f2', 'rms_v3f2'}:
            c['merge']['schedule'] = 'none'
        if method == 'original_v3f2':
            c['step_mode'] = 'l2'
        if method == 'rms_v3f2_mean':
            c['merge']['representative'] = 'mean'
        if method == 'rms_v3f2_every_level':
            c['merge']['schedule'] = 'every_level'
        result[method] = c
    return result


def definition_hash():
    return fingerprint({'protocol': PROTOCOL, 'ddim_steps': 50, 'definitions': definitions()})


def configure(config):
    """Resolve once at execution too: edited YAML cannot silently tune the protocol."""
    c = merge_dict(DEFAULT, config)
    if c.get('protocol', PROTOCOL) != PROTOCOL:
        raise ValueError('Unknown publication protocol')
    if c['ddim_steps'] != 50 or c['initialization'] != 'ddim':
        raise ValueError('Publication requires 50-step DDIM initialization')
    expected = definitions()['rms_v3f2_merge']
    if 'search' in config and config['search'] != expected:
        raise ValueError('Publication search definition is frozen; use exploratory runner for tuning')
    methods = c['methods'] if 'methods' in config else METHODS
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS + ABLATIONS):
        raise ValueError('Invalid publication methods')
    for method in methods:
        if method in MATCH_TARGETS and MATCH_TARGETS[method] not in methods:
            raise ValueError(f'{method} needs its actual NFE target {MATCH_TARGETS[method]}')
    if c.get('publication_series') == 'publication_main' and methods != METHODS:
        raise ValueError('publication_main methods are frozen')
    if c['population_size'] < 1:
        raise ValueError('Empty population')
    c.update(protocol=PROTOCOL, definition_hash=definition_hash(), search=expected,
             methods=list(methods), kind='publication_search')
    return c


def matched_budget(target_nfe, ddim_steps=50):
    if target_nfe < ddim_steps or ddim_steps < 1:
        raise ValueError('Target cannot fund one DDIM sample')
    n = int(target_nfe) // ddim_steps
    return dict(target_v3f2_nfe=int(target_nfe), matched_ddim_sample_count=n,
                matched_ddim_nfe=n * ddim_steps,
                nfe_difference=n * ddim_steps - int(target_nfe))


def estimate_publication(c):
    c = configure(c)
    p = c['population_size']
    movement_upper = p * (50 + 9 * 5 + 10)
    matched = matched_budget(movement_upper)
    costs = {}
    diagnostic = {}
    stability = c['stability']
    per_candidate = (stability['steps'] * (1 + len(stability['scales']) * stability['directions'])
                     if stability['enabled'] else 0)
    for method in c['methods']:
        costs[method] = (p * 50 if method in {'ddim', 'ddim_cluster'} else
                         matched['matched_ddim_nfe'] if method in MATCH_TARGETS else movement_upper)
        n = matched['matched_ddim_sample_count'] if method in MATCH_TARGETS else p
        diagnostic[method] = n * per_candidate if method in STABILITY_METHODS else 0
        if c['curvature']['enabled']:
            diagnostic[method] += n * 2 * c['curvature']['directions']
    # Raw float32 noise, initial/candidate and decoded images + stability endpoints.
    raw = sum((matched['matched_ddim_sample_count'] if m in MATCH_TARGETS else p)
              * 784 * 4 * 5 for m in costs)
    endpoints = sum((matched['matched_ddim_sample_count'] if m in MATCH_TARGETS else p)
                    * 784 * 4 * len(stability['scales']) * stability['directions']
                    for m in costs if m in STABILITY_METHODS) if stability['enabled'] else 0
    return dict(model_trainings=0, nfe_upper_bound=sum(costs.values()),
                diagnostic_nfe_upper=sum(diagnostic.values()),
                total_nfe_upper=sum(costs.values()) + sum(diagnostic.values()),
                per_method_nfe_upper=costs, per_method_diagnostic_nfe_upper=diagnostic,
                matched_samples_upper=matched['matched_ddim_sample_count'],
                raw_tensor_bytes_estimate=raw + endpoints,
                note='All methods charged their own DDIM initialization. No-merge upper bound. '
                     'Storage excludes JSON, figures, reference cache and checkpoint copies.')
