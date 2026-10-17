"""Exact-score problems; complete references are computed only after the search.

Product modes are Cartesian products of certified 1D maxima, mapped by an
invertible A. Collinear common-covariance GMMs reduce to the same 1D problem.
Interval exclusion/monotonicity exhausts the interval between extreme means;
the density derivative has a known strict sign outside it. An unresolved box,
degenerate critical point or failed residual check makes completeness FALSE.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import ndtr

from research_experiments.common.distributions import GMM, Schedule, ExactField, generate, reference_modes
from .storage import fingerprint


class ExactSchedule(Schedule):
    def __init__(self, clean_zero=True, **kwargs):
        super().__init__(**kwargs)
        self.clean_zero = clean_zero

    def alpha(self, t):
        value = super().alpha(t)
        return 1.0 if t == 0 and self.clean_zero else value


def certify_scalar(means, variance, weights, max_boxes=100000, min_width=1e-11, density_dimension=1,
                   restriction_centers=None):
    """Outward-rounded certificate for a 1D Gaussian density restriction.

    Certification concerns the supplied floating-point parameters, interpreted
    as exact numbers. Double-precision roots are localization estimates within
    certified isolating brackets; they are not themselves exact algebraic roots.
    This is not a generic multivariate GMM completeness claim.
    """
    from mpmath import iv
    iv.dps = 40
    means, weights = np.asarray(means, float), np.asarray(weights, float)
    variances = np.broadcast_to(np.asarray(variance, float), means.shape)
    if (np.any(variances <= 0) or not np.isfinite(variances).all() or len(means) != len(weights) or np.any(weights <= 0)
            or not np.isfinite(means).all() or not np.isfinite(weights).all()):
        raise ValueError("Invalid scalar mixture")
    if np.ptp(means) == 0:
        return dict(complete=True, roots=[float(means[0])], maxima=[float(means[0])],
                    minima=[], brackets=[[float(means[0]), float(means[0])]],
                    unresolved=[], boxes=0, proof="single distinct center, strictly decreasing concentric Gaussian mixture")
    vm = [iv.mpf(float(m)) for m in means]
    vv = [iv.mpf(float(v)) for v in variances]
    vw = [iv.mpf(float(w)) for w in weights]
    distance2 = iv.mpf(1)
    numeric_distance2 = 1.
    if restriction_centers is not None:
        first, second = restriction_centers
        # Work on u in [0,1] along the original center segment. Computing the
        # squared distance inside IV avoids treating a rounded Euclidean norm
        # as an exact geometric parameter in a completeness proof.
        distance2 = sum((iv.mpf(float(b)) - iv.mpf(float(a))) ** 2 for a, b in zip(first, second))
        delta = np.asarray(second) - np.asarray(first)
        numeric_distance2 = float(delta @ delta)

    def intervals(lo, hi):
        x = iv.mpf([float(lo), float(hi)])
        f, df = iv.mpf(0), iv.mpf(0)
        for mu, w, vs in zip(vm, vw, vv):
            diff = mu - x
            e = w / (vs ** (iv.mpf(density_dimension) / 2)) * iv.exp(-distance2 * (diff ** 2) / (2 * vs))
            f += diff / vs * e
            df += (distance2 * diff ** 2 / (vs ** 2) - 1 / vs) * e
        return f, df

    def sign(v):
        return 1 if v.a > 0 else -1 if v.b < 0 else 0

    log_amplitudes = np.log(weights) - density_dimension / 2 * np.log(variances)
    amplitudes = np.exp(log_amplitudes - log_amplitudes.max())

    def derivative(x):
        diff = means - x
        return float(np.sum(amplitudes * np.exp(-numeric_distance2 * diff ** 2 / (2 * variances)) * diff / variances))

    pending = [(float(means.min()), float(means.max()))]
    brackets, unresolved, examined = [], [], 0
    while pending and examined < max_boxes:
        lo, hi = pending.pop()
        examined += 1
        f, df = intervals(lo, hi)
        if sign(f):
            continue
        if sign(df):
            left, right = sign(intervals(lo, lo)[0]), sign(intervals(hi, hi)[0])
            if left and right:
                if left != right:
                    brackets.append((lo, hi))
                continue
        if hi - lo <= min_width:
            unresolved.append((lo, hi))
            continue
        # Avoid putting symmetric roots exactly on every bisection boundary.
        mid = lo + .473 * (hi - lo)
        pending.extend([(lo, mid), (mid, hi)])
    unresolved.extend(pending)
    roots = sorted(brentq(derivative, lo, hi, xtol=1e-14) for lo, hi in brackets)
    log_effective = np.log(weights) - (density_dimension - 1) / 2 * np.log(variances)
    g = GMM(np.array(means)[:, None], variances[:, None, None] / numeric_distance2, np.exp(log_effective - log_effective.max()))
    maxima, minima, degenerate = [], [], []
    for root in roots:
        curvature = float(g.hessian([[root]])[0, 0, 0])
        if abs(curvature) <= 1e-8 or abs(float(g.score([[root]])[0, 0])) > 1e-7:
            degenerate.append(root)
        elif curvature < 0:
            maxima.append(root)
        else:
            minima.append(root)
    return dict(complete=not unresolved and not degenerate and bool(maxima), roots=roots,
                maxima=maxima, minima=minima, brackets=sorted(brackets),
                unresolved=unresolved, degenerate=degenerate, boxes=examined,
                proof="outward interval derivative exclusion or strict monotonicity; analytic tail signs",
                restriction_distance_squared_interval=[str(distance2.a), str(distance2.b)],
                interval_precision_decimal=40, min_width=min_width, max_boxes=max_boxes)


@dataclass
class Problem:
    spec: dict
    seed: int
    A: np.ndarray
    factors: list
    covariance_diagonal: np.ndarray
    expanded: GMM
    orthogonal_axes: bool

    @property
    def dim(self):
        return self.A.shape[0]

    @property
    def identity(self):
        return fingerprint(dict(spec=self.spec, seed=self.seed, A=self.A,
                                factors=self.factors, covariance=self.covariance_diagonal))

    def field(self, schedule=None):
        return ProductField(self, schedule or ExactSchedule())

    def reference(self):
        certificates = [certify_scalar(f["means"], f["variance"], f["weights"]) for f in self.factors]
        modes_latent = np.asarray([list(v) + [0.] * (self.dim - len(self.factors))
                                   for v in product(*(c["maxima"] for c in certificates))], float).reshape(-1, self.dim)
        modes = modes_latent @ self.A.T
        residual = np.linalg.norm(self.expanded.score(modes), axis=1) if len(modes) else np.array([])
        eig = np.linalg.eigvalsh(self.expanded.hessian(modes)).max(1) if len(modes) else np.array([])
        cell_masses = []
        for f, c in zip(self.factors, certificates):
            bounds = [-np.inf] + c["minima"] + [np.inf]
            axis_masses = [sum(w * (ndtr((hi - m) / math.sqrt(f["variance"])) -
                                  ndtr((lo - m) / math.sqrt(f["variance"])))
                               for w, m in zip(f["weights"], f["means"]))
                           for lo, hi in zip(bounds, bounds[1:])]
            cell_masses.append(axis_masses)
        masses = np.array([np.prod(v) for v in product(*cell_masses)])
        count_consistent = all(len(c["maxima"]) == len(c["minima"]) + 1 for c in certificates)
        complete = (all(c["complete"] for c in certificates) and count_consistent and len(modes) > 0
                    and bool(np.all(residual < 1e-6)) and bool(np.all(eig < -1e-8)))
        rare = masses < .03 if len(masses) == len(modes) else np.zeros(len(modes), bool)
        info = dict(complete=complete, status="complete_product_certificate" if complete else "incomplete_certificate",
                    proof="invertible change of variables preserves all critical points and Hessian inertia",
                    certificates=certificates, modes_are_component_centers=False,
                    score_residuals=residual, max_hessian_eigenvalues=eig,
                    modal_cell_masses=masses, rare_mask=rare,
                    rare_definition="latent modal-cell probability < 0.03",
                    mass_is_euclidean_basin_probability=self.orthogonal_axes,
                    density_identity=self.identity)
        return modes, info

    def basin_labels(self, x, reference):
        """Exact clean-gradient basins only for orthogonal latent axes.

        Shear preserves critical points, but not Euclidean gradient trajectories.
        Unknown labels (-1) prevent a nearest-mode proxy becoming a false proof.
        """
        if not self.orthogonal_axes or not reference["complete"]:
            return np.full(len(x), -1, int)
        z = np.asarray(x) @ np.linalg.inv(self.A).T
        labels = np.zeros(len(z), int)
        for j, c in enumerate(reference["certificates"]):
            labels = labels * len(c["maxima"]) + np.searchsorted(c["minima"], z[:, j])
        return labels


class ProductField:
    kind = "analytic"

    def __init__(self, problem, schedule):
        self.problem, self.dim, self.schedule = problem, problem.dim, schedule
        self.scales = np.linalg.norm(problem.A, axis=0)
        self.Q = problem.A / self.scales[None]
        self.axes = [GMM(np.asarray(f["means"])[:, None] * self.scales[j],
                         np.full((len(f["means"]), 1, 1), f["variance"] * self.scales[j] ** 2), f["weights"])
                     for j, f in enumerate(problem.factors)]

    def score(self, x, t):
        alpha = self.schedule.alpha(t)
        if not self.problem.orthogonal_axes:
            return self.problem.expanded.score(x, alpha)
        y = np.asarray(x) @ self.Q
        s = -y / (alpha * self.scales ** 2 * self.problem.covariance_diagonal + 1 - alpha)
        for j, axis in enumerate(self.axes):
            s[:, j] = axis.score(y[:, j:j+1], alpha)[:, 0]
        return s @ self.Q.T


def make_problem(spec, seed):
    """Generate parameters without computing or exposing any reference modes."""
    family, d = spec["family"], spec["dim"]
    rng = np.random.default_rng(seed)
    if family == "legacy_alignment":
        return generate(dict(family="legacy_random", dim=d, components=spec.get("components", 2),
                             bounds=[-10., 10.]), seed)
    if family == "collinear_gmm":
        centers = np.cumsum([0., 2.6, 4., 6.])
        centers -= centers.mean()
        centers *= rng.uniform(.94, 1.06)
        factors = [dict(means=centers.tolist(), variance=1., weights=[.02, .18, .3, .5])]
    else:
        count = min(4 if family == "many" else 3, d)
        gap = {"close": 2.3, "overlap": 1.45, "rare": 3.4, "many": 3.4,
               "anisotropic": 3.4, "uneven": 3.4, "sheared": 3.4, "moderate": 3.4}[family]
        factors = []
        for j in range(count):
            separation = ([2.3, 3.8, 6.][j % 3] if family == "uneven" else
                          7.5 if family == "rare" and j == 0 else gap)
            if family == "overlap" and j > 0:
                separation = 3.4
            separation *= rng.uniform(.96, 1.04)
            w = rng.uniform(.44, .56) if family not in {"close", "overlap"} else .5
            if family == "rare" and j == 0:
                w = float(spec.get("rare_weight", .003))
            factors.append(dict(means=[-separation / 2, separation / 2], variance=1., weights=[w, 1 - w]))
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    q *= np.where(np.diag(r) >= 0, 1., -1.)[None]
    scales = np.geomspace(.4, 2.5, d) if family in {"anisotropic", "sheared"} else np.ones(d)
    A = q @ np.diag(scales)
    if family == "sheared":
        shear = np.eye(d)
        shear[0, 1] = 1.2
        shear[1, min(2, d - 1)] = .5
        A = A @ shear
    variance = np.ones(d)
    means, weights = [], []
    for indices in product(*(range(len(f["means"])) for f in factors)):
        z = np.zeros(d)
        weight = 1.
        for j, index in enumerate(indices):
            z[j] = factors[j]["means"][index]
            weight *= factors[j]["weights"][index]
        means.append(A @ z)
        weights.append(weight)
    covariance = A @ np.diag(variance) @ A.T
    mixture = GMM(means, np.repeat(covariance[None], len(means), axis=0), weights)
    return Problem(spec, seed, A, factors, variance, mixture, family != "sheared")


def numerical_basin_endpoints(mixture, x, modes, tolerance=1e-5):
    """Post hoc optimizer diagnostic, not a completeness/basin certificate."""
    endpoints, labels, converged = [], [], []
    for point in x:
        fit = minimize(lambda y: -float(mixture.logpdf(y)[0]), point,
                       jac=lambda y: -mixture.score(y)[0], method="BFGS",
                       options=dict(gtol=1e-7, maxiter=300))
        endpoint = fit.x
        ok = np.linalg.norm(mixture.score(endpoint)) < tolerance
        distances = np.linalg.norm(modes - endpoint, axis=1) if len(modes) else []
        label = int(np.argmin(distances)) if ok and len(modes) and min(distances) < .01 else -1
        endpoints.append(endpoint)
        labels.append(label)
        converged.append(bool(ok))
    return dict(endpoints=np.asarray(endpoints).reshape(-1, mixture.dim), labels=labels,
                converged=converged, certified=False)


def certify_isotropic_pair(g):
    """All critical points of two isotropic Gaussians lie on their center line.

    Unlike the v1 common-covariance shortcut this also supports different scalar
    variances. Transverse score is -z*sum(responsibility/variance), so z=0 at
    every stationary point and all transverse Hessian eigenvalues are negative.
    The 1D restriction includes the full d-dimensional density amplitudes.
    """
    if g.k != 2:
        return reference_modes(g)
    variances = g.covariances[:, 0, 0]
    if not np.array_equal(g.covariances, variances[:, None, None] * np.eye(g.dim)):
        return reference_modes(g)
    delta = g.means[1] - g.means[0]
    length = np.linalg.norm(delta)
    if length == 0:
        modes = g.means[:1].copy()
        certificate = dict(complete=True, maxima=[0.], proof="concentric isotropic components, strictly decreasing radial density")
    else:
        certificate = certify_scalar([0., 1.], variances, g.weights, density_dimension=g.dim,
                                     restriction_centers=g.means)
        modes = g.means[0] + np.asarray(certificate["maxima"])[:, None] * delta
    residual = np.linalg.norm(g.score(modes), axis=1)
    eig = np.linalg.eigvalsh(g.hessian(modes)).max(1)
    complete = certificate["complete"] and bool(np.all(residual < 1e-6)) and bool(np.all(eig < -1e-8))
    resp = g.components(modes)[1]
    rare = g.weights[np.argmax(resp, axis=1)] < .03
    return modes, dict(complete=complete, status="complete_isotropic_pair" if complete else "incomplete_isotropic_pair_certificate",
                       certificate=certificate, score_residuals=residual, max_hessian_eigenvalues=eig,
                       rare_mask=rare, rare_definition="dominant component weight <0.03; basin probabilities unknown",
                       components_are_not_modes=True,
                       proof="all critical points lie on the center line; certified 1D restriction; transverse Hessian negative")
