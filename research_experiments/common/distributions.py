"""Exact VP-noised GMMs, controlled geometry and evaluation-only references."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp, expit, logit
from scipy.spatial.distance import cdist


@dataclass
class Schedule:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = .02

    def __post_init__(self):
        if self.T < 2 or not 0 < self.beta_start <= self.beta_end < 1:
            raise ValueError("Invalid VP schedule")
        self.betas = np.linspace(self.beta_start, self.beta_end, self.T)
        self.alphas_cumprod = np.cumprod(1 - self.betas)

    def alpha(self, t):
        if type(t) != int or not 0 <= t < self.T:
            raise ValueError("Invalid indexed timestep")
        return float(self.alphas_cumprod[t])

    def sigma(self, t):
        return math.sqrt(1 - self.alpha(t))


class GMM:
    def __init__(self, means, covariances, weights):
        self.means = np.asarray(means, float)
        self.covariances = np.asarray(covariances, float)
        self.weights = np.asarray(weights, float)
        self.k, self.dim = self.means.shape
        if self.covariances.shape != (self.k, self.dim, self.dim):
            raise ValueError("Covariance shape")
        if self.weights.shape != (self.k,) or np.any(self.weights <= 0) or not np.isfinite(self.weights).all():
            raise ValueError("Strictly positive weights required")
        self.weights = self.weights / self.weights.sum()
        self._parameter_cache = {}
        if (not np.isfinite(self.means).all() or not np.isfinite(self.covariances).all()
                or not np.allclose(self.covariances, self.covariances.transpose(0, 2, 1))
                or np.any(np.linalg.eigvalsh(self.covariances) <= 0)):
            raise ValueError("Finite means and symmetric positive definite covariances required")

    def parameters(self, alpha=1.0):
        return self.means * math.sqrt(alpha), alpha * self.covariances + (1 - alpha) * np.eye(self.dim)[None]

    def components(self, x, alpha=1.0):
        x = np.asarray(x, float).reshape(-1, self.dim)
        if alpha not in self._parameter_cache:
            means, cov = self.parameters(alpha)
            self._parameter_cache[alpha] = means, np.linalg.inv(cov), np.linalg.slogdet(cov)[1]
        means, inv, logdet = self._parameter_cache[alpha]
        diff = x[:, None] - means
        scores = -np.einsum("kij,nkj->nki", inv, diff)
        logp = (np.log(self.weights)[None] - .5 * (self.dim * math.log(2 * math.pi)
                + logdet[None] - np.einsum("nki,nki->nk", diff, scores)))
        lp = logsumexp(logp, axis=1)
        return lp, np.exp(logp - lp[:, None]), scores, inv

    def logpdf(self, x, alpha=1.0):
        return self.components(x, alpha)[0]

    def score(self, x, alpha=1.0):
        _, r, s, _ = self.components(x, alpha)
        return np.einsum("nk,nki->ni", r, s)

    def hessian(self, x, alpha=1.0):
        _, r, s, inv = self.components(x, alpha)
        mean = np.einsum("nk,nki->ni", r, s)
        return (np.einsum("nk,nki,nkj->nij", r, s, s)
                - np.einsum("nk,kij->nij", r, inv)
                - np.einsum("ni,nj->nij", mean, mean))

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        c = rng.choice(self.k, n, p=self.weights)
        z = rng.standard_normal((n, self.dim))
        return self.means[c] + np.einsum("nij,nj->ni", np.linalg.cholesky(self.covariances)[c], z)

    def as_dict(self):
        return dict(means=self.means.tolist(), covariances=self.covariances.tolist(), weights=self.weights.tolist())


class ExactField:
    kind = "analytic"
    def __init__(self, mixture, schedule=None):
        self.mixture = mixture
        self.dim = mixture.dim
        self.schedule = schedule or Schedule()

    def score(self, x, t):
        return self.mixture.score(x, self.schedule.alpha(t))


def generate(spec, seed):
    """Separation is in within-component SD along a signal direction, not sqrt(d).

    Except for legacy_random, increasing d adds transverse noise and keeps the
    signal separation fixed. Rotate with an independent seeded orthogonal basis.
    """
    family, d = spec["family"], int(spec["dim"])
    rng = np.random.default_rng(seed)
    k = int(spec.get("components", 2))
    sep = float(spec.get("separation") if spec.get("separation") is not None else
        {"separated": 8., "moderate": 2.6, "overlap": 1.5, "rare": 6., "anisotropic": 4.,
         "many": 4., "bifurcation": 2., "uneven": 3., "basins": 2.6}.get(family, 4.))
    sigma = float(spec.get("sigma", 1.))
    weights = np.ones(k) / k
    means = np.zeros((k, d))
    means[:, 0] = (np.arange(k) - (k - 1) / 2) * sep * sigma
    cov = np.repeat((np.eye(d) * sigma ** 2)[None], k, axis=0)
    if family == "rare":
        weights[0] = float(spec.get("rare_weight", .01))
        weights[1:] = (1 - weights[0]) / (k - 1)
    elif family == "uneven":
        means[:, 0] = np.cumsum([0] + [sep * sigma * (1 if i % 2 else 3) for i in range(1, k)])
        means[:, 0] -= means[:, 0].mean()
    elif family in {"anisotropic", "basins"}:
        diag = np.geomspace(1., float(spec.get("condition", 16)), d) * sigma ** 2
        cov[:] = np.diag(diag)
        if family == "basins":
            # Different orientations: numerical, explicitly incomplete reference.
            for i in range(k):
                q, _ = np.linalg.qr(rng.standard_normal((d, d)))
                cov[i] = q @ np.diag(diag) @ q.T
    elif family == "legacy_random":
        # Same RNG call order as src.utils.distribution.GaussianMixture.random.
        accepted = []
        bounds = spec.get("bounds", [-10, 10])
        for _ in range(200000):
            if len(accepted) == k:
                break
            p = rng.uniform(*bounds, size=d)
            if all(np.linalg.norm(p - q) >= 2. for q in accepted):
                accepted.append(p)
        if len(accepted) != k:
            raise RuntimeError("Cannot place legacy means")
        means = np.array(accepted)
        sigmas = rng.uniform(.5, 1.2, k)
        weights = rng.dirichlet(np.ones(k))
        cov = np.eye(d)[None] * sigmas[:, None, None] ** 2
        return GMM(means, cov, weights)
    elif family not in {"separated", "moderate", "overlap", "many", "bifurcation"}:
        raise ValueError(f"Unknown family {family}")
    if spec.get("rotate", True):
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        means, cov = means @ q.T, np.einsum("ij,kjl,ml->kim", q, cov, q)
    return GMM(means, cov, weights)


def reference_modes(mixture, seed=701, restarts=32, tolerance=1e-7):
    """Called only AFTER candidates are final. Never used for tuning/selection.

    For <=2 components with common covariance, whitening reduces all stationary
    points to one scalar coordinate. Its derivative has at most two zeros; split
    the interval there and exhaustively bracket roots. This establishes the full
    mode set away from a degenerate bifurcation. Other mixtures use multistart
    optimization and are always labelled an incomplete numerical reference.
    """
    g, d = mixture, mixture.dim
    found, complete, degeneracy = [], False, False
    if g.k == 1:
        found, complete = [g.means[0]], True
    elif g.k == 2 and np.allclose(g.covariances[0], g.covariances[1], rtol=1e-10, atol=1e-12):
        delta = g.means[1] - g.means[0]
        inv = np.linalg.inv(g.covariances[0])
        distance = float(np.sqrt(delta @ inv @ delta))
        if distance < 1e-10:
            found, complete = [g.means[0]], True
        else:
            # y is distance along whitened line, measured from first mean.
            def score1(y):
                r = expit(math.log(g.weights[1] / g.weights[0]) + distance * y - .5 * distance ** 2)
                return distance * r - y
            cuts = [0., distance]
            if distance > 2:
                for r in ((1 - math.sqrt(1 - 4 / distance ** 2)) / 2,
                          (1 + math.sqrt(1 - 4 / distance ** 2)) / 2):
                    y = (logit(r) - math.log(g.weights[1] / g.weights[0]) + .5 * distance ** 2) / distance
                    if 0 < y < distance:
                        cuts.append(float(y))
            cuts = sorted(cuts)
            roots = [y for y in cuts if abs(score1(y)) < 1e-12]
            for a, b in zip(cuts, cuts[1:]):
                if score1(a) * score1(b) < 0:
                    roots.append(brentq(score1, a, b, xtol=1e-13))
            for y in sorted(roots):
                x = g.means[0] + y / distance * delta
                eig = np.linalg.eigvalsh(g.hessian(x)[0])
                if abs(eig.max()) <= tolerance:
                    degeneracy = True
                if eig.max() < -tolerance:
                    if not found or min(np.linalg.norm(x - p) for p in found) > 1e-6:
                        found.append(x)
            complete = not degeneracy
    else:
        rng = np.random.default_rng(seed)
        starts = np.vstack([g.means, g.sample(restarts, seed),
            rng.uniform(g.means.min(0) - 2, g.means.max(0) + 2, size=(restarts, d))])
        for x0 in starts:
            opt = minimize(lambda x: -float(g.logpdf(x)[0]), x0,
                           jac=lambda x: -g.score(x)[0], method="BFGS", options=dict(gtol=tolerance, maxiter=500))
            x = opt.x
            if np.linalg.norm(g.score(x)) <= 10 * tolerance and np.linalg.eigvalsh(g.hessian(x)[0]).max() < -tolerance:
                if not found or min(np.linalg.norm(x - p) for p in found) > 1e-5:
                    found.append(x)
    modes = np.asarray(found).reshape(-1, d)
    residuals = np.linalg.norm(g.score(modes), axis=1) if len(modes) else []
    max_eigen = np.linalg.eigvalsh(g.hessian(modes)).max(1) if len(modes) else []
    rare_mask = np.zeros(len(modes), bool)
    if len(modes):
        # This identifies a rare-associated reference, not a basin probability.
        r = g.components(modes)[1]
        rare_mask = g.weights[np.argmax(r, axis=1)] < .03
    return modes, dict(complete=complete, degenerate=degeneracy,
        status="complete_common_covariance_two_component" if complete else "incomplete_numerical_reference",
        score_residuals=np.asarray(residuals).tolist(), max_hessian_eigenvalues=np.asarray(max_eigen).tolist(),
        rare_mask=rare_mask.tolist(), rare_definition="dominant component weight < 0.03; not basin mass",
        components_are_not_modes=True)
