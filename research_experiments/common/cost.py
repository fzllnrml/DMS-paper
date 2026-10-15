"""Intrinsic algorithm cost, actual execution cost, and diagnostic cost differ."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
import numpy as np


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class Cost:
    limit: int | None = None
    budget_kind: str = "neural"
    neural_nfe: int = 0
    analytic_evaluations: int = 0
    diagnostic_neural_nfe: int = 0
    diagnostic_analytic_evaluations: int = 0
    actual_neural_nfe: int = 0
    actual_analytic_evaluations: int = 0
    calls: int = 0
    timings: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)

    @property
    def search_cost(self):
        return self.neural_nfe + self.analytic_evaluations

    @property
    def budget_used(self):
        return self.neural_nfe if self.budget_kind == "neural" else self.analytic_evaluations

    def charge(self, n, kind, diagnostic=False, cached=False):
        if type(n) != int or n < 0 or kind not in {"neural", "analytic"}:
            raise ValueError("Invalid cost charge")
        if not diagnostic and kind == self.budget_kind and self.limit is not None and self.budget_used + n > self.limit:
            raise BudgetExhausted(f"Need {n}; remaining {self.limit - self.budget_used} {kind} evaluations")
        key = "neural_nfe" if kind == "neural" else "analytic_evaluations"
        if diagnostic:
            key = "diagnostic_" + key
        setattr(self, key, getattr(self, key) + n)
        if not cached:
            key = "actual_neural_nfe" if kind == "neural" else "actual_analytic_evaluations"
            setattr(self, key, getattr(self, key) + n)
        self.calls += int(not cached)

    def event(self, key, n=1):
        self.events[key] = self.events.get(key, 0) + n

    @contextmanager
    def time(self, key):
        begin = time.perf_counter()
        try:
            yield
        finally:
            self.timings[key] = self.timings.get(key, 0.0) + time.perf_counter() - begin


class CountedScore:
    def __init__(self, field, cost):
        self.field, self.cost = field, cost
        self.dim, self.kind, self.schedule = field.dim, field.kind, field.schedule

    def __call__(self, x, t, diagnostic=False):
        a = np.asarray(x, float).reshape(-1, self.dim)
        self.cost.charge(len(a), self.kind, diagnostic)
        with self.cost.time("diagnostic_score" if diagnostic else "score"):
            s = np.asarray(self.field.score(a, int(t)), float)
        if s.shape != a.shape or not np.isfinite(s).all():
            raise FloatingPointError("Invalid score output")
        return s
