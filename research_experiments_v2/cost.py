"""Reuse v1 score counting; add a hard combined-query cap and component ledger."""
from contextlib import contextmanager

from research_experiments.common.cost import Cost, CountedScore, BudgetExhausted


class Ledger(Cost):
    def __init__(self, limit):
        super().__init__(limit=limit, budget_kind="combined_queries")
        self.components = {}
        self.current_component = "unspecified"

    @property
    def budget_used(self):
        return self.search_cost

    @property
    def remaining(self):
        return self.limit - self.search_cost if self.limit is not None else 2**63

    def charge(self, n, kind, diagnostic=False, cached=False):
        if not diagnostic and self.limit is not None and self.search_cost + n > self.limit:
            raise BudgetExhausted(f"Need {n} queries; remaining {self.remaining}")
        super().charge(n, kind, diagnostic, cached)
        entry = self.components.setdefault(self.current_component, {})
        key = ("diagnostic_" if diagnostic else "intrinsic_") + kind
        entry[key] = entry.get(key, 0) + n
        if not cached:
            key = "executed_" + kind
            entry[key] = entry.get(key, 0) + n

    @contextmanager
    def scope(self, name):
        previous, self.current_component = self.current_component, name
        try:
            with self.time(name):
                yield
        finally:
            self.current_component = previous

    def as_dict(self):
        from dataclasses import asdict
        return dict(asdict(self), components=self.components, budget_used=self.budget_used,
                    remaining=self.remaining,
                    units="per-point score queries; analytic and neural counters remain separate",
                    cached_cost_policy="charge intrinsic queries, record executed queries separately")
