from __future__ import annotations

import ast
from copy import deepcopy
import io
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import typing
import unittest
from unittest.mock import patch

import numpy as np

import research_experiments_v2  # disables bytecode writes before any frozen imports

from research_experiments.common.distributions import GMM, Schedule
from research_experiments.common.config import merge
from research_experiments.common.storage import array_hash
from research_experiments_v2 import config
from research_experiments_v2 import analysis
from research_experiments_v2.adapters import split_kernel, corrected_soft_direction
from research_experiments_v2.cost import Ledger, CountedScore, BudgetExhausted
from research_experiments_v2.engine import run, output_candidates
from research_experiments_v2.initialization import noise_rows, StartFactory
from research_experiments_v2.problems import certify_scalar, certify_isotropic_pair, make_problem
from research_experiments_v2.evaluation import evaluate_synthetic
from research_experiments_v2.runner import resolve, plan_summary, run_groups
from research_experiments_v2.storage import (REPO, ROOT, RESULTS, output_path, write_json, write_arrays,
                                             completion, completed, job_claim)
from research_experiments_v2.protocol import filter_selected, validate_lock


class Quadratic:
    kind = "analytic"
    dim = 2
    schedule = Schedule(T=4)

    def score(self, x, t):
        return -np.asarray(x)


class FixedStarts:
    """A test fixture, not a scientific initializer."""
    per_start = 1

    def __init__(self, ledger):
        self.ledger = ledger

    def __call__(self, ids):
        ids = list(ids)
        self.ledger.charge(len(ids), "analytic")
        return np.asarray([[2. + i, -1.] for i in ids])


def method(**values):
    return merge(config.METHOD, dict(levels=[0], steps=1, step_size=.1, final_steps=1, **values))


def original_class(kind):
    """TEST ONLY: unchanged upstream NumPy class AST; production imports normally.

    This allows the actual Hessian/probe arithmetic to be checked without torch.
    No AST transformation, training, torch model, or sampler is executed.
    """
    from research_experiments.common.historical import CLASSES
    from src.algorithms.clustering import merge_close, agglomerative_merge
    module, name = CLASSES[kind]
    source = (REPO / "src" / "algorithms" / (module + ".py")).read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)
    env = dict(np=np, logger=logging.getLogger("v2_unit"), merge_close=merge_close,
               agglomerative_merge=agglomerative_merge, ModeFinderResultV2=SimpleNamespace, **vars(typing))
    exec(compile("from __future__ import annotations\n" + ast.get_source_segment(source, node),
                 "<unchanged legacy test fixture>", "exec"), env)
    return env[name]


class StorageFixtures(unittest.TestCase):
    def setUp(self):
        RESULTS.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="unit_v2_", dir=RESULTS)
        self.path = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_outside_output_and_parent_escape(self):
        for value in [REPO / "research_experiments" / "results" / "forbidden.json", ROOT / "../x.json", "/tmp/v2.json"]:
            with self.assertRaises(ValueError):
                write_json(value, {})

    def test_symlink_escape(self):
        link = self.path / "escape"
        link.symlink_to(REPO / "research_experiments", target_is_directory=True)
        with self.assertRaises(ValueError):
            write_json(link / "forbidden.json", {})

    def test_atomic_completion_integrity(self):
        with job_claim(self.path):
            write_json(self.path / "metadata.json", {"x": 1})
            write_arrays(self.path / "x.npz", a=np.zeros((2, 2)))
            completion(self.path, "identity")
        self.assertTrue(completed(self.path, "identity"))
        self.assertFalse(completed(self.path, "wrong"))
        write_json(self.path / "metadata.json", {"x": 2})
        self.assertFalse(completed(self.path))

    def test_single_writer(self):
        with job_claim(self.path):
            with self.assertRaises(RuntimeError):
                with job_claim(self.path):
                    pass

    def test_cache_replay_has_intrinsic_cost(self):
        field = Quadratic()
        a, b = Ledger(50), Ledger(50)
        # Cache is redirected to this V2 fixture tree; no persistent result job.
        with patch("research_experiments_v2.initialization.RESULTS", self.path):
            first = StartFactory(CountedScore(field, a), 9, 2, dict(unit=True), cache=True)
            second = StartFactory(CountedScore(field, b), 9, 2, dict(unit=True), cache=True)
            x, y = first([0, 1]), second([0, 1])
        np.testing.assert_array_equal(x, y)
        self.assertEqual(a.analytic_evaluations, b.analytic_evaluations)
        self.assertGreater(a.actual_analytic_evaluations, 0)
        self.assertEqual(b.actual_analytic_evaluations, 0)
        self.assertTrue(second.records[0]["reused"])

    def test_corrupt_lock_rejected_before_any_execution(self):
        lock = self.path / "lock.json"
        write_json(lock, dict(format="dms_v2_confirmation_lock", digest="wrong"))
        with self.assertRaises(ValueError):
            validate_lock(lock)

    def test_common_prefix_keeps_its_own_batch_for_larger_pool(self):
        field = Quadratic()
        a, b = Ledger(50), Ledger(50)
        first = StartFactory(CountedScore(field, a), 4, 2, dict(unit=True), cache=False, prefix_count=2)
        large = StartFactory(CountedScore(field, b), 4, 2, dict(unit=True), cache=False, prefix_count=2)
        x, y = first(range(2)), large(range(5))
        np.testing.assert_array_equal(x, y[:2])
        self.assertEqual(a.budget_used, 4)
        self.assertEqual(b.budget_used, 10)
        self.assertEqual(len(large.records), 2)

    def test_end_to_end_fixture_persistence_and_resume(self):
        from research_experiments.common.distributions import ExactField
        from research_experiments_v2.runner import execute_job
        class FixtureResources:
            source, runtime, overrides = dict(digest="unit-source"), dict(unit=True), {}
            reference_cache = {}
            def field(self, c):
                g = GMM([[0., 0.]], [np.eye(2)], [1.])
                return g, ExactField(g, Schedule(T=4)), None, None, {}
        c = deepcopy(config.DEFAULT)
        c.update(study="unit_fixture", population=2)
        c["problem"].update(dim=2)
        c["model"]["T"] = 4
        c["method"] = method()
        c["initialization"].update(steps=2, cache=False)
        c["budget"].update(limit=8, tolerance=0)
        c["evaluation"]["basin_diagnostics"] = False
        config.validate(c)
        job = dict(id="unit_fixture", label="unit_fixture", config=c)
        group = dict(path="unit_fixture.json", config_digest="unit")
        status, directory = execute_job(job, group, FixtureResources(), self.path)
        self.assertEqual(status, "done")
        self.assertTrue(completed(directory))
        rows = analysis.load_rows(self.path)
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(r["queries"] == 8 for r in rows))
        self.assertTrue(all(not r["ranking_eligible"] for r in rows if r["output_kind"] == "exact"))
        status, _ = execute_job(job, group, FixtureResources(), self.path)
        self.assertEqual(status, "resumed")

    def test_reference_truth_is_masked_when_incomplete(self):
        g = GMM([[0., 0.]], [np.eye(2)], [1.])
        result = dict(initial=np.array([[0., 0.]]), candidates=np.array([[0., 0.]]), merge_events=[], split_events=[])
        out = dict(native=dict(candidates=result["candidates"]))
        with patch("research_experiments_v2.evaluation.certify_isotropic_pair", return_value=(np.array([[0., 0.]]), dict(complete=False, rare_mask=[False]))):
            values, _, _ = evaluate_synthetic(g, result, out, [.1], False)
        self.assertIsNone(values["native"][0]["one_to_one_recall"])
        self.assertEqual(values["native"][0]["reference_subset_one_to_one_recall"], 1.)

    def test_source_identity_excludes_local_environment(self):
        from research_experiments_v2.storage import source_identity
        (self.path / ".venv").mkdir()
        (self.path / ".venv" / "not_source.py").write_text("fixture")
        (self.path / "source.py").write_text("fixture")
        with patch("research_experiments_v2.storage.ROOT", self.path), patch("research_experiments_v2.storage.REPO", self.path.parent):
            files = source_identity()["files_sha256"]
        self.assertTrue(any(name.endswith("source.py") for name in files))
        self.assertFalse(any(".venv" in name for name in files))

    def test_valid_lock_detects_changed_input_bytes(self):
        from research_experiments_v2.storage import fingerprint, file_hash
        data = self.path / "input.bin"
        data.write_bytes(b"unit fixture")
        payload = dict(format="dms_v2_confirmation_lock", source=dict(digest="unit"), runtime=dict(unit=True),
                       input_sha256={str(data): file_hash(data)}, groups=[])
        payload["digest"] = fingerprint(payload)
        lock = self.path / "lock.json"
        write_json(lock, payload, exclusive=True)
        with self.assertRaises(FileExistsError):
            write_json(lock, payload, exclusive=True)
        with patch("research_experiments_v2.protocol.source_identity", return_value=dict(digest="unit")), patch("research_experiments_v2.protocol.runtime_identity", return_value=dict(unit=True)):
            self.assertEqual(validate_lock(lock)["digest"], payload["digest"])
            data.write_bytes(b"changed fixture")
            with self.assertRaisesRegex(ValueError, "input changed"):
                validate_lock(lock)


class MathFixtures(unittest.TestCase):
    def test_noise_prefix_and_restarts_are_same_logical_stream(self):
        x = noise_rows(3, range(8), 2)
        np.testing.assert_array_equal(x[:2], noise_rows(3, range(2), 2))
        np.testing.assert_array_equal(x[7:8], noise_rows(3, [7], 2))

    def test_certificate_modes_are_not_component_centers(self):
        c = certify_scalar([-2., 2.], 1., [.5, .5])
        self.assertTrue(c["complete"])
        self.assertEqual(len(c["maxima"]), 2)
        self.assertEqual(len(c["minima"]), 1)
        self.assertGreater(abs(c["maxima"][0] + 2), 1e-4)

    def test_three_component_complete_reference(self):
        c = certify_scalar([-4., 0., 4.], 1., [.3, .4, .3])
        self.assertTrue(c["complete"])
        self.assertEqual(len(c["maxima"]), 3)

    def test_degeneracy_and_unresolved_boxes_are_incomplete(self):
        c = certify_scalar([-1., 1.], 1., [.5, .5], max_boxes=128)
        self.assertFalse(c["complete"])
        self.assertTrue(c["unresolved"] or c["degenerate"])
        limited = certify_scalar([-4., 0., 4.], 1., [.3, .4, .3], max_boxes=1)
        self.assertFalse(limited["complete"])

    def test_product_truth_and_score_equivalence_tiny_2d(self):
        problem = make_problem(dict(family="moderate", dim=2), 1)
        modes, info = problem.reference()
        self.assertTrue(info["complete"])
        self.assertEqual(len(modes), 4)
        self.assertEqual(set(problem.basin_labels(modes, info)), set(range(4)))
        x = np.array([[.1, .2], [-1., 2.]])
        field = problem.field()
        for t in [0, 500]:
            np.testing.assert_allclose(field.score(x, t), problem.expanded.score(x, field.schedule.alpha(t)), atol=1e-11)

    def test_shear_preserves_modes_but_not_certified_euclidean_basins(self):
        problem = make_problem(dict(family="sheared", dim=2), 1)
        modes, info = problem.reference()
        self.assertTrue(info["complete"])
        self.assertFalse(info["mass_is_euclidean_basin_probability"])
        self.assertTrue(np.all(problem.basin_labels(modes, info) == -1))

    def test_unequal_isotropic_pair_certificate(self):
        g = GMM([[-3., 0.], [3., 0.]], [np.eye(2), np.eye(2)*.64], [.4, .6])
        modes, info = certify_isotropic_pair(g)
        self.assertTrue(info["complete"])
        self.assertEqual(len(modes), 2)
        self.assertLess(np.linalg.norm(g.score(modes), axis=1).max(), 1e-6)

    def test_nearly_isotropic_is_not_promoted_to_a_proved_family(self):
        g = GMM([[-2., 0.], [2., 0.]], [np.eye(2), np.diag([1., 1.+1e-13])], [.5, .5])
        sentinel = (np.zeros((0, 2)), dict(complete=False))
        with patch("research_experiments_v2.problems.reference_modes", return_value=sentinel) as fallback:
            result = certify_isotropic_pair(g)
        self.assertIs(result, sentinel)
        fallback.assert_called_once()

    def test_original_v3f1_negative_branch_and_explicit_fix(self):
        obj = original_class("v3f1")(SimpleNamespace(dim=2, device="cpu"))
        values, vectors = np.array([-.01, -1.]), np.eye(2)
        self.assertEqual(len(obj._select_split_directions(values, vectors)), 0)
        np.testing.assert_array_equal(corrected_soft_direction(values, vectors), [[1., 0.]])

    def test_merge_records_immediate_one_to_one_coverage_loss(self):
        problem = make_problem(dict(family="moderate", dim=2), 1)
        modes, _ = problem.reference()
        before = modes[[0, 3]]
        after = before.mean(0, keepdims=True)
        result = dict(initial=before, candidates=after, split_events=[], intervention_events_observed=True,
                      merge_events=[dict(stage="unit", t=0, before=before, after=after, groups=[[0, 1]])])
        _, diagnostics, _ = evaluate_synthetic(problem, result, dict(native=dict(candidates=after)), [.25], False)
        record = diagnostics["merge"][0]["direct_mode_metrics"][0]
        self.assertEqual(record["near_mode_cross_groups"], 1)
        self.assertEqual(record["n_matched_before"], 2)
        self.assertEqual(record["event_lost_matches"], 2)


class BudgetFixtures(unittest.TestCase):
    def test_mixed_queries_cannot_evade_hard_cap(self):
        ledger = Ledger(3)
        ledger.charge(2, "neural")
        ledger.charge(1, "analytic")
        with self.assertRaises(BudgetExhausted):
            ledger.charge(1, "analytic")
        ledger.charge(10, "neural", diagnostic=True)
        self.assertEqual(ledger.budget_used, 3)
        self.assertEqual(ledger.diagnostic_neural_nfe, 10)

    def test_matched_partial_batch_spends_exact_budget(self):
        ledger = Ledger(7)
        result = run(CountedScore(Quadratic(), ledger), FixedStarts(ledger), method(),
                     dict(limit=7, regime="matched", tolerance=0), 2)
        self.assertEqual(ledger.budget_used, 7)
        self.assertEqual(len(result["candidates"]), 2)

    def test_cap_does_not_refill(self):
        ledger = Ledger(20)
        run(CountedScore(Quadratic(), ledger), FixedStarts(ledger), method(),
            dict(limit=20, regime="cap", tolerance=0), 2)
        self.assertEqual(ledger.budget_used, 6)

    def test_restart_transaction_no_dangling_anchors(self):
        ledger = Ledger(9)
        result = run(CountedScore(Quadratic(), ledger), FixedStarts(ledger), method(refill="restarts"),
                     dict(limit=9, regime="matched", tolerance=0), 2)
        self.assertEqual(len(result["initial"]), 3)
        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual(result["lineage"], [[0], [1], [2]])
        self.assertEqual(ledger.budget_used, 9)

    def test_unaffordable_restart_is_not_started(self):
        ledger = Ledger(8)
        result = run(CountedScore(Quadratic(), ledger), FixedStarts(ledger), method(refill="restarts"),
                     dict(limit=8, regime="matched", tolerance=0), 2)
        self.assertEqual(len(result["initial"]), 2)
        self.assertFalse(result["restart_events"])

    def test_split_cost_includes_actual_2d_hessian_and_probes(self):
        ledger = Ledger(100)
        m = method()
        cfg = dict(m["split"], kind="v2", eigen_threshold=-2.)
        with patch("research_experiments_v2.adapters.get_class", side_effect=original_class):
            split_kernel(CountedScore(Quadratic(), ledger), np.array([.5, 0.]), 0, cfg, m)
        self.assertEqual(ledger.events["finite_difference_score_evaluations"], 4)
        self.assertEqual(ledger.components["split_hessian"]["intrinsic_analytic"], 4)
        self.assertGreater(ledger.budget_used, 4)

    def test_partial_split_retains_parent_and_spent_cost(self):
        ledger = Ledger(4)
        m = method()
        m.update(steps=0, final_steps=0, refill="none")
        m["split"]["kind"] = "v2"
        with patch("research_experiments_v2.adapters.get_class", side_effect=original_class):
            result = run(CountedScore(Quadratic(), ledger), FixedStarts(ledger), m,
                         dict(limit=4, regime="cap", tolerance=0), 1)
        np.testing.assert_array_equal(result["candidates"], [[2., -1.]])
        self.assertEqual(ledger.budget_used, 4)
        self.assertFalse(result["split_events"][0]["completed"])

    def test_strict_k_and_upper_bound_are_distinct(self):
        x, lineage = np.array([[0., 0.], [0., 0.], [2., 0.]]), [[0], [1], [2]]
        policy = dict(name="eq3", kind="exact", k=3, radius=1., metric="l2", representative="medoid")
        strict = output_candidates(x, lineage, policy)
        upper = output_candidates(x, lineage, dict(policy, kind="at_most"))
        self.assertEqual(len(strict["candidates"]), 2)
        self.assertFalse(strict["k_constraint_met"])
        self.assertTrue(upper["k_constraint_met"])

    def test_sampling_matched_uses_complete_trajectories_only(self):
        ledger = Ledger(9)
        starts = FixedStarts(ledger)
        starts.per_start = 2
        # FixedStarts normally charges one; adapt to emulate two-query trajectories.
        class TwoQueryStarts(FixedStarts):
            per_start = 2
            def __call__(self, ids):
                ids = list(ids)
                self.ledger.charge(len(ids), "analytic")
                return super().__call__(ids)
        m = method(name="sampling", refill="none")
        result = run(CountedScore(Quadratic(), ledger), TwoQueryStarts(ledger), m,
                     dict(limit=9, regime="matched", tolerance=1), 2)
        self.assertEqual(len(result["initial"]), 4)
        self.assertEqual(ledger.budget_used, 8)

    def test_original_v3f2_and_v2_coordinator_tiny_parity(self):
        a, b = Ledger(100), Ledger(100)
        m = method(refill="none")
        m["merge"].update(when="every", radius=.01, radius_factor=0., representative="mean")
        new = run(CountedScore(Quadratic(), a), FixedStarts(a), m, dict(limit=100, regime="cap", tolerance=0), 2)
        old_method = dict(m, implementation="historical_v3f2")
        with patch("research_experiments.common.historical.get_class", side_effect=original_class):
            old = run(CountedScore(Quadratic(), b), FixedStarts(b), old_method, dict(limit=100, regime="cap", tolerance=0), 2)
        np.testing.assert_allclose(sorted(new["candidates"].tolist()), sorted(old["candidates"].tolist()), atol=1e-12)
        self.assertEqual(a.budget_used, b.budget_used)
        self.assertFalse(old["intervention_events_observed"])


def sample_row(label, **updates):
    row = dict(cohort="unit", domain="exact", field="exact", phase="pilot", study="unit",
               dimension=2, family="unit", budget=10, regime="matched", output_policy="native", output_geometry="g",
               radius=.25, P0=2, zero_convention="clean_zero", distribution_seed=0, search_seed=0, model_seed=None,
               density_identity="density", checkpoint_sha256=None, label=label, treatment=label, job_id=label,
               ranking_eligible=True, nfe_valid=True, algorithm_valid=True, k_constraint_met=True,
               reference_complete=True, compatibility_verified=True, queries=10, nfe_tolerance=0,
               common_noise_sha256="noise", common_initial_sha256="starts", one_to_one_recall=.5)
    row.update(updates)
    return row


class StatisticsFixtures(unittest.TestCase):
    def test_no_interval_from_one_run(self):
        s = analysis.interval([[1.]], 200)
        self.assertIsNone(s["ci_low"])
        self.assertEqual(s["n_search_runs"], 1)

    def test_equal_cluster_weight_not_particle_or_run_weight(self):
        s = analysis.interval([[0.], [1., 1., 1.]], 500)
        self.assertEqual(s["mean"], .5)
        self.assertEqual(s["n_clusters"], 2)

    def test_insufficient_tail_resolution_is_not_a_fake_corrected_ci(self):
        s = analysis.interval([[0., 1.]], 100, alpha=.0001)
        self.assertIsNone(s["ci_low"])
        self.assertIn("tail", s["interval_status"])

    def test_nonempty_contrast_report_even_before_any_job(self):
        group = dict(jobs=[], contrasts=[dict(id="test", left="a", right="b", metrics=["one_to_one_recall"])])
        reports, deltas = analysis.contrasts([], [group], 200)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["n_pairs"], 0)
        self.assertFalse(deltas)

    def test_paired_delta_and_invalid_exclusion(self):
        a, b = sample_row("a", one_to_one_recall=1.), sample_row("b")
        group = dict(jobs=[dict(config=dict(study="unit"))], contrasts=[dict(id="test", left="a", right="b", metrics=["one_to_one_recall"])])
        reports, _ = analysis.contrasts([a, b], [group], 500)
        self.assertEqual(reports[0]["mean"], .5)
        b["k_constraint_met"] = False
        b["ranking_eligible"] = False
        reports, deltas = analysis.contrasts([a, b], [group], 500)
        self.assertEqual(reports[0]["n_search_runs"], 0)
        self.assertFalse(deltas[0]["ranking_eligible"])

    def test_duplicate_settings_cannot_be_silently_paired(self):
        group = dict(jobs=[dict(config=dict(study="unit"))], contrasts=[dict(id="test", left="a", right="b", metrics=["one_to_one_recall"])])
        with self.assertRaises(ValueError):
            analysis.contrasts([sample_row("a"), sample_row("a", treatment="other"), sample_row("b")], [group])

    def test_noise_only_pairing_is_explicit_for_different_ddim_schedules(self):
        a, b = sample_row("a"), sample_row("b", common_initial_sha256="other")
        self.assertTrue(analysis.eligible_pair(a, b, dict(pairing="starts")))
        self.assertFalse(analysis.eligible_pair(a, b, dict(pairing="noise")))

    def test_fixed_models_do_not_gain_ci_from_more_particles_or_models_alone(self):
        s = analysis.interval([[0.], [1.]], 500, fixed_clusters=True)
        self.assertIsNone(s["ci_low"])
        self.assertIn("fixed pretrained", s["scope"])

    def test_aggregate_front_uses_common_eligible_units(self):
        rows = [sample_row(label, search_seed=seed, one_to_one_recall=.5, soft_error=.1, K=2)
                for label in ["a", "b"] for seed in [0, 1]]
        rows[-1]["ranking_eligible"] = False
        fronts, status = analysis.aggregate_pareto(rows, 500)
        self.assertEqual(len(fronts), 2)
        self.assertTrue(all(r["n_runs"] == 1 for r in fronts))
        self.assertEqual(status[0]["n_common_eligible_units"], 1)

    def test_analytic_and_neural_costs_are_not_one_pareto_price(self):
        rows = [sample_row("a", field="exact", soft_error=.2, K=2, queries=10),
                sample_row("b", field="learned", soft_error=.1, K=1, queries=1)]
        fronts = analysis.pareto(rows)
        self.assertTrue(all(r["pareto"] for r in fronts))

    def test_partial_confirmation_cannot_support_hypothesis(self):
        paired = [dict(study="confirmation_exact", contrast="c", metric="rare_recall", primary=True,
                       mean=.2, familywise_ci_low=.1, familywise_ci_high=.3, dimension=2, family="rare")]
        hypotheses = [dict(id="H", clauses=[dict(study="confirmation_exact", contrast="c", metric="rare_recall", direction="greater", threshold=0.)])]
        audit = [dict(study=["confirmation_exact"], missing_jobs=1)]
        decisions = analysis.hypothesis_decisions(paired, hypotheses, audit, [])
        self.assertEqual(decisions[0]["status"], "incomplete planned confirmation")


class ProtocolFixtures(unittest.TestCase):
    def test_all_configs_expand_without_torch_or_scores(self):
        _, paths = config.manifest(ROOT / "configs" / "pilot_all.json")
        groups = resolve(paths)
        plan = plan_summary(groups)
        self.assertEqual(plan["n_jobs"], 4408)
        self.assertTrue(plan["no_score_evaluated"])
        self.assertEqual(plan["by_domain"]["mnist"], 250)
        self.assertGreater(analysis.planned_primary_count(groups), 0)

    def test_confirmation_cannot_run_through_unlocked_runner(self):
        with self.assertRaisesRegex(ValueError, "validated lock"):
            run_groups([dict(jobs=[dict(config=dict(phase="confirmation"))])])

    def test_unreviewed_or_unoptimized_selection_cannot_lock(self):
        with self.assertRaises(ValueError):
            filter_selected([], dict(reviewed=False))
        with self.assertRaises(ValueError):
            filter_selected([], dict(reviewed=True, review_notes="unit", hypotheses=["unit"], chosen_labels={"exact": [], "mnist": []}))

    def test_unknown_config_keys_and_hidden_v1_policies_rejected(self):
        with self.assertRaises(ValueError):
            merge(config.DEFAULT, dict(unknown=True))
        c = deepcopy(config.DEFAULT)
        c["method"]["budget_policy"] = "restarts"
        with self.assertRaises(ValueError):
            config.validate(c)

    def test_confirmation_blueprints_have_fresh_seeds_and_cap_control(self):
        _, paths = config.manifest(ROOT / "configs" / "confirmation_all.json")
        groups = resolve(paths)
        jobs = [j for g in groups for j in g["jobs"]]
        self.assertEqual(len(jobs), 1740)
        self.assertTrue(all(j["config"]["seed"] >= 1000 for j in jobs))
        self.assertTrue(any(j["config"]["study"] == "confirmation_exact_cap" for j in jobs))

    def test_arbitrary_external_model_map_is_read_only_path_resolution(self):
        from research_experiments_v2.storage import input_path
        value = input_path("runs/mnist/checkpoints/best.pt", dict(model_paths={"runs/mnist/checkpoints/best.pt": "/tmp/external42.pt"}))
        self.assertEqual(value, Path("/tmp/external42.pt").resolve())

    def test_selected_confirmation_resolves_optimized_single_contrast(self):
        from research_experiments_v2.protocol import resolved_hypotheses
        _, paths = config.manifest(ROOT / "configs" / "confirmation_all.json")
        groups = resolve(paths)
        selection = dict(reviewed=True, review_notes="unit protocol fixture", hypotheses=[dict(id="H", claim="unit",
            clauses=[dict(study="confirmation_mnist", contrast="SELECTED_SINGLE_VS_CURRENT", metric="pixel_knn", direction="greater", threshold=0.)])],
            chosen_labels=dict(exact=["M1_ddim", "M2_single", "M3_multi", "M4_merge", "M6_split_v2", "M7_extra_starts", "M8_merge_restarts"],
                               mnist=["DDIM105", "DDIM50_matched_pool", "multi_l2_003", "single300_l2_006"]),
            candidates=[dict(label="single300_l2_006", eligible_for_selection=True, support_frontier=True)])
        selected = filter_selected(groups, selection)
        self.assertEqual(sum(len(g["jobs"]) for g in selected), 1480)
        hypotheses = resolved_hypotheses(selection, selected)
        self.assertIn("single300_l2_006", hypotheses[0]["clauses"][0]["contrast"])


if __name__ == "__main__":
    unittest.main()
