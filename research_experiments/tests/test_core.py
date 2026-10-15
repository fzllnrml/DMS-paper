"""Only tiny artificial arrays/functions; no checkpoint load or training."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
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

from research_experiments.common.config import DEFAULT, load, merge, validate, input_path
from research_experiments.common.cost import Cost, CountedScore, BudgetExhausted
from research_experiments.common.distributions import GMM, Schedule, ExactField, generate, reference_modes
from research_experiments.common.geometry import step_delta, cluster, select, farthest_indices
from research_experiments.common.metrics import evaluate_modes, one_to_one, pareto_mask, support_percentile
from research_experiments.common.search import run_search, split_points
from research_experiments.common.storage import ALLOWED, REPO, safe_path, write_json, write_arrays, file_hash, JobStore
from research_experiments.common.statistics import hierarchical_interval


class Quadratic:
    kind = "analytic"
    def __init__(self, dim=2):
        self.dim, self.schedule = dim, Schedule(T=4)
    def score(self, x, t):
        return -np.asarray(x)


def source_class(kind):
    """Exercise upstream NumPy class bodies without the optional torch import.

    This test-only loader compiles the UNCHANGED class AST from the checked-out
    source. It does not replace score arithmetic. Production uses normal imports.
    No model construction, tensors or sampling methods are called here.
    """
    from research_experiments.common.historical import CLASSES
    from src.algorithms.clustering import merge_close, agglomerative_merge
    module, name = CLASSES[kind]
    text = (REPO / "src" / "algorithms" / (module + ".py")).read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == name)
    env = dict(np=np, logger=logging.getLogger("legacy_unit_fixture"),
        merge_close=merge_close, agglomerative_merge=agglomerative_merge,
        ModeFinderResultV2=SimpleNamespace, ModeFinderResult=SimpleNamespace, **vars(typing))
    exec(compile("from __future__ import annotations\n" + ast.get_source_segment(text, node),
                 "<unchanged upstream class: " + name + ">", "exec"), env)
    return env[name]


class Fixtures(unittest.TestCase):
    def setUp(self):
        ALLOWED.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="unit_", dir=ALLOWED)
        self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def score(self, limit=None):
        return CountedScore(Quadratic(), Cost(limit=limit, budget_kind="analytic"))
    def method(self, **kwargs):
        m = deepcopy(DEFAULT["method"])
        m.update(dict(levels=[0], steps=2, step_size=.1, final_steps=0))
        return merge(m, kwargs)

    def test_rms_equals_fixed_l2_784(self):
        s = np.random.default_rng(0).normal(size=(2, 784))
        np.testing.assert_allclose(step_delta(s, .03, "rms"), step_delta(s, .84, "l2"), rtol=1e-14, atol=1e-15)

    def test_rms_equivalence_per_dimension(self):
        for d in [1, 2, 10, 30, 50]:
            s = np.ones((2, d))
            np.testing.assert_allclose(step_delta(s, .1, "rms"), step_delta(s, .1 * np.sqrt(d), "l2"))

    def test_zero_score_is_finite(self):
        np.testing.assert_array_equal(step_delta(np.zeros((2, 3)), .1), np.zeros((2, 3)))

    def test_raw_score_step(self):
        np.testing.assert_array_equal(step_delta(np.array([[2., 3.]]), .5, "raw"), [[1., 1.5]])

    def test_invalid_step(self):
        with self.assertRaises(ValueError):
            step_delta(np.ones((2, 2)), -1.)

    def test_matching_maximizes_cardinality_before_distance(self):
        pairs = one_to_one([[.1, .2], [.15, 2.]], .3)
        self.assertEqual(set(map(tuple, pairs)), {(0, 1), (1, 0)})

    def test_matching_minimizes_distance_at_equal_cardinality(self):
        pairs = one_to_one([[.1, .8], [.8, .2]], 1.)
        self.assertEqual(set(map(tuple, pairs)), {(0, 0), (1, 1)})

    def test_one_candidate_cannot_cover_two_modes_in_matching(self):
        m = evaluate_modes([[-.1], [.1]], [[0.]], [.2])[0]
        self.assertEqual(m["recall"], 1.)
        self.assertEqual(m["one_to_one_recall"], .5)

    def test_duplicates_are_penalized_by_one_to_one_precision(self):
        m = evaluate_modes([[0.]], [[0.], [0.], [0.]], [.1])[0]
        self.assertEqual(m["precision"], 1.)
        self.assertEqual(m["one_to_one_precision"], 1 / 3)
        self.assertEqual(m["duplicate_count"], 2)

    def test_empty_candidates(self):
        m = evaluate_modes([[0., 0.]], np.empty((0, 2)), [.1])[0]
        self.assertEqual(m["recall"], 0.)
        self.assertTrue(np.isinf(m["soft_error"]))
        self.assertEqual(m["duplicate_count"], 0)

    def test_empty_reference(self):
        m = evaluate_modes(np.empty((0, 2)), [[0., 0.]], [.1])[0]
        self.assertIsNone(m["recall"])
        self.assertIsNone(m["one_to_one_precision"])

    def test_single_mode(self):
        m = evaluate_modes([[0., 0.]], [[.03, .04]], [.1])[0]
        self.assertEqual(m["one_to_one_f1"], 1.)
        self.assertAlmostEqual(m["localization_error"], .05)

    def test_rare_mask(self):
        m = evaluate_modes([[0.], [2.]], [[0.]], [.1], [False, True])[0]
        self.assertEqual(m["rare_recall"], 0.)

    def test_complete_linkage_blocks_chain(self):
        x = np.array([[0.], [.9], [1.8]])
        self.assertEqual(len(cluster(x, 1., method="complete")[0]), 2)
        self.assertEqual(len(cluster(x, 1., method="single")[0]), 1)

    def test_mean_can_move_representative_medoid_cannot(self):
        x = np.array([[0.], [.8]])
        np.testing.assert_array_equal(cluster(x, 1., "mean")[0], [[.4]])
        self.assertIn(float(cluster(x, 1., "medoid")[0][0, 0]), [0., .8])

    def test_zero_radius_removes_exact_duplicates(self):
        self.assertEqual(len(cluster(np.array([[0.], [0.], [1.]]), 0)[0]), 2)

    def test_empty_merge(self):
        self.assertEqual(cluster(np.empty((0, 3)), 1.)[0].shape, (0, 3))

    def test_fixed_k_in_tied_cloud(self):
        self.assertEqual(len(cluster(np.zeros((5, 2)), 0., k=3)[0]), 3)

    def test_direction_gate(self):
        out = cluster(np.array([[0.], [.01]]), .1, directions=np.array([[1.], [-1.]]), cosine_threshold=.9)[0]
        self.assertEqual(len(out), 2)

    def test_farthest_no_duplicate_indices(self):
        self.assertEqual(len(set(farthest_indices(np.zeros((4, 2)), 3))), 3)

    def test_facility_uses_only_input_points(self):
        x = np.arange(4.).reshape(-1, 1)
        out = select(x, dict(kind="facility", k=2))
        self.assertEqual(len(out), 2)
        self.assertTrue(all(float(p[0]) in x[:, 0] for p in out))

    def test_empirical_mean_shift(self):
        out = select(np.array([[-.1], [.1]]), dict(kind="mean_shift", k=None, bandwidth=1., iterations=4))
        self.assertTrue(np.max(np.abs(out)) < .01)

    def test_neural_nfe_counts_objects(self):
        cost = Cost()
        cost.charge(7, "neural")
        cost.charge(3, "neural")
        self.assertEqual(cost.neural_nfe, 10)
        self.assertEqual(cost.calls, 2)

    def test_analytic_cost_is_separate(self):
        cost = Cost()
        cost.charge(3, "analytic")
        self.assertEqual(cost.neural_nfe, 0)
        self.assertEqual(cost.analytic_evaluations, 3)

    def test_diagnostics_do_not_spend_algorithm_budget(self):
        cost = Cost(limit=1)
        cost.charge(100, "neural", diagnostic=True)
        cost.charge(1, "neural")
        self.assertEqual(cost.diagnostic_neural_nfe, 100)

    def test_cached_intrinsic_cost_is_still_charged(self):
        cost = Cost()
        cost.charge(50, "neural", cached=True)
        self.assertEqual(cost.neural_nfe, 50)
        self.assertEqual(cost.actual_neural_nfe, 0)

    def test_budget_failure_is_atomic(self):
        cost = Cost(limit=2)
        with self.assertRaises(BudgetExhausted):
            cost.charge(3, "neural")
        self.assertEqual(cost.neural_nfe, 0)

    def test_counted_score_shape(self):
        score = self.score()
        np.testing.assert_array_equal(score([[1., 2.], [3., 4.]], 0), [[-1., -2.], [-3., -4.]])
        self.assertEqual(score.cost.analytic_evaluations, 2)

    def test_exact_score_matches_density_derivative(self):
        g = GMM([[-1., 0.], [1., .5]], [np.eye(2), np.eye(2) * 2], [.3, .7])
        x, h = np.array([[.2, -.1]]), 1e-5
        grad = [(g.logpdf(x + np.eye(2)[j] * h)[0] - g.logpdf(x - np.eye(2)[j] * h)[0]) / (2 * h) for j in range(2)]
        np.testing.assert_allclose(g.score(x)[0], grad, atol=1e-8)

    def test_exact_hessian_matches_score_derivative(self):
        g = GMM([[-1., 0.], [1., .5]], [np.eye(2), np.eye(2) * 2], [.3, .7])
        x, h = np.array([[.2, -.1]]), 1e-5
        H = np.column_stack([(g.score(x + np.eye(2)[j] * h)[0] - g.score(x - np.eye(2)[j] * h)[0]) / (2 * h) for j in range(2)])
        np.testing.assert_allclose(g.hessian(x)[0], H, atol=1e-8)

    def test_noisy_score_single_gaussian(self):
        g = GMM([[2., 0.]], [np.eye(2) * 4], [1.])
        np.testing.assert_allclose(g.score([[0., 0.]], .25), [[1 / 1.75, 0.]])

    def test_reproducible_distributions(self):
        spec = dict(family="moderate", dim=2, components=2)
        np.testing.assert_array_equal(generate(spec, 1).means, generate(spec, 1).means)

    def test_dimension_does_not_increase_signal_separation(self):
        for d in [2, 10, 30, 50]:
            g = generate(dict(family="moderate", dim=d, components=2), 0)
            self.assertAlmostEqual(np.linalg.norm(g.means[1] - g.means[0]), 2.6)

    def test_common_covariance_reference_one_gaussian(self):
        refs, status = reference_modes(GMM([[2., 1.]], [np.eye(2)], [1.]))
        self.assertTrue(status["complete"])
        np.testing.assert_array_equal(refs, [[2., 1.]])

    def test_overlap_components_are_not_modes(self):
        refs, status = reference_modes(GMM([[-.5], [.5]], [[[1.]], [[1.]]], [.5, .5]))
        self.assertTrue(status["complete"])
        self.assertEqual(len(refs), 1)
        self.assertAlmostEqual(refs[0, 0], 0.)

    def test_two_separated_gaussians_have_two_reference_modes(self):
        refs, status = reference_modes(GMM([[-2.], [2.]], [[[1.]], [[1.]]], [.5, .5]))
        self.assertTrue(status["complete"])
        self.assertEqual(len(refs), 2)

    def test_degenerate_bifurcation_is_not_certified_strict_mode(self):
        _, status = reference_modes(GMM([[-1.], [1.]], [[[1.]], [[1.]]], [.5, .5]))
        self.assertFalse(status["complete"])
        self.assertTrue(status["degenerate"])

    def test_ascent_is_independent_without_population_operations(self):
        x, cfg = np.array([[1., 0.], [0., 1.]]), self.method()
        together = run_search(self.score(), x, cfg)["candidates"]
        separate = np.vstack([run_search(self.score(), p[None], cfg)["candidates"] for p in x])
        np.testing.assert_array_equal(together, separate)

    def test_projection_records_actual_step(self):
        score = self.score()
        score.field.score = lambda x, t: np.ones_like(x)
        result = run_search(score, np.array([[.99, .99]]), self.method(box=[-1., 1.], steps=1))
        self.assertGreater(result["movement"][0]["outside_fraction"], 0)
        self.assertLess(result["movement"][0]["actual_l2"][0], .1)

    def test_budget_retains_last_completed_step(self):
        r = run_search(self.score(2), np.array([[1., 0.]]), self.method(steps=5, save_every_step=False))
        self.assertTrue(r["budget_exhausted"])
        np.testing.assert_allclose(r["candidates"], [[.8, 0.]])

    def test_legacy_split_fd_and_probe_cost(self):
        from research_experiments.common import historical
        score = self.score()
        cfg = self.method()
        split = deepcopy(cfg["split"])
        split.update(kind="v2", eigen_threshold=0., probe_steps=3)
        with patch.object(historical, "get_class", side_effect=source_class):
            x, parents, ev = split_points(score, np.array([[1., 0.]]), 0, split, cfg)
        self.assertEqual(len(x), 1)
        self.assertEqual(score.cost.analytic_evaluations, 2 * 2 + 3)
        self.assertEqual(parents, [0])
        self.assertFalse(ev[0]["accepted"])

    def test_legacy_split_creates_children_with_full_cost(self):
        from research_experiments.common import historical
        score = self.score()
        score.field.score = lambda x, t: x * np.array([1., -1.])
        cfg, s = self.method(), deepcopy(DEFAULT["method"]["split"])
        s.update(kind="v2", eigen_threshold=0., probe_steps=2, absolute_threshold=.1)
        with patch.object(historical, "get_class", side_effect=source_class):
            out, parents, events = split_points(score, np.zeros((1, 2)), 0, s, cfg)
        self.assertEqual(len(out), 2)
        self.assertEqual(score.cost.analytic_evaluations, 4 + 4)
        self.assertEqual(parents, [0, 0])
        self.assertTrue(events[0]["accepted"])

    def test_v3f1_negative_ratio_does_not_activate_positive_threshold(self):
        cls = source_class("v3f1")
        from research_experiments.common.historical import LegacyProxy
        obj = cls(LegacyProxy(self.score()), softness_threshold=.15)
        self.assertEqual(len(obj._select_split_directions(np.array([-2., -.1]), np.eye(2))), 0)

    def test_legacy_nfe_reset_cannot_erase_total(self):
        from research_experiments.common.historical import LegacyProxy
        score = self.score()
        proxy = LegacyProxy(score)
        proxy.score_numpy(np.ones(2), 0)
        proxy.reset_nfe()
        proxy.score_numpy(np.ones(2), 0)
        self.assertEqual(proxy.nfe, 1)
        self.assertEqual(score.cost.analytic_evaluations, 2)

    def test_full_legacy_v2_numpy_body_runs_on_tiny_quadratic(self):
        from research_experiments.common import historical
        cfg = self.method(steps=1, final_steps=1)
        cfg["name"] = "historical_v2"
        cfg["split"]["eigen_threshold"] = 0.
        with patch.object(historical, "get_class", side_effect=source_class):
            out = historical.run_legacy(self.score(), np.array([[1., 0.]]), cfg)
        self.assertEqual(out["candidates"].shape, (1, 2))

    def test_all_shipped_configs_validate_without_models(self):
        for p in sorted((REPO / "research_experiments" / "configs").glob("*.json")):
            with self.subTest(config=p.name):
                jobs, _, _ = load(p)
                self.assertTrue(jobs)

    def test_unknown_keys_rejected(self):
        with self.assertRaises(ValueError):
            merge(DEFAULT, {"method": {"stepps": 10}})

    def test_bad_single_level_step_list_rejected(self):
        c = deepcopy(DEFAULT)
        c["method"].update(levels=[300], steps=[1, 2])
        with self.assertRaises(ValueError):
            validate(c)

    def test_dry_run_performs_no_writes_or_models(self):
        from research_experiments.run import main
        before = set(ALLOWED.rglob("*"))
        with patch("research_experiments.run.run_job", side_effect=AssertionError("must not execute")):
            with redirect_stdout(io.StringIO()) as buf:
                rc = main(["run", "--config", "research_experiments/configs/09_mnist_diagnostics.pilot.json", "--dry-run", "--max-jobs", "1"])
        self.assertEqual(rc, 0)
        self.assertFalse(json.loads(buf.getvalue())["writes"])
        self.assertEqual(set(ALLOWED.rglob("*")), before)

    def test_output_escape_rejected(self):
        with self.assertRaises(ValueError):
            safe_path(ALLOWED / ".." / "escape")

    def test_output_symlink_escape_rejected(self):
        link = self.root / "escape"
        link.symlink_to(REPO / "configs", target_is_directory=True)
        with self.assertRaises(ValueError):
            write_json(link / "do_not_write.json", {})

    def test_numeric_roundtrip_no_pickle(self):
        p = self.root / "tiny.npz"
        write_arrays(p, x=np.array([[1., 2.]]))
        with np.load(p, allow_pickle=False) as z:
            np.testing.assert_array_equal(z["x"], [[1., 2.]])

    def test_completed_store_cannot_be_overwritten(self):
        with JobStore(self.root, "fixture") as store:
            store.json("test.json", {"unit_fixture": True})
            store.complete()
        with self.assertRaises(FileExistsError):
            JobStore(self.root, "fixture")
        JobStore(self.root, "fixture", resume=True).verify()

    def test_corruption_detected(self):
        with JobStore(self.root, "fixture") as store:
            store.json("test.json", {"unit_fixture": True})
            store.complete()
        (self.root / "fixture" / "test.json").write_text("{}")
        with self.assertRaises(ValueError):
            JobStore(self.root, "fixture", resume=True).verify()

    def test_lock_serializes_concurrent_writer(self):
        with JobStore(self.root, "fixture"):
            with self.assertRaises(FileExistsError):
                with JobStore(self.root, "fixture", resume=True):
                    pass

    def test_bootstrap_single_run_no_fake_interval(self):
        ci = hierarchical_interval([[1.]])
        self.assertIsNone(ci["low"])
        self.assertEqual(ci["n_runs"], 1)

    def test_bootstrap_equal_cluster_weight(self):
        ci = hierarchical_interval([[0.] * 3, [2.]], repeats=100)
        self.assertEqual(ci["mean"], 1.)

    def test_pareto_does_not_weight_metrics(self):
        np.testing.assert_array_equal(pareto_mask([[1, 1], [2, 2], [1, 1], [0, 3]], [True, True]), [True, False, True, True])

    def test_support_midrank_ties(self):
        np.testing.assert_allclose(support_percentile([1, 2, 3], [1, 2, 3]), [100-100/6, 50, 100/6])

    def test_paired_analysis_rejects_different_starts(self):
        from research_experiments.analysis.analyze import paired_contrasts
        common = dict(phase="development", pair_key="same", radius=.1, cluster="one", recall=1.)
        rows = [dict(common, label="a", setting="a", initial_hash="x"), dict(common, label="b", setting="b", initial_hash="y")]
        with self.assertRaises(ValueError):
            paired_contrasts(rows, [("a", "b")], repeats=100)

    def test_retrospective_manifest_on_tiny_explicit_arrays(self):
        from research_experiments.common.retrospective import analyze_archives
        p = self.root / "toy.npz"
        write_arrays(p, ref=np.array([[0., 0.]]), cand=np.array([[0., 0.], [0., 0.]]))
        manifest = self.root / "manifest.json"
        write_json(manifest, dict(runs=[dict(path="toy.npz", method="v2", dimension=2,
            reference_key="ref", candidate_key="cand", reference_complete=True, neural_nfe=7)]))
        report = analyze_archives([], dict(radii=[.1], reductions=[], max_cluster_points=20), manifest)
        self.assertEqual(report["recomputed"][0]["one_to_one_precision"], .5)
        self.assertEqual(report["available_methods"], ["v2"])

    def test_missing_archive_never_invents_results(self):
        from research_experiments.common.retrospective import analyze_archives
        report = analyze_archives([self.root / "absent"], dict(radii=[.1], reductions=[], max_cluster_points=20))
        self.assertEqual(report["recomputed"], [])

    def test_all_block_entrypoints_import_without_execution(self):
        for path in sorted((REPO / "research_experiments").glob("[0-9][0-9]_*/run.py")):
            spec = importlib.util.spec_from_file_location("unit_import_" + path.parent.name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertTrue(callable(module.main))

    def test_local_checkpoint_override(self):
        out = input_path("runs/mnist/checkpoints/best.pt", {"mnist_root": str(self.root)})
        self.assertEqual(out, self.root / "mnist/checkpoints/best.pt")

    def test_ddim_counts_each_artificial_field_evaluation(self):
        from research_experiments.common.initialization import ddim
        score = self.score()
        x = ddim(score, 2, 2, 1)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(score.cost.analytic_evaluations, 4)

    def test_initialization_cache_intrinsic_cost_survives_reuse(self):
        from research_experiments.common.initialization import initialize
        c = deepcopy(DEFAULT)
        c.update(population=2)
        c["initialization"]["ddim_steps"] = 2
        a, ma = initialize(self.score(), c, self.root, "quadratic_unit_fixture")
        s = self.score()
        b, mb = initialize(s, c, self.root, "quadratic_unit_fixture")
        np.testing.assert_array_equal(a, b)
        self.assertTrue(mb["reused"])
        self.assertEqual(s.cost.analytic_evaluations, 4)
        self.assertEqual(s.cost.actual_analytic_evaluations, 0)

    def test_no_ground_truth_in_search_api(self):
        import inspect
        self.assertEqual(list(inspect.signature(run_search).parameters), ["score", "initial", "cfg", "replenish", "seed"])
        source = (REPO / "research_experiments/common/search.py").read_text()
        imports = [n.module for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any(n and ("metrics" in n or "distribution" in n) for n in imports))

    def test_mean_merge_can_increase_radius_recall(self):
        x = np.array([[-.1], [.1]])
        before = evaluate_modes([[0.]], x, [.05])[0]["recall"]
        after = evaluate_modes([[0.]], cluster(x, .3, "mean")[0], [.05])[0]["recall"]
        self.assertEqual((before, after), (0., 1.))

    def test_scaled_coordinate_chain_rule(self):
        g, y, a, h = GMM([[2., 1.]], [np.eye(2)], [1.]), np.array([[.3, -.2]]), .7, 1e-5
        grad = [(g.logpdf(a * (y + np.eye(2)[j] * h))[0] - g.logpdf(a * (y - np.eye(2)[j] * h))[0]) / (2 * h) for j in range(2)]
        np.testing.assert_allclose(a * g.score(a * y)[0], grad, atol=1e-8)

    def test_exclusive_json_cannot_replace_file(self):
        p = self.root / "protocol.json"
        write_json(p, {"original": True}, exclusive=True)
        with self.assertRaises(FileExistsError):
            write_json(p, {"original": False}, exclusive=True)
        self.assertTrue(json.loads(p.read_text())["original"])

    def test_confirmation_requires_lock_before_execution(self):
        from research_experiments.run import main
        c = deepcopy(DEFAULT)
        c["phase"] = "confirmation"
        with patch("research_experiments.run.load", return_value=([dict(id="unit", label="unit", config=c)], self.root, "unit_digest")):
            with patch("research_experiments.run.source_version", return_value={"source_sha256": "unit_code"}):
                with self.assertRaisesRegex(ValueError, "requires --lock"):
                    main(["run", "--config", "unit_fixture_unused"])

    def test_confirmation_lock_and_changed_code_guard(self):
        from research_experiments.run import main
        c = deepcopy(DEFAULT)
        c["phase"] = "confirmation"
        result = ([dict(id="unit", label="unit", config=c)], self.root, "unit_digest")
        with patch("research_experiments.run.load", return_value=result):
            with patch("research_experiments.run.source_version", return_value={"source_sha256": "unit_code"}):
                with redirect_stdout(io.StringIO()):
                    main(["lock", "--config", "unit_fixture_unused"])
            path = self.root / "protocols/unit_digest.json"
            with patch("research_experiments.run.source_version", return_value={"source_sha256": "changed_code"}):
                with self.assertRaisesRegex(ValueError, "differ from lock"):
                    main(["run", "--config", "unit_fixture_unused", "--lock", str(path)])

    def test_mixed_initialization_estimates_are_separate(self):
        from research_experiments.common.runner import estimate
        c = deepcopy(DEFAULT)
        c.update(population=2, field="exact")
        c["initialization"].update(source="learned", ddim_steps=2)
        c["method"].update(levels=[0], steps=3, final_steps=0)
        e = estimate(c)
        self.assertEqual(e["no_population_change"], dict(neural_nfe=4, analytic_evaluations=6))

    def test_saved_start_prefix_reuse_without_model(self):
        from research_experiments.common.initialization import initialize
        from research_experiments.common.storage import array_hash
        p = self.root / "imported.npz"
        cloud = np.array([[1., 0.], [0., 1.], [.5, .5]])
        write_arrays(p, initial=cloud)
        write_json(p.with_suffix(".json"), dict(format="dms_saved_starts_v1", seed=0, ddim_steps=2,
            initialization_nfe=6, checkpoint_sha256="unit_checkpoint", array_hash=array_hash(cloud)))
        field = Quadratic()
        field.kind = "neural"  # Artificial oracle, not an actual neural network.
        cost = Cost()
        score = CountedScore(field, cost)
        c = deepcopy(DEFAULT)
        c.update(population=2)
        c["initialization"]["ddim_steps"] = 2
        a, meta = initialize(score, c, self.root / "cache", dict(checkpoint_sha256="unit_checkpoint"), external_start=p)
        np.testing.assert_array_equal(a, cloud[:2])
        self.assertEqual((cost.neural_nfe, cost.actual_neural_nfe), (4, 0))

    def test_saved_start_checkpoint_mismatch_rejected(self):
        from research_experiments.common.initialization import initialize
        from research_experiments.common.storage import array_hash
        p, cloud = self.root / "imported.npz", np.zeros((2, 2))
        write_arrays(p, initial=cloud)
        write_json(p.with_suffix(".json"), dict(format="dms_saved_starts_v1", seed=0, ddim_steps=2,
            initialization_nfe=4, checkpoint_sha256="other", array_hash=array_hash(cloud)))
        field = Quadratic()
        field.kind = "neural"
        c = deepcopy(DEFAULT)
        c.update(population=2)
        c["initialization"]["ddim_steps"] = 2
        with self.assertRaisesRegex(ValueError, "do not belong"):
            initialize(CountedScore(field, Cost()), c, self.root / "cache", dict(checkpoint_sha256="expected"), external_start=p)

    def test_bad_sampling_reallocation_rejected(self):
        c = deepcopy(DEFAULT)
        c["method"].update(name="sampling", budget_policy="continue")
        with self.assertRaises(ValueError):
            validate(c)

    def test_paired_analysis_uses_differences_not_particles(self):
        from research_experiments.analysis.analyze import paired_contrasts
        common = dict(phase="development", radius=.1, cluster="one")
        rows = []
        for i in range(2):
            rows += [dict(common, label="a", setting="a", pair_key=str(i), initial_hash=str(i), recall=.5),
                     dict(common, label="b", setting="b", pair_key=str(i), initial_hash=str(i), recall=.25)]
        out = paired_contrasts(rows, [("a", "b")], repeats=100)
        self.assertEqual(out[0]["mean"], .25)
        self.assertEqual(out[0]["n_runs"], 2)

    def test_resume_checks_context_without_loading_model(self):
        from research_experiments.common.runner import run_job
        from research_experiments.common.storage import fingerprint
        c = deepcopy(DEFAULT)
        job, version = dict(id="resume_fixture", label="unit", config=c), dict(source_sha256="unit_code")
        root = self.root / c["suite"] / c["phase"]
        context = dict(config=c, code_sha256="unit_code", dependencies=[], overrides={}, runtime_dependencies=None)
        with JobStore(root, job["id"]) as store:
            store.json("metadata.json", dict(context_sha256=fingerprint(context)))
            store.complete()
        self.assertEqual(run_job(job, self.root, version, {}, resume=True), "already_complete")
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            run_job(job, self.root, dict(source_sha256="changed"), {}, resume=True)

    def test_legacy_random_matches_actual_generator(self):
        text = (REPO / "src/utils/distribution.py").read_text()
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "GaussianMixture")
        env = dict(np=np, logger=logging.getLogger("legacy_unit_fixture"))
        exec(compile("from __future__ import annotations\n" + ast.get_source_segment(text, node), "<actual upstream generator>", "exec"), env)
        original = env["GaussianMixture"].random(K=2, dim=2, seed=1)
        adapted = generate(dict(family="legacy_random", dim=2, components=2), 1)
        np.testing.assert_array_equal(original.mus, adapted.means)
        np.testing.assert_array_equal(original.weights, adapted.weights)
        np.testing.assert_array_equal(original.sigmas ** 2, adapted.covariances[:, 0, 0])

    def test_no_projection_still_records_outside_box(self):
        score = self.score()
        score.field.score = lambda x, t: np.ones_like(x)
        out = run_search(score, np.array([[.99, .99]]), self.method(box=[-1., 1.], projection="none", steps=1))
        self.assertTrue(np.all(out["candidates"] > 1.))
        self.assertEqual(out["movement"][0]["outside_fraction"], 1.)
        self.assertEqual(out["movement"][0]["fraction_clipped"], 0.)

    def test_warm_ddim_uses_noisy_endpoint(self):
        from research_experiments.common.initialization import ddim
        s1, s2 = self.score(), self.score()
        warm, clean = ddim(s1, 2, 1, 0, stop_t=2), ddim(s2, 2, 1, 0, stop_t=0)
        self.assertEqual(s1.cost.analytic_evaluations, 2)
        self.assertFalse(np.allclose(warm, clean))

    def test_disabled_cache_executes_all_artificial_calls(self):
        from research_experiments.common.initialization import initialize
        c = deepcopy(DEFAULT)
        c.update(population=2)
        c["initialization"].update(ddim_steps=2, cache=False)
        s = self.score()
        initialize(s, c, self.root / "never_created", "unit_no_cache")
        initialize(s, c, self.root / "never_created", "unit_no_cache")
        self.assertEqual(s.cost.actual_analytic_evaluations, 8)
        self.assertFalse((self.root / "never_created").exists())


if __name__ == "__main__":
    unittest.main()
