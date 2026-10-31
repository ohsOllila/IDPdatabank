"""Tests for fairmd.idp.saxs_gp_quality and fairmd.idp.saxs_gp_fit.

Scoring (gp_fit.dat roundtrip, CRPS math, the evaluate function, and a
torch-gated end-to-end fit) is exercised here. The heavier validation of the
scoring math (mixture CRPS against brute-force estimates, coverage, draw
mixtures) lives in Scratch/GP/SASGP/IDPs/test_*.py against the identical
algorithms during development; these tests protect the shipped interface.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fairmd.idp import saxs_gp_fit, saxs_gp_quality


class GpFitFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "gp_fit.dat"

    def test_roundtrip_preserves_columns_and_metadata(self):
        q = np.linspace(0.01, 0.4, 25)
        columns = dict(
            q_per_A=q,
            I_mean=1000.0 * np.exp(-30 * q**2),
            I_predictive_sd=np.full_like(q, 25.0),
            I_latent_sd=np.full_like(q, 20.0),
            I_lower_95=np.full_like(q, -50.0),
            I_upper_95=np.full_like(q, 2050.0),
        )
        metadata = {"noise_sigma": 25.0, "nuts_diagnostics_passed": True, "model": "sinc"}
        saxs_gp_quality.write_gp_fit(self.path, **columns, metadata=metadata)
        read, meta = saxs_gp_quality.read_gp_fit(self.path)
        for name, values in columns.items():
            np.testing.assert_allclose(read[name], values, rtol=1e-8)
        self.assertEqual(meta["noise_sigma"], 25.0)
        self.assertTrue(meta["nuts_diagnostics_passed"])
        self.assertEqual(meta["model"], "sinc")

    def test_reader_rejects_files_without_metadata(self):
        self.path.write_text("1 2 3 4 5 6\n7 8 9 10 11 12\n")
        with self.assertRaises(ValueError):
            saxs_gp_quality.read_gp_fit(self.path)

    def test_reader_rejects_wrong_column_count(self):
        self.path.write_text("# metadata: {}\n1 2 3\n4 5 6\n")
        with self.assertRaises(ValueError):
            saxs_gp_quality.read_gp_fit(self.path)


class ScoringTests(unittest.TestCase):
    def test_affine_equivalent_curve_scores_as_covered(self):
        q = np.linspace(0.01, 0.4, 30)
        simulation = np.exp(-10 * q * q)
        mean = 3 * simulation + 0.2
        score, detail = saxs_gp_quality.score_simulation(q, simulation, q, mean, np.full_like(q, 0.05))
        self.assertAlmostEqual(score["scale_sim_to_experiment"], 3.0, places=10)
        self.assertAlmostEqual(score["background"], 0.2, places=10)
        self.assertAlmostEqual(score["gp_coverage_95"], 1.0)
        self.assertAlmostEqual(score["gp_crps_sd_units"], (np.sqrt(2.0) - 1.0) / np.sqrt(np.pi), places=10)
        np.testing.assert_allclose(detail["z"], 0.0, atol=1e-10)

    def test_crps_penalizes_offset_and_width(self):
        self.assertLess(
            saxs_gp_quality.gaussian_crps(0.0, 0.0, 1.0), saxs_gp_quality.gaussian_crps(2.0, 0.0, 1.0)
        )
        self.assertLess(
            saxs_gp_quality.gaussian_crps(0.0, 0.0, 0.5), saxs_gp_quality.gaussian_crps(0.0, 0.0, 2.0)
        )


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        self.q = np.linspace(0.01, 0.4, 40)
        self.gp_mean = 1000.0 * np.exp(-30 * self.q**2)
        self.gp_sd = np.full_like(self.q, 25.0)
        experiment_dir = self.root / "Data/Experiments/saxs/SYNTH"
        experiment_dir.mkdir(parents=True)
        (experiment_dir / "saxs.dat").write_text(
            "\n".join(f"{q:.5f} {i:.4f} 1.0" for q, i in zip(self.q, self.gp_mean)) + "\n"
        )
        saxs_gp_quality.write_gp_fit(
            experiment_dir / "gp_fit.dat",
            q_per_A=self.q,
            I_mean=self.gp_mean,
            I_predictive_sd=self.gp_sd,
            I_latent_sd=self.gp_sd,
            I_lower_95=self.gp_mean - 2 * self.gp_sd,
            I_upper_95=self.gp_mean + 2 * self.gp_sd,
            metadata={"noise_sigma": 25.0, "background": 0.0, "nuts_diagnostics_passed": True},
        )

        sim_dir = self.root / "Data/Simulations/ab/cd/abcdef"
        sim_dir.mkdir(parents=True)
        sim_i = 2.0 * self.gp_mean + 5.0
        with (sim_dir / "SAXS.yaml").open("w") as handle:
            yaml.dump(
                [{"q[1/A]": float(q), "mean_I(q)[a.u.]": float(i)} for q, i in zip(self.q, sim_i)],
                handle,
            )
        self.system = {"path": "ab/cd/abcdef", "EXPERIMENT": {"saxs": {"path": ["SYNTH"]}}}

    def test_scores_and_reports_metadata(self):
        rows = saxs_gp_quality.evaluate_SAXS_quality_GP(self.system, str(self.root) + "/")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["experiment"], "SYNTH")
        self.assertAlmostEqual(row["SAXS_scale_sim_to_experiment"], 0.5, places=8)
        self.assertAlmostEqual(row["SAXS_background"], -2.5, places=6)
        self.assertAlmostEqual(row["SAXS_gp_coverage_95"], 1.0, places=10)
        self.assertAlmostEqual(
            row["SAXS_gp_crps_sd_units"], (np.sqrt(2.0) - 1.0) / np.sqrt(np.pi), places=10
        )
        self.assertTrue(row["gp_nuts_diagnostics_passed"])
        self.assertEqual(row["gp_inferred_noise_sigma"], 25.0)

    def test_evaluation_plot_written_to_simulation_folder(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("matplotlib not available")
        plot_file = self.root / "Data/Simulations/ab/cd/abcdef/saxs_gp_evaluation_SYNTH.png"
        self.assertFalse(plot_file.exists())
        rows = saxs_gp_quality.evaluate_SAXS_quality_GP(self.system, str(self.root) + "/", plot=True)
        self.assertEqual(len(rows), 1)
        self.assertTrue(plot_file.exists() and plot_file.stat().st_size > 0)

    def test_missing_fit_skips_experiment_gracefully(self):
        (self.root / "Data/Experiments/saxs/SYNTH/gp_fit.dat").unlink()
        rows = saxs_gp_quality.evaluate_SAXS_quality_GP(self.system, str(self.root) + "/")
        self.assertEqual(rows, [])

    def test_no_link_or_missing_saxs_yaml_returns_empty(self):
        self.assertEqual(
            saxs_gp_quality.evaluate_SAXS_quality_GP(
                {"path": "ab/cd/abcdef", "EXPERIMENT": {}}, str(self.root) + "/"
            ),
            [],
        )
        (self.root / "Data/Simulations/ab/cd/abcdef/SAXS.yaml").unlink()
        self.assertEqual(
            saxs_gp_quality.evaluate_SAXS_quality_GP(self.system, str(self.root) + "/"), []
        )

    def test_scoring_module_never_imports_torch(self):
        code = "import sys; import fairmd.idp.saxs_gp_quality; print('torch' in sys.modules)"
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")


class FitTests(unittest.TestCase):
    """End-to-end fitting; skipped automatically when torch is unavailable."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch not available")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fit_dir = Path(self.tmp.name) / "fit"

    def test_laplace_fit_export_and_score_roundtrip(self):
        rng = np.random.default_rng(3)
        s1 = 35.0
        r = np.linspace(0, 200, 101)
        density = (r**2) * np.exp(-(r**2) / (2 * s1**2))
        q = np.linspace(0.005, 0.3, 25)
        weights = np.full(len(r), r[1])
        operator = np.sinc(np.outer(q, r) / np.pi) * weights
        truth = operator @ density
        truth = 10000 * truth / truth[0]
        observed = truth + rng.normal(0, 50, len(q))
        report = saxs_gp_fit.fit_saxs(
            q, observed, self.fit_dir, r_max=200, points=101, inference="laplace", samples=100, make_plots=False
        )
        self.assertLess(report["noise_sigma"], 150)
        experiment_dir = Path(self.tmp.name) / "experiment"
        experiment_dir.mkdir()
        saxs_gp_fit.export_gp_fit(self.fit_dir, experiment_dir)
        fit, metadata = saxs_gp_quality.read_gp_fit(experiment_dir / "gp_fit.dat")
        self.assertEqual(metadata["source_pipeline"], "fairmd.idp.saxs_gp_fit")
        self.assertEqual(metadata["model"], "I(q) = background + integral P(r) sinc(qr) dr; sinc(x)=sin(x)/x")
        self.assertIsNotNone(metadata["noise_sigma"])
        score, _ = saxs_gp_quality.score_simulation(
            q, truth, fit["q_per_A"], fit["I_mean"], fit["I_predictive_sd"]
        )
        self.assertLess(score["gp_crps_sd_units"], 0.8)

    def test_fitting_requires_torch_message(self):
        if saxs_gp_fit.torch is not None:
            raise unittest.SkipTest("torch installed; ImportError path not reachable")
        with self.assertRaises(ImportError):
            saxs_gp_fit.sample_hyperparameters(np.arange(5.0), np.eye(5), np.ones(5), np.ones((6, 2)),
                                               np.ones(6), self.fit_dir, 1.0)


class RankingCsvTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        sys.path.insert(0, str(REPO_ROOT / "Scripts/AnalyzeDatabank"))
        import calcSAXSGPFits

        self.calc = calcSAXSGPFits

    def tearDown(self):
        sys.path.pop(0)

    def test_ranking_csv_sorted_best_first(self):
        q = np.linspace(0.01, 0.4, 30)
        mean = 1000.0 * np.exp(-30 * q**2)
        scored = []
        for system_id, perturbation in (("best", 0.0), ("mid", 0.5), ("worst", 3.0)):
            sim_intensity = (1 + perturbation) * mean
            result, _ = saxs_gp_quality.score_simulation(q, sim_intensity, q, mean, np.full_like(q, 25.0))
            scored.append(({"SYSTEM": system_id, "ID": 1}, result, None))
        path = Path(self.tmp.name) / "gp_quality_ranking.csv"
        count = self.calc.write_ranking_csv(scored, path)
        self.assertEqual(count, 3)
        import csv

        with open(path) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["system"] for row in rows], ["best", "mid", "worst"])
        self.assertEqual([int(row["rank"]) for row in rows], [1, 2, 3])
        self.assertEqual([row["simulation_id"] for row in rows], ["1", "1", "1"])


if __name__ == "__main__":
    unittest.main()
