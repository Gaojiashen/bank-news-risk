"""检查语义时间对齐、训练期PCA和神经网络实际输入。"""
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


class DailySemanticTests(unittest.TestCase):
    def module(self, name):
        full = f"code.{name}"
        self.assertIsNotNone(importlib.util.find_spec(full), f"Missing semantic implementation: {full}")
        return importlib.import_module(full)

    def test_semantics_enter_both_graph_models_and_risk_only_benchmark_ignores_them(self):
        from code.forecasting.risk_models import DailyNewsNetworkRiskModel, DailyBenchmarkRiskGRU
        risk = torch.randn(2, 3, 4)
        graph = torch.ones(2, 3, 4, 4)
        semantic = torch.randn(2, 3, 4, 3, requires_grad=True)
        for kind in ("attention_bias", "diffusion"):
            model = DailyNewsNetworkRiskModel(kind, semantic_dim=3)
            prediction = model(risk, graph, graph, semantic=semantic)
            prediction.sum().backward()
            self.assertGreater(float(semantic.grad.abs().sum()), 0.)
            self.assertTrue(torch.isfinite(prediction).all())
            semantic.grad.zero_()
        benchmark = DailyBenchmarkRiskGRU()
        self.assertTrue(torch.equal(benchmark(risk, graph, graph, semantic=semantic),
                                    benchmark(risk, graph, graph, semantic=semantic * 0)))
        semantic_benchmark = DailyBenchmarkRiskGRU(semantic_dim=3)
        self.assertFalse(torch.allclose(semantic_benchmark(risk, graph, graph, semantic=semantic),
                                       semantic_benchmark(risk, graph, graph, semantic=semantic * 0)))

    def test_pca_is_fit_only_on_visible_training_news_and_missing_stays_zero(self):
        m = self.module("forecasting.semantic_features")
        rng = np.random.default_rng(4)
        raw = rng.normal(size=(12, 2, 6)).astype(np.float32)
        counts = np.ones((12, 2), dtype=np.int64)
        counts[1, 0] = 0
        raw[1, 0] = 0
        visible = np.arange(12) < 6
        transformed, state = m.fit_semantic_pca(raw, counts, visible, 2)
        changed = raw.copy(); changed[6:] += 10000
        transformed2, state2 = m.fit_semantic_pca(changed, counts, visible, 2)
        np.testing.assert_array_equal(state["components"], state2["components"])
        np.testing.assert_array_equal(transformed[:6], transformed2[:6])
        np.testing.assert_array_equal(transformed[1, 0], [0., 0., 0.])
        self.assertEqual(transformed.shape, (12, 2, 3))
        self.assertEqual(state["fit_bank_days"], 11)
        reconstructed = m.transform_semantics(raw, counts, state)
        np.testing.assert_allclose(reconstructed, transformed, atol=1e-6)

    def test_weekend_means_are_article_weighted_and_never_sent_to_friday(self):
        m = self.module("news_features.aggregate_daily_embeddings")
        from code.news_features.extract_news_embeddings import save_day
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            source = Path(directory)
            banks = ["A", "B"]
            for day, values in (("20240105", [2.]), ("20240106", [10., 10., 10.]),
                                ("20240107", []), ("20240108", [2.])):
                records = [dict(source_file=f"2024/01/{day}.csv", source_row=i+2, bank="A", bank_index=0) for i in range(len(values))]
                features = [dict(truncated=False, pooling_fallback=False, orig_tokens=5, visible_mention_tokens=1) for _ in values]
                vectors = np.repeat(np.asarray(values, dtype=np.float32).reshape(-1, 1), 4, axis=1)
                meta = dict(day=f"2024-01-{day[-2:]}", profile_sha256="test")
                save_day(source/f"2024/01/{day}.npz", records, features, vectors, banks, meta)
            raw, counts, report = m.aggregate_calendar_embeddings(source, np.array(["2024-01-05", "2024-01-08"]),
                                                                  np.array(banks), "2024-01-05", "test", 4)
            self.assertEqual(raw[:, 0, 0].tolist(), [2., 8.])
            self.assertEqual(counts[:, 0].tolist(), [1, 4])
            self.assertFalse(raw[:, 1].any())
            (source/"2024/01/20240107.npz").unlink()
            with self.assertRaises(FileNotFoundError):
                m.aggregate_calendar_embeddings(source, np.array(["2024-01-05", "2024-01-08"]), np.array(banks), "2024-01-05", "test", 4)

    def test_semantic_windows_do_not_include_target_day_features(self):
        from code.forecasting.train_risk import DailyWindows
        risk = np.ones((10, 2), dtype=np.float32)
        graph = np.zeros((10, 2, 2), dtype=np.float32)
        semantics = np.tile(np.arange(10, dtype=np.float32)[:, None, None], (1, 2, 3))
        dataset = DailyWindows(risk, graph, graph, np.array([3]), 3, 1, semantic=semantics)
        batch = dataset[0]
        self.assertEqual(len(batch), 5)
        self.assertEqual(batch[3][:, 0, 0].tolist(), [1., 2., 3.])


if __name__ == "__main__":
    unittest.main()
