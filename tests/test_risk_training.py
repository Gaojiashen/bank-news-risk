"""检查日度模型的信息流、文章计数与历史窗口边界。"""
import csv
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


class DailyRiskTests(unittest.TestCase):
    def module(self, name):
        full = f"code.{name}"
        self.assertIsNotNone(importlib.util.find_spec(full), f"Missing daily implementation: {full}")
        return importlib.import_module(full)

    def test_no_direct_risk_enters_sentiment_grus_but_fusion_receives_it(self):
        models = self.module("forecasting.risk_models")
        for kind in ("attention_bias", "diffusion"):
            torch.manual_seed(7)
            model = models.DailyNewsNetworkRiskModel(kind)
            seen = {}
            def capture(name):
                def hook(module, inputs, output):
                    seen.setdefault(name, []).append(output.detach().clone())
                return hook
            model.positive_gru.register_forward_hook(capture("positive"))
            model.negative_gru.register_forward_hook(capture("negative"))
            model.fusion_gru.register_forward_hook(capture("fusion"))
            risk = torch.randn(2, 3, 4)
            empty = torch.zeros(2, 3, 4, 4)
            a = model(risk, empty, empty)
            states_a = {key: torch.stack(value) for key, value in seen.items()}
            seen.clear()
            b = model(risk + 2, empty, empty)
            self.assertTrue(torch.equal(states_a["positive"], torch.stack(seen["positive"])))
            self.assertTrue(torch.equal(states_a["negative"], torch.stack(seen["negative"])))
            self.assertFalse(torch.equal(states_a["fusion"], torch.stack(seen["fusion"])))
            self.assertEqual(a.shape, (2, 4))
            self.assertTrue(torch.isfinite(b).all())

    def test_all_models_backpropagate_without_macro_and_observed_edges_matter(self):
        models = self.module("forecasting.risk_models")
        risk = torch.randn(2, 4, 3)
        graph = torch.ones(2, 4, 3, 3)
        graph[:, :, range(3), range(3)] = 0
        for model in (models.DailyBenchmarkRiskGRU(), models.DailyNewsNetworkRiskModel("attention_bias"),
                      models.DailyNewsNetworkRiskModel("diffusion")):
            prediction = model(risk, graph, graph)
            prediction.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            if isinstance(model, models.DailyNewsNetworkRiskModel):
                self.assertFalse(torch.allclose(prediction, model(risk, graph * 0, graph * 0)))

    def test_daily_articles_count_once_and_weekend_enters_monday_not_friday(self):
        builder = self.module("news_features.build_daily_news_graphs")
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tests") as directory:
            root = Path(directory)
            fields = ["SourceFile", "SourceRow", "BankName", "积极消极"]
            for day, rows in (("20240105", [("a", "1", "A", "积极"), ("a", "1", "A", "积极"),
                                             ("a", "1", "B", "积极")]),
                              ("20240106", [("b", "1", "A", "消极"), ("b", "1", "B", "消极")]),
                              ("20240107", []), ("20240108", [])):
                path = root / "2024/01" / f"{day}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.writer(handle); writer.writerow(fields); writer.writerows(rows)
            pos, neg, report = builder.build_graphs(root, np.array(["2024-01-05", "2024-01-08"]),
                                                   np.array(["A", "B"]))
            self.assertEqual(pos[:, 0, 1].tolist(), [1., 0.])
            self.assertEqual(neg[:, 0, 1].tolist(), [0., 1.])
            self.assertTrue(np.array_equal(pos, pos.transpose(0, 2, 1)))
            (root / "2024/01/20240107.csv").unlink()
            with self.assertRaises(FileNotFoundError):
                builder.build_graphs(root, np.array(["2024-01-05", "2024-01-08"]), np.array(["A", "B"]))

    def test_window_target_alignment_and_missing_risk_not_filled(self):
        training = self.module("forecasting.train_risk")
        risk = np.repeat(np.arange(12, dtype=np.float32)[:, None], 2, axis=1)
        graph = np.zeros((12, 2, 2), dtype=np.float32)
        ends = training.valid_windows(risk, 3, 2)
        ds = training.DailyWindows(risk, graph, graph, ends, 3, 2)
        sample = ds[0]
        self.assertEqual(sample[0][:, 0].tolist(), [0., 1., 2.])
        self.assertEqual(sample[3].tolist(), [4., 4.])
        risk[3, 0] = np.nan
        ends = training.valid_windows(risk, 3, 2)
        self.assertNotIn(3, ends)
        self.assertNotIn(4, ends)
        self.assertNotIn(5, ends)
        self.assertNotIn(1, ends)

    def test_long_horizon_purges_labels_unavailable_at_next_split_origin(self):
        training = self.module("forecasting.train_risk")
        risk = np.ones((200, 2), dtype=np.float32)
        groups = training.split_windows(training.valid_windows(risk, 10, 10), 10)
        self.assertLessEqual(groups["train"][-1] + 10, groups["validation"][0])
        self.assertLessEqual(groups["validation"][-1] + 10, groups["test"][0])


if __name__ == "__main__":
    unittest.main()
