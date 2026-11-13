"""检查整理后的日度研究流程与目录。"""
import importlib
import importlib.util
from pathlib import Path
import unittest
import os

ROOT = Path(__file__).resolve().parents[1]


class ProjectLayoutTests(unittest.TestCase):
    def module(self, name):
        self.assertIsNotNone(importlib.util.find_spec(name), f"Missing organized workflow: {name}")
        return importlib.import_module(name)

    @unittest.skipUnless(os.environ.get("RUN_PRIVATE_INTEGRATION") == "1", "需要私有研究数据，请在本地显式启用")
    def test_final_text_datasets_are_separate_and_binary(self):
        m = self.module("code.common.text_training")
        relevance = m.read_split(ROOT / "data/annotation/relevance/train.csv", "train")
        sentiment = m.read_split(ROOT / "data/annotation/sentiment/train.csv", "train")
        self.assertEqual({r["rel"] for r in relevance}, {0, 1})
        self.assertTrue(all(r["senti_raw"] is None for r in relevance))
        self.assertEqual({r["senti_raw"] for r in sentiment}, {-1, 1})

    def test_public_entrypoints_import_without_running_jobs(self):
        for name in ("annotation.sample_news", "annotation.split_annotations", "annotation.validate_annotations",
                     "text_models.train_relevance", "text_models.train_sentiment", "news_features.annotate_daily_news",
                     "news_features.extract_news_embeddings", "news_features.aggregate_daily_embeddings",
                     "news_features.build_daily_news_graphs", "risk_measure.prepare_bank_prices",
                     "risk_measure.compute_daily_delta_covar", "forecasting.train_risk",
                     "analysis.plot_results", "analysis.export_label_errors"):
            self.module("code." + name)

    @unittest.skipUnless(os.environ.get("RUN_PRIVATE_INTEGRATION") == "1", "需要私有研究文件，请在本地显式启用")
    def test_final_artifacts_and_both_completed_risk_experiments_exist(self):
        self.module("code.common.paths")
        for relative in ("models/relevance_bert/best_model.pt", "models/sentiment_bert_mention_balanced/best_model.pt",
                         "data/raw/bank_prices_raw.xlsx", "data/raw/bank_aliases.xlsx",
                         "data/risk/daily_delta_covar/daily_delta_covar_16_banks.npz",
                         "experiments/risk_graph/lookback_60/horizon_1/comparison_metrics.csv",
                         "experiments/risk_graph_semantic/lookback_60/horizon_1/semantic_16/comparison_metrics.csv"):
            self.assertTrue((ROOT / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
