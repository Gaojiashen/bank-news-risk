"""检查文章与银行组合唯一性、分类前池化和经校验的续跑。"""
import csv
import contextlib
import importlib
import importlib.util
import json
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


class DailyEmbeddingTests(unittest.TestCase):
    def module(self):
        name = "code.news_features.extract_news_embeddings"
        self.assertIsNotNone(importlib.util.find_spec(name), "Daily semantic extraction is not implemented")
        return importlib.import_module(name)

    def fixture(self, root):
        raw, labels = root / "raw/2024/01/20240105.csv", root / "labels/2024/01/20240105.csv"
        raw.parent.mkdir(parents=True); labels.parent.mkdir(parents=True)
        with raw.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["NewsID", "DeclareDate", "Title", "NewsContent", "NewsSource"])
            w.writerow(["n1", "2024-01-05", "银行新闻", "中国银行与工商银行公布消息。", "source"])
            w.writerow(["n2", "2024-01-05", "", "中国银行公布第二则消息。", "source"])
        with labels.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f); w.writerow(["SourceFile", "SourceRow", "BankName", "积极消极"])
            w.writerows([["2024/01/20240105.csv", "2", "中国银行", "积极"],
                         ["2024/01/20240105.csv", "2", "中国银行", "消极"],
                         ["2024/01/20240105.csv", "2", "工商银行", "积极"],
                         ["2024/01/20240105.csv", "3", "中国银行", "积极"]])
        return raw, labels

    def test_duplicate_annotation_snippets_produce_one_input_per_article_bank(self):
        m = self.module()
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            raw, labels = self.fixture(Path(directory))
            records = m.article_bank_inputs(raw, labels, "2024/01/20240105.csv", ["中国银行", "工商银行"])
            self.assertEqual([(r["source_row"], r["bank_index"]) for r in records], [(2, 0), (2, 1), (3, 0)])
            self.assertEqual(records[0]["content"], "银行新闻\n中国银行与工商银行公布消息。")
            self.assertEqual(records[2]["content"], "中国银行公布第二则消息。")
            with labels.open("a", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerow(["../outside.csv", "2", "中国银行", "积极"])
            with self.assertRaises(ValueError):
                m.article_bank_inputs(raw, labels, "2024/01/20240105.csv", ["中国银行", "工商银行"])

    def test_each_article_has_equal_daily_weight_and_empty_banks_have_zero(self):
        m = self.module()
        means, counts = m.daily_means(np.array([[1., 3.], [9., 9.], [3., 5.]], dtype=np.float32),
                                     np.array([0, 1, 0]), 3)
        np.testing.assert_array_equal(means, [[2., 4.], [9., 9.], [0., 0.]])
        np.testing.assert_array_equal(counts, [2, 1, 0])

    def test_extracts_target_tokens_before_classifier_and_ignores_padding(self):
        m = self.module()
        from transformers import BertConfig, BertModel
        torch.manual_seed(5)
        encoder = BertModel(BertConfig(vocab_size=20, hidden_size=8, num_hidden_layers=1,
                                       num_attention_heads=2, intermediate_size=16)).eval()
        features = [dict(input_ids=[2, 5, 6, 7, 3], attention_mask=[1]*5, token_type_ids=[0, 0, 2, 2, 1]),
                    dict(input_ids=[2, 8, 3], attention_mask=[1]*3, token_type_ids=[0, 3, 1])]
        vectors = m.encode_features(encoder, features, torch.device("cpu"), batch_size=2, pad_token_id=0)
        with torch.inference_mode():
            out = encoder(input_ids=torch.tensor([[2, 5, 6, 7, 3]]), attention_mask=torch.ones(1, 5, dtype=torch.long),
                          token_type_ids=torch.tensor([[0, 0, 1, 1, 1]])).last_hidden_state
            expected = out[0, 2:4].mean(0).numpy()
        np.testing.assert_allclose(vectors[0], expected, atol=1e-6)
        self.assertEqual(vectors.shape, (2, 8))
        self.assertTrue(np.isfinite(vectors).all())

    def test_daily_artifact_can_resume_but_changed_input_or_wrong_identity_cannot(self):
        m = self.module()
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            output = Path(directory) / "20240105.npz"
            records = [dict(source_file="2024/01/20240105.csv", source_row=2, bank="中国银行", bank_index=0)]
            features = [dict(truncated=False, pooling_fallback=False, orig_tokens=10, visible_mention_tokens=4)]
            meta = dict(day="2024-01-05", source_sha256="raw-hash", labels_sha256="label-hash", profile_sha256="profile")
            m.save_day(output, records, features, np.ones((1, 8), dtype=np.float32), ["中国银行", "工商银行"], meta)
            report = m.verify_day(output, records, ["中国银行", "工商银行"], meta)
            self.assertEqual(report["article_bank_pairs"], 1)
            changed = {**meta, "source_sha256": "changed"}
            with self.assertRaises(ValueError):
                m.verify_day(output, records, ["中国银行", "工商银行"], changed)
            wrong = [{**records[0], "source_row": 3}]
            with self.assertRaises(ValueError):
                m.verify_day(output, wrong, ["中国银行", "工商银行"], meta)
            with self.assertRaises(FileExistsError):
                m.save_day(output, records, features, np.ones((1, 8), dtype=np.float32), ["中国银行", "工商银行"], meta)

    def test_empty_day_cli_checks_without_writes_and_resumes_without_bert(self):
        m = self.module()
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            raw, labels = self.fixture(root)
            with raw.open("w", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerow(["NewsID", "DeclareDate", "Title", "NewsContent", "NewsSource"])
            with labels.open("w", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerow(["SourceFile", "SourceRow", "BankName", "积极消极"])
            model = root / "model"
            (model / "tokenizer").mkdir(parents=True)
            # 故意提供无法加载的权重和分词器，以检查空新闻日的行为。
            # 即使未使用--check，空新闻日也不应调用BERT或分类器。
            (model / "best_model.pt").write_bytes(b"unused checkpoint")
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer/tokenizer.json").write_text("{}", encoding="utf-8")
            np.savez(root / "banks.npz", bank_names=np.array([f"Bank{i}" for i in range(16)]),
                     bank_codes=np.array([str(i) for i in range(16)]))
            output = root / "output"
            args = ["--source", str(root / "raw"), "--labels", str(root / "labels"), "--model-dir", str(model),
                    "--bank-reference", str(root / "banks.npz"), "--output", str(output),
                    "--start-date", "2024-01-05", "--end-date", "2024-01-05", "--device", "cpu"]
            with contextlib.redirect_stdout(io.StringIO()):
                m.main([*args, "--check"])
                self.assertFalse(output.exists())
                m.main([*args, "--resume"])
                target = output / "2024/01/20240105.npz"
                modified = target.stat().st_mtime_ns
                m.main([*args, "--resume"])
            self.assertEqual(target.stat().st_mtime_ns, modified)
            with np.load(target, allow_pickle=False) as d:
                self.assertEqual(d["embeddings"].shape, (0, 768))
                self.assertFalse(d["has_news"].any())


if __name__ == "__main__":
    unittest.main()
