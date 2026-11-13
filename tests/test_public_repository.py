"""检查发布边界，排除私有数据、模型检查点和结果文件。"""
import os
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "没有Git元数据，跳过发布范围检查")
class PublicRepositoryTests(unittest.TestCase):
    def test_private_artifacts_are_ignored(self):
        paths = ["data/raw/news/2020/01/20200101.csv", "models/model/best_model.pt",
                 "experiments/run/metrics.json", "docs/annotation/decisions/batch_001_decisions.csv",
                 "docs/organization_report.md", ".env", "unexpected_model.safetensors"]
        for path in paths:
            result = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT)
            self.assertEqual(result.returncode, 0, f"Private artifact not excluded: {path}")

    def test_git_candidates_contain_only_public_source_material(self):
        result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                                cwd=ROOT, capture_output=True, check=True)
        paths = result.stdout.decode("utf-8").split("\0")
        bad = [p for p in paths if p and (p.startswith(("data/", "models/", "experiments/", "docs/annotation/decisions/"))
                                        or Path(p).suffix.lower() in {".csv", ".xlsx", ".npy", ".npz", ".pt", ".safetensors"})]
        self.assertEqual(bad, [], f"Private files would enter repository: {bad[:10]}")

if __name__ == "__main__":
    unittest.main()
