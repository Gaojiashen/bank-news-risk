"""检查迁移后的权重能否复现整理前保存的预测。"""
import csv
from pathlib import Path
import unittest
import os
import numpy as np
import torch
from code.forecasting.risk_models import DailyBenchmarkRiskGRU, DailyNewsNetworkRiskModel
from code.forecasting.semantic_features import transform_semantics

ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless(os.environ.get("RUN_PRIVATE_INTEGRATION") == "1", "需要私有模型和结果，请在本地显式启用")
class PreservedPredictionTests(unittest.TestCase):
    def test_all_retained_risk_models_reproduce_last_saved_forecast(self):
        torch.set_num_threads(2)
        with np.load(ROOT/"data/risk/daily_delta_covar/daily_delta_covar_16_banks.npz") as d:
            dates=d["dates"].astype(str);risk=d["delta_covar"].astype(np.float32)
        with np.load(ROOT/"data/features/daily_news_cooccurrence.npz") as d:
            gd=d["dates"].astype(str);positive=d["positive"];negative=d["negative"]
        with np.load(ROOT/"data/features/daily_semantic_features.npz") as d:
            sd=d["dates"].astype(str);sem=d["features"];counts=d["article_counts"]
        for experiment,models in (("risk_graph/lookback_60/horizon_1",("benchmark","attention_bias","diffusion")),
                                  ("risk_graph_semantic/lookback_60/horizon_1/semantic_16",("benchmark","benchmark_semantic","attention_bias","diffusion"))):
            for name in models:
                folder=ROOT/"experiments"/experiment/name
                checkpoint=torch.load(folder/"best_model.pt",map_location="cpu",weights_only=True)
                with (folder/"predictions.csv").open(encoding="utf-8-sig",newline="") as handle:
                    rows=[r for r in csv.DictReader(handle) if r["split"]=="test"][-16:]
                date=rows[0]["input_end_date"];end=int(np.searchsorted(dates,date))
                past=dates[end-59:end+1]
                graph_idx=np.searchsorted(gd,past)
                dim=checkpoint.get("semantic_input_dim",0)
                model=DailyBenchmarkRiskGRU(semantic_dim=dim) if name.startswith("benchmark") else DailyNewsNetworkRiskModel(name,semantic_dim=dim)
                model.load_state_dict(checkpoint["model_state"]);model.eval()
                x=torch.from_numpy((risk[end-59:end+1]-checkpoint["risk_mean"])/checkpoint["risk_std"]).unsqueeze(0)
                semantic=None
                if dim:
                    idx=np.searchsorted(sd,past)
                    semantic=torch.from_numpy(transform_semantics(sem[idx],counts[idx],checkpoint["semantic_pca"])).unsqueeze(0)
                with torch.inference_mode():
                    prediction=model(x,torch.from_numpy(positive[graph_idx]).unsqueeze(0),
                                     torch.from_numpy(negative[graph_idx]).unsqueeze(0),semantic=semantic)
                actual=prediction.numpy()[0]*checkpoint["risk_std"]+checkpoint["risk_mean"]
                expected=np.array([float(r["predicted_delta_covar"]) for r in rows])
                np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-5,err_msg=f"{experiment}/{name}")

if __name__=="__main__":unittest.main()
