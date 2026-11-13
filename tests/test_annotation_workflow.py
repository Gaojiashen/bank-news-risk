"""检查合并后的最终二分类标注流程。"""
import unittest
import csv
from pathlib import Path
import tempfile
from code.annotation.split_annotations import partition
from code.text_models.train_sentiment import mild_sampling_weights

class AnnotationWorkflowTests(unittest.TestCase):
    def test_reader_accepts_final_binary_sentiment_column(self):
        from code.common.text_training import read_split
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/"tests") as directory:
            path=Path(directory)/"train.csv"
            with path.open("w",encoding="utf-8-sig",newline="") as handle:
                writer=csv.writer(handle)
                writer.writerow(["NewsContent","BankName","SourceFile","SourceRow","SentenceIndex","分析依据","相关/不相关","积极消极"])
                writer.writerow(["银行公布利好消息。","银行A","source.csv","2","1","正向信息","1","1"])
                writer.writerow(["银行出现风险事件。","银行A","source.csv","3","1","负向信息","1","-1"])
            rows=read_split(path,"train")
            self.assertEqual([r["senti_raw"] for r in rows],[1,-1])
            self.assertEqual([r["senti_idx"] for r in rows],[1,0])

    def test_both_selected_architectures_accept_local_config_without_download(self):
        import torch
        from transformers import BertConfig
        from code.common.text_models import BertClassifier, MentionBertClassifier
        config=BertConfig(vocab_size=20,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16)
        ids=torch.tensor([[2,5,3]])
        for cls in (BertClassifier,MentionBertClassifier):
            model=cls(2,config=config).eval()
            segments=torch.tensor([[0,2,1]]) if cls is MentionBertClassifier else torch.tensor([[0,1,1]])
            with torch.inference_mode():output=model(ids,torch.ones_like(ids),segments)
            self.assertEqual(output.shape,(1,2))
            self.assertTrue(torch.isfinite(output).all())

    def test_articles_stay_together_and_retired_labels_are_rejected(self):
        rows=[]
        for i in range(20):
            for bank in ("A","B"):
                rows.append(dict(SourceFile="source.csv",SourceRow=str(i+2),SentenceIndex="1",BankName=bank,
                                 NewsContent="text",分析依据="reason",**{"相关/不相关":"1","积极消极":"1"}))
        groups=partition(rows)
        memberships={}
        for name,records in groups.items():
            for row in records:memberships.setdefault(row["SourceRow"],set()).add(name)
        self.assertEqual(len(memberships),20)
        self.assertTrue(all(len(owners)==1 for owners in memberships.values()))
        rows[0]["积极消极"]="0"
        with self.assertRaises(ValueError):partition(rows)

    def test_sampling_is_mild_inverse_square_root_not_equal_class_sampling(self):
        weights=mild_sampling_weights([0,0,0,0,1])
        self.assertAlmostEqual(weights[4]/weights[0],2.)
        self.assertAlmostEqual(sum(weights),1.)

if __name__=="__main__":unittest.main()
