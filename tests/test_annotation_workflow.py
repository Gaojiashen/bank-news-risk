"""检查合并后的最终二分类标注流程。"""
import unittest
from code.annotation.split_annotations import partition
from code.text_models.train_sentiment import mild_sampling_weights

class AnnotationWorkflowTests(unittest.TestCase):
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
                                 NewsContent="text",分析依据="reason",**{"相关/不相关":"1","积极中性消极":"1"}))
        groups=partition(rows)
        memberships={}
        for name,records in groups.items():
            for row in records:memberships.setdefault(row["SourceRow"],set()).add(name)
        self.assertEqual(len(memberships),20)
        self.assertTrue(all(len(owners)==1 for owners in memberships.values()))
        rows[0]["积极中性消极"]="0"
        with self.assertRaises(ValueError):partition(rows)

    def test_sampling_is_mild_inverse_square_root_not_equal_class_sampling(self):
        weights=mild_sampling_weights([0,0,0,0,1])
        self.assertAlmostEqual(weights[4]/weights[0],2.)
        self.assertAlmostEqual(sum(weights),1.)

if __name__=="__main__":unittest.main()
