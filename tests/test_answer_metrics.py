import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'eval'))
from answer_metrics import binary_answer_metrics


class AnswerMetricTests(unittest.TestCase):
    def test_threshold_shift_can_destroy_accuracy_without_destroying_auc(self):
        labels=torch.tensor([0,0,1,1])
        scores=torch.tensor([-2.,-1.,1.,2.])
        original=binary_answer_metrics(scores,labels,(scores>0).long())
        shifted=binary_answer_metrics(scores+10,labels,(scores+10>0).long())
        self.assertEqual(original['yes_no_roc_auc'],1)
        self.assertEqual(shifted['yes_no_roc_auc'],1)
        self.assertEqual(original['balanced_accuracy'],1)
        self.assertEqual(shifted['balanced_accuracy'],0.5)

    def test_ties_receive_half_credit(self):
        result=binary_answer_metrics(torch.zeros(2),torch.tensor([0,1]),torch.tensor([0,0]))
        self.assertEqual(result['yes_no_roc_auc'],0.5)

    def test_both_labels_required(self):
        with self.assertRaises(ValueError):
            binary_answer_metrics(torch.zeros(2),torch.zeros(2),torch.zeros(2))


if __name__=='__main__':
    unittest.main()
