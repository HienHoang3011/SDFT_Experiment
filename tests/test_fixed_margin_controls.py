import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'methods'))
from train_fixed_margin_controls import control_loss, selected_logits, auc


class FixedMarginControlTests(unittest.TestCase):
    def setUp(self):
        self.logits=torch.tensor([[[2.,0.,0.],[0.,1.,.2],[.1,.2,.3],[0.,0.,0.]]],requires_grad=True)
        self.labels=torch.tensor([[-100,0,-100,2]])
        self.outputs=SimpleNamespace(logits=self.logits)

    def test_ce_half_and_masking(self):
        ce,_=control_loss(self.outputs,self.labels,4,'ce')
        half,_=control_loss(self.outputs,self.labels,4,'ce_half')
        full_grad=torch.autograd.grad(ce,self.logits,retain_graph=True)[0]
        half_grad=torch.autograd.grad(half,self.logits)[0]
        torch.testing.assert_close(half_grad,.5*full_grad)
        self.assertEqual(full_grad[0,1].abs().sum().item(),0)
        self.assertEqual(full_grad[0,3].abs().sum().item(),0)

    def test_floor_zero_matches_original_exactly(self):
        original,_=control_loss(self.outputs,self.labels,2,'steer')
        floor,_=control_loss(self.outputs,self.labels,2,'steer_floor',floor=0.)
        torch.testing.assert_close(original,floor)
        torch.testing.assert_close(torch.autograd.grad(original,self.logits,retain_graph=True)[0],
                                   torch.autograd.grad(floor,self.logits)[0])

    def test_floor_restores_gradient_for_satisfied_token(self):
        original,_=control_loss(self.outputs,self.labels,2,'steer')
        floor,_=control_loss(self.outputs,self.labels,2,'steer_floor',floor=.1)
        g0=torch.autograd.grad(original,self.logits,retain_graph=True)[0]
        g1=torch.autograd.grad(floor,self.logits)[0]
        self.assertLess(g0[0,0].abs().max().item(),1e-6)
        expected=self.logits[0,0].detach().softmax(-1)
        expected[0]-=1
        torch.testing.assert_close(g1[0,0],.1*expected/2,atol=1e-6,rtol=1e-5)

    def test_anchor_uses_same_assistant_positions_and_denominator(self):
        logits,_=selected_logits(self.outputs,self.labels)
        reference=torch.tensor([[.8,.1,.1],[.2,.3,.5]]).log()
        plain,_=control_loss(self.outputs,self.labels,4,'ce')
        anchor,_=control_loss(self.outputs,self.labels,4,'ce_anchor',reference)
        expected=plain+F.kl_div(logits.log_softmax(-1),reference,log_target=True,reduction='sum')/4
        torch.testing.assert_close(anchor,expected)

    def test_auc_handles_ties(self):
        self.assertEqual(auc([0,1],[0,1]),1.)
        self.assertEqual(auc([0,1],[1,0]),0.)
        self.assertEqual(auc([0,1,0,1],[1,1,1,1]),.5)


if __name__=='__main__':unittest.main()
