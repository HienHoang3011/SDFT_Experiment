import sys
import unittest
import copy
import json
import tempfile
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
from update_direction_metrics import effective_factors, factor_inner, cosine, gram_rows, validate_resume
from finish_checkpoint_direction_study import choose_batches, summarize


class UpdateDirectionTests(unittest.TestCase):
    def test_effective_update_includes_cross_term(self):
        torch.manual_seed(3)
        a, b = torch.randn(3, 7, dtype=torch.float64), torch.randn(9, 3, dtype=torch.float64)
        a1, b1 = a + torch.randn_like(a)*.2, b + torch.randn_like(b)*.2
        u, v = effective_factors(a, b, a1, b1, 2.)
        dense = 2*(b1@a1-b@a)
        torch.testing.assert_close(u@v, dense)
        self.assertAlmostEqual(factor_inner(u, v), dense.square().sum().item(), places=8)

    def test_factor_pair_matches_weight_space_dot(self):
        torch.manual_seed(4)
        u, v = torch.randn(9, 6, dtype=torch.float64), torch.randn(6, 7, dtype=torch.float64)
        x, y = torch.randn_like(u), torch.randn_like(v)
        self.assertAlmostEqual(factor_inner(u,v,x,y), ((u@v)*(x@y)).sum().item(), places=8)

    def test_zero_direction_is_undefined(self):
        self.assertIsNone(cosine(0,0,1))
        self.assertEqual(cosine(2,4,1),1.)
        self.assertEqual(cosine(-2,4,1),-1.)

    def test_chunk_gram_preserves_pair_directions(self):
        a = torch.tensor([1.,-2.,3.])
        gram = gram_rows([a, -a, torch.zeros_like(a)])
        self.assertEqual(cosine(gram[0,1],gram[0,0],gram[1,1]),-1.)
        self.assertIsNone(cosine(gram[0,2],gram[0,0],gram[2,2]))

    def test_resume_requires_same_data_and_loss_scope(self):
        old = dict(models=['full_sft', 'lora'], accumulation=2, dev_indices=[4],
                   boolq_doc_ids=[9], seed=42, learning_rate=1e-5,
                   fixed_margin=.9, optimizer='AdamW8bit', batches=2, train_indices=[1,2,3,4])
        new = dict(old, batches=3, train_indices=[1,2,3,4,5,6])
        validate_resume(old,new)
        for key, value in [('train_indices',[2,1,3,4,5,6]), ('fixed_margin',1.),
                           ('models',['lora','full_sft']), ('batches',1)]:
            changed = copy.deepcopy(new);changed[key] = value
            with self.assertRaises(ValueError):validate_resume(old,changed)

    def test_budget_selects_equal_batches_within_target(self):
        batches, setup = choose_batches(4*500+200,7.5*3600,500,4,128)
        self.assertEqual(setup,200)
        predicted = (2200+setup+(batches-4)*500*1.05)/3600
        self.assertGreaterEqual(predicted,7)
        self.assertLessEqual(predicted,8)

    def test_exhausted_budget_does_not_add_padding_work(self):
        self.assertEqual(choose_batches(8*3600,7.5*3600,500,4,128)[0],4)
        self.assertEqual(choose_batches(100,7.5*3600,1,4,32)[0],32)

    def test_summary_pairs_probe_effects_by_batch(self):
        probes=('dev_nll','base_confident_probability','boolq_chat_yes_no',
                'boolq_chat_gold_no_yes_no','boolq_chat_accuracy')
        rows=[]
        for model, offset in [('full_sft',2.),('lora',1.)]:
            for i in range(3):
                trials={o:{'effective_common_update_l2':1.,
                           'probe_delta':{p: i+(offset if o=='steer' else 0) for p in probes}}
                        for o in ('ce','steer')}
                rows.append(dict(model=model,batch=i,trials=trials,
                    gradient_cosine_post_clip=.8,native_adamw_update_cosine=.7,
                    effective_common_update_cosine=.6))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'part2a').mkdir()
            (root/'part2a/results.json').write_text(json.dumps(rows))
            (root/'part2a/run_manifest.json').write_text(json.dumps(dict(models=['full_sft','lora'],batches=3)))
            result=summarize(root)
            self.assertEqual(result['methods']['full_sft']['steer_minus_ce_probe_delta']['dev_nll']['mean'],2.)
            contrast=result['matched_checkpoint_contrasts']['full_sft_minus_lora']['dev_nll']
            self.assertEqual(contrast['ci95'],[1.,1.])
            (root/'part2a/results.json').write_text(json.dumps(rows[:-1]))
            with self.assertRaises(ValueError):summarize(root)


if __name__ == '__main__':
    unittest.main()
