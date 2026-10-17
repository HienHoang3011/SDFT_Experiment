"""Shared CUDA probes used by the completed gradient and training studies."""
from datetime import datetime, timezone

import torch


def log(message):
    print(datetime.now(timezone.utc).isoformat(), message, flush=True)


@torch.inference_mode()
def samsum_forward(model, probes):
    for _, ids, positions in probes:
        indices = torch.tensor(positions, device='cuda')
        out = model(input_ids=ids.to('cuda'), logits_to_keep=indices, use_cache=False)
        yield out.logits[0].float().log_softmax(-1).cpu()


@torch.inference_mode()
def boolq_forward(model, prompts, tokenizer, batch_size, choice_ids, score_newline=False):
    newline = tokenizer.encode('\n', add_special_tokens=False)
    assert len(newline) == 1
    for start in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[start:start+batch_size], padding=True,
                          add_special_tokens=False, return_tensors='pt').to('cuda')
        out = model(**batch, logits_to_keep=1, use_cache=False)
        distribution = out.logits[:, -1].float().log_softmax(-1)
        sequence_scores = None
        if score_newline:
            sequence_scores = distribution[:, choice_ids].clone()
            for j, choice in enumerate(choice_ids):
                appended = torch.full((len(batch.input_ids), 1), int(choice), device='cuda', dtype=torch.long)
                continued = {'input_ids': torch.cat((batch.input_ids, appended), -1),
                             'attention_mask': torch.cat((batch.attention_mask, torch.ones_like(appended)), -1)}
                nxt = model(**continued, logits_to_keep=1, use_cache=False)
                sequence_scores[:, j] += nxt.logits[:, -1].float().log_softmax(-1)[:, newline[0]]
        yield distribution.cpu(), sequence_scores.cpu() if sequence_scores is not None else None
