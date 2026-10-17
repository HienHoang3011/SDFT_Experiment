"""Analyze completed fixed-margin controls using paired saved observations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def auc(labels,scores):
    order=np.argsort(scores,kind='stable')
    _,counts=np.unique(scores[order],return_counts=True)
    ends=np.cumsum(counts);starts=ends-counts
    ranks=np.repeat((starts+1+ends)/2,counts)
    positives=int(labels.sum());negatives=len(labels)-positives
    return float((ranks[labels[order]==1].sum()-positives*(positives+1)/2)/(positives*negatives))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('artifacts/samsum/fixed-margin-causal-128step-20261003'))
    args=parser.parse_args();root=args.root;sources=[]
    def read(path):
        data=path.read_bytes();sources.append({'path':str(path.resolve()),'sha256':hashlib.sha256(data).hexdigest()})
        return json.loads(data)
    def lines(path):
        data=path.read_bytes();sources.append({'path':str(path.resolve()),'sha256':hashlib.sha256(data).hexdigest()})
        return [json.loads(line) for line in data.splitlines() if line.strip()]
    manifest=read(root/'run_manifest.json');status=read(root/'status.json')
    assert status['phase']=='complete'
    assert (root/'logs/exit-status.txt').read_text().strip()=='0'
    arms=manifest['arms'];steps=manifest['steps']
    trajectory=read(root/'trajectories.json')
    final={a:next(r for r in trajectory if r['arm']==a and r['step']==steps) for a in arms}
    initial={a:next(r for r in trajectory if r['arm']==a and r['step']==0) for a in arms}
    train={a:lines(root/f'{a}.train.jsonl') for a in arms}
    rouge={a:read(root/f'{a}.rouge.json') for a in arms}
    choices={a:{fmt:read(root/f'{a}.step{steps}.boolq-{fmt}.json') for fmt in ('chat','raw')} for a in arms}
    labels=np.asarray(choices['ce']['chat']['labels'])
    scores={a:{fmt:np.diff(np.asarray(p['choice_logprobs'],dtype=np.float32),axis=1)[:,0] for fmt,p in choices[a].items()} for a in arms}
    for a in arms:
        assert [r['step'] for r in train[a]]==list(range(1,steps+1))
        assert [r['assistant_tokens'] for r in train[a]]==[r['assistant_tokens'] for r in train['ce']]
        assert abs(initial[a]['samsum']['model_target_nll_nats']-initial['ce']['samsum']['model_target_nll_nats'])<1e-6
        for fmt in ('chat','raw'):
            assert choices[a][fmt]['labels']==labels.tolist()
            calculated=float(((scores[a][fmt]>0)==labels).mean())
            assert abs(calculated-final[a]['boolq'][fmt]['accuracy'])<1e-6
            assert abs(auc(labels,scores[a][fmt])-final[a]['boolq'][fmt]['roc_auc'])<1e-6
        assert [r['test_index'] for r in rouge[a]['predictions']]==[r['test_index'] for r in rouge['ce']['predictions']]
        assert [(r['example_id'],r['tokens']) for r in final[a]['samsum_examples']]==[(r['example_id'],r['tokens']) for r in final['ce']['samsum_examples']]
    rng=np.random.default_rng(20261003)
    def bootstrap(values,weights=None):
        values=np.asarray(values);idx=rng.integers(len(values),size=(5000,len(values)))
        if weights is None:point=float(values.mean());draws=values[idx].mean(axis=1)
        else:
            weights=np.asarray(weights);point=float(values.sum()/weights.sum())
            draws=values[idx].sum(axis=1)/weights[idx].sum(axis=1)
        return {'delta':point,'ci95':np.quantile(draws,[.025,.975]).tolist()}
    pairs=[('steer','ce'),('ce_half','ce'),('steer_floor','steer'),('ce_anchor','ce'),('steer_anchor','steer')]
    paired={}
    for a,b in pairs:
        nll=np.array([r['target_nll_sum'] for r in final[a]['samsum_examples']])-np.array([r['target_nll_sum'] for r in final[b]['samsum_examples']])
        counts=np.array([r['tokens'] for r in final[a]['samsum_examples']])
        quality=np.array([r['rougeL'] for r in rouge[a]['predictions']])-np.array([r['rougeL'] for r in rouge[b]['predictions']])
        paired[a+' minus '+b]={'rouge_delta_pp':bootstrap(quality*100),
            'dev_nll_delta_nats':bootstrap(nll,counts),
            'boolq_chat_delta_pp':bootstrap(100*(((scores[a]['chat']>0)==labels).astype(float)-((scores[b]['chat']>0)==labels).astype(float))),
            'boolq_raw_delta_pp':bootstrap(100*(((scores[a]['raw']>0)==labels).astype(float)-((scores[b]['raw']>0)==labels).astype(float)))}
    auc_differences={}
    for a,b in [('steer','ce'),('steer_floor','steer'),('steer_anchor','steer')]:
        draws=[]
        for _ in range(2000):
            idx=rng.integers(len(labels),size=len(labels))
            draws.append(auc(labels[idx],scores[a]['chat'][idx])-auc(labels[idx],scores[b]['chat'][idx]))
        auc_differences[a+' minus '+b]={'delta':auc(labels,scores[a]['chat'])-auc(labels,scores[b]['chat']),
            'ci95':np.quantile(draws,[.025,.975]).tolist()}
    # Exploratory readout diagnostic; no model weights or benchmark scores change.
    split_rng=np.random.default_rng(42);cal=[]
    for gold in (0,1):
        indices=np.where(labels==gold)[0];indices=split_rng.permutation(indices)
        cal.extend(indices[:int(.2*len(indices))].tolist())
    cal=np.asarray(sorted(cal));hold=np.setdiff1d(np.arange(len(labels)),cal)
    calibration={}
    for a in arms:
        values=scores[a]['chat'][cal];unique=np.unique(values)
        thresholds=np.concatenate(([-np.inf],(unique[:-1]+unique[1:])/2,[np.inf]))
        accuracy=np.array([((values>t)==labels[cal]).mean() for t in thresholds])
        tied=np.where(accuracy==accuracy.max())[0]
        best=tied[np.argmin(np.abs(thresholds[tied]))];threshold=float(thresholds[best])
        before=(scores[a]['chat'][hold]>0);after=(scores[a]['chat'][hold]>threshold)
        gain=bootstrap(100*((after==labels[hold]).astype(float)-(before==labels[hold]).astype(float)))
        calibration[a]={'threshold':threshold,'calibration_examples':len(cal),'heldout_examples':len(hold),
            'heldout_default_accuracy_percent':float((before==labels[hold]).mean()*100),
            'heldout_shifted_accuracy_percent':float((after==labels[hold]).mean()*100),
            'heldout_shifted_yes_percent':float(after.mean()*100),'heldout_auc':auc(labels[hold],scores[a]['chat'][hold]),
            'gain_pp':gain}
    train_summary={}
    for a in arms:
        rows=train[a]
        motion={str(r['step']):r['actual_adapter_update_l2'] for r in rows if r['actual_adapter_update_l2'] is not None}
        train_summary[a]={'mean_grad_norm':float(np.mean([r['grad_norm_before_clip'] for r in rows])),
            'steps_clipped':int(sum(r['grad_norm_before_clip']>1 for r in rows)),
            'motion_l2_at_probes':motion,'last':rows[-1],
            'sum_train_step_seconds':sum(r['elapsed_seconds'] for r in rows)}
    report=['# Phân tích sau thí nghiệm: vì sao Steer kém baseline','',
        'Đủ sáu nhóm × 128 bước, exit code 0. Code tính lại điểm từ từng mẫu và kiểm tra cùng token count/thứ tự bước, cùng phân phối khởi đầu, cùng dev/test/BoolQ labels. Margin cố định log(2.5); không chọn lại margin.','',
        '## Kết quả cùng ngân sách','',
        '| Nhóm | SAMSum ROUGE-L % (128 test ví dụ) | BoolQ ChatML % (512) | Predict yes % | AUC | Raw accuracy % | Δ P(y) token Base ≥0.9 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a in arms:
        b=final[a]['boolq']
        report.append(f"| {a} | {100*rouge[a]['rougeL']:.2f} | {100*b['chat']['accuracy']:.2f} | {100*b['chat']['predict_yes_fraction']:.2f} | {b['chat']['roc_auc']:.4f} | {100*b['raw']['accuracy']:.2f} | {final[a]['base_confident_probability_delta']:+.4f} |")
    report+=['','## Chênh lệch paired và khoảng tin cậy','',
             '| So sánh | Δ ROUGE-L (pp), CI 95% | Δ dev NLL (nats), CI 95% | Δ BoolQ ChatML (pp), CI 95% |',
             '|---|---|---|---|']
    def fmt(d):return f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
    for name,d in paired.items():report.append(f"| {name} | {fmt(d['rouge_delta_pp'])} | {fmt(d['dev_nll_delta_nats'])} | {fmt(d['boolq_chat_delta_pp'])} |")
    report+=['','## 1. Chỉ giảm độ mạnh CE chưa tái hiện lỗi','',
        f"Mean grad norm CE = {train_summary['ce']['mean_grad_norm']:.4f}, CE×0.5 = {train_summary['ce_half']['mean_grad_norm']:.4f}, Steer = {train_summary['steer']['mean_grad_norm']:.4f}. Ở bước 128, L2 update adapter CE/CE-half/Steer lần lượt {train_summary['ce']['motion_l2_at_probes']['128']:.6f}, {train_summary['ce_half']['motion_l2_at_probes']['128']:.6f}, {train_summary['steer']['motion_l2_at_probes']['128']:.6f}.",'',
        'CE-half vẫn gần CE về ROUGE, NLL và BoolQ. AdamW/clipping làm loss scaling không đồng nghĩa update giảm đúng một nửa. Đối chứng này làm yếu giải thích đơn giản “Steer thua chỉ vì gradient yếu”; không phải đối chứng khớp chính xác norm/update trên mọi bước.','',
        '## 2. Starvation/reweighting ảnh hưởng fit token nhưng chưa đủ giải thích BoolQ','',
        'Steer giảm P(y) trên token Base đã tự tin nhiều hơn CE. Floor=0.1 giảm mức mất tự tin và cải thiện dev NLL. Chênh lệch ROUGE của floor phải đọc cùng CI ở bảng trên. Floor phục hồi BoolQ tạm thời ở bước 64 nhưng sụp lại từ bước 96; cuối cả hai vẫn luôn chọn yes. Vì vậy khẳng định “chỉ vì token dễ có gradient 0” là chưa đủ.','',
        '## 3. Lệch đáp án mạnh trong ChatML, cộng thêm giảm phân biệt','',
        'Steer đã chọn yes 100% ở bước 32, khi CE vẫn khoảng 71%. Raw BoolQ ít đổi. Tuy nhiên AUC của Steer cũng giảm: lỗi không chỉ là một offset hằng số ở ngưỡng quyết định. Floor còn có AUC cuối thấp hơn Steer.','',
        '| Chênh lệch AUC ChatML | Δ và CI 95% paired |','|---|---|']
    for name,d in auc_differences.items():report.append(f'| {name} | {fmt(d)} |')
    report+=['','## 4. Giữ Base ngăn phần lớn lỗi BoolQ nhưng có giá về thích nghi','',
        'Steer-anchor phục hồi gần mức CE trên BoolQ và token Base tự tin, trong khi CE-anchor không có lợi ích BoolQ tương ứng. Đối chứng bốn nhóm hỗ trợ tương tác giữa loss Steer và output drift. Nhưng λ=1 có độ mạnh tương đối khác khi gradient Steer yếu hơn CE; chưa tách được mọi tác động optimizer/clipping/regularization.','',
        'ROUGE của Steer-anchor giảm mạnh so với Steer và CE. Thêm KL Base chưa phải một bản sửa cân bằng tốt; nó chứng minh khả năng ngăn drift trong cấu hình đã thử, không chứng minh nên dùng nguyên λ=1 cho phương pháp cuối.','',
        '## Diagnostic bổ sung: đổi ngưỡng bằng split nội bộ','',
        'Post-hoc: 20% của cùng probe 512 mẫu dùng chọn ngưỡng accuracy, 80% còn lại dùng đánh giá. Đây là phân tích readout trên validation đã quan sát, không thay benchmark, không cập nhật mô hình, không chứng minh giữ nguyên năng lực. Ngưỡng không thay ROC AUC.','',
        '| Nhóm | Ngưỡng yes−no | Holdout mặc định % | Holdout sau chỉnh ngưỡng % |',
        '|---|---:|---:|---:|']
    for a,c in calibration.items():report.append(f"| {a} | {c['threshold']:.4f} | {c['heldout_default_accuracy_percent']:.2f} | {c['heldout_shifted_accuracy_percent']:.2f} |")
    report+=['','## Kết luận về phương pháp','',
        'Chưa thấy lỗi dấu/công thức margin. Gradient loss gốc bằng CE nhân hệ số phụ thuộc token; đạt margin thì hệ số về 0. Target lấy từ logits hiện tại rồi detach nên không bảo vệ distribution Base. Những đối chứng trên cho thấy quá trình Steer tạo drift làm giảm độ tự tin và lệch readout ChatML; token-floor chỉ chữa được một phần fit token, còn Base anchor ngăn drift rõ hơn nhưng làm giảm SAMSum. CE cũng không có Base anchor mà vẫn tốt hơn ở đây: thiếu anchor không thể là lời giải duy nhất.','',
        'Không kết luận log(2.5) sai về giá trị, không kết luận thay margin sẽ sửa được. Đây là một seed và trajectory 128 bước của LoRA. Full SFT/Steer-Full và FAPM giữ nguyên artifact hai epoch để so bối cảnh; không coi đó là causal contrast cùng ngân sách. FAPM giữ Base tốt nhưng ROUGE gần Base nên mức giữ năng lực gắn với mức thích nghi nhỏ.','',
        'Có thêm đối chứng Steer-anchor hai epoch đã hoàn tất: trên 3.270 BoolQ ChatML, 62.202% → 81.529%; SAMSum full test 44.229% → 36.846%. Đây là bằng chứng bổ sung cùng hướng, khác runtime/ngân sách so với lượt 128 bước.','',
        'Các CI là paired example bootstrap, không phản ánh biến thiên giữa training seed. P(y) nhóm Base tự tin chỉ có aggregate đã lưu nên chưa có CI theo ví dụ cho riêng nhóm này. Không suy ra quên HumanEval/IFEval/BBH từ BoolQ.','']
    (root/'ANALYSIS.md').write_text('\n'.join(report))
    payload={'validation':{'all_arms_complete':True,'same_token_counts':True,'same_initial_distribution':True,
        'same_dev_test_samples':True},'paired_comparisons':paired,'auc_differences':auc_differences,
        'exploratory_calibration':{'calibration_indices':cal.tolist(),'heldout_indices':hold.tolist(),'arms':calibration},
        'training_summary':train_summary,'sources':sources}
    (root/'analysis_statistics.json').write_text(json.dumps(payload,indent=2)+'\n')
    print((root/'ANALYSIS.md').resolve())
    print(json.dumps({'paired':paired,'auc':auc_differences,'calibration':calibration},indent=2))


if __name__=='__main__':main()
