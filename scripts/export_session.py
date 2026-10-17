"""Export a Claude Code session transcript to readable JSON and Markdown.

Keeps user messages, Claude's replies verbatim and a one-line summary per tool
call (no tool output). Real Hugging Face tokens (hf_...) are redacted.

Usage:
    python3 scripts/export_session.py [SESSION_ID] [--out-prefix claude-session-YYYYMMDD]
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path.home() / '.claude/projects/-data-catastrophic-forgetting--SDFT-Experiment'
HF_TOKEN = re.compile(r'hf_(?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*[a-z])[A-Za-z0-9]{30,}')
REMINDER = re.compile(r'<system-reminder>.*?</system-reminder>', re.S)


def clean(text):
    return HF_TOKEN.sub('<REDACTED>', REMINDER.sub('', text)).strip()


def tool_summary(name, inp):
    if name == 'AskUserQuestion':
        qs = '; '.join(q.get('question', '') for q in inp.get('questions', []))
        return f'Hỏi người dùng: {qs}'
    return str(inp.get('description') or inp.get('file_path') or inp.get('query') or inp.get('url') or '')


def load(path):
    messages, seen = [], set()
    for line in open(path):
        d = json.loads(line)
        if d.get('type') not in ('user', 'assistant') or d.get('isMeta') or d.get('uuid') in seen:
            continue
        seen.add(d.get('uuid'))
        content, ts = d.get('message', {}).get('content'), d.get('timestamp')
        if isinstance(content, str):
            content = [{'type': 'text', 'text': content}]
        for b in content or []:
            if b.get('type') == 'text' and clean(b['text']):
                messages.append({'role': d['type'], 'timestamp': ts, 'text': clean(b['text'])})
            elif b.get('type') == 'tool_use':
                messages.append({'role': 'tool_call', 'timestamp': ts, 'tool': b['name'],
                                 'summary': clean(tool_summary(b['name'], b.get('input', {})))[:300]})
            elif b.get('type') == 'tool_result' and 'Your questions have been answered' in str(b.get('content')):
                answer = str(b['content']).split('answered:')[-1].split('. You can now')[0].strip()
                messages.append({'role': 'tool_call', 'timestamp': ts, 'tool': 'AskUserQuestion',
                                 'summary': clean(f'Người dùng chọn: {answer}')[:300]})
    return messages


def to_markdown(session_id, messages):
    lines = [f'# Phiên Claude Code — SDFT Experiment ({messages[0]["timestamp"][:10]})', '',
             f'Session `{session_id}`. Chỉ gồm tin nhắn, câu trả lời và tóm tắt lệnh công cụ (không có output); '
             'token đã được che.', '']
    pending, turn = [], 0

    def flush():
        if pending:
            lines.append(f'<details><summary>{len(pending)} lệnh công cụ</summary>\n')
            lines.extend(f'- `{t["tool"]}` — {t["summary"]}' for t in pending)
            lines.append('\n</details>\n')
            pending.clear()

    for m in messages:
        if m['role'] == 'tool_call':
            pending.append(m)
            continue
        flush()
        if m['role'] == 'user':
            turn += 1
            lines += ['---', '', f'## {turn}. Người dùng · {m["timestamp"][11:16]}', '',
                      '> ' + m['text'].replace('\n', '\n> '), '']
        else:
            lines += [f'**Claude** · {m["timestamp"][11:16]}', '', m['text'], '']
    flush()
    return '\n'.join(lines)


def parse_turns(spec):
    keep = set()
    for part in spec.split(','):
        a, _, b = part.partition('-')
        keep.update(range(int(a), int(b or a) + 1))
    return keep


def select_turns(messages, keep):
    """Keep the chosen user turns, each with only its final Claude reply and no tool calls."""
    turns, current = [], None
    for m in messages:
        if m['role'] == 'user':
            current = {'user': m, 'reply': None}
            turns.append(current)
        elif m['role'] == 'assistant' and current is not None:
            current['reply'] = m
    out = []
    for i, t in enumerate(turns, 1):
        if i in keep:
            user = dict(t['user'], text=re.sub(r'</?pasted_content[^>]*>', '', t['user']['text']).strip())
            out += [user] + ([t['reply']] if t['reply'] else [])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('session_id', nargs='?', help='default: most recently modified session')
    ap.add_argument('--out-prefix', default=None)
    ap.add_argument('--turns', help="keep only these user turns (as numbered in a full export), e.g. '8,15,17-18'; "
                                    'each keeps its final reply only, without tool calls')
    args = ap.parse_args()

    path = (PROJECT_DIR / f'{args.session_id}.jsonl' if args.session_id
            else max(PROJECT_DIR.glob('*.jsonl'), key=lambda p: p.stat().st_mtime))
    messages = load(path)
    if args.turns:
        messages = select_turns(messages, parse_turns(args.turns))
    prefix = ROOT / (args.out_prefix or f'claude-session-{messages[0]["timestamp"][:10].replace("-", "")}')
    doc = {'session_id': path.stem, 'messages': messages}
    Path(f'{prefix}.json').write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    Path(f'{prefix}.md').write_text(to_markdown(path.stem, messages))
    turns = sum(m['role'] == 'user' for m in messages)
    print(f'Wrote {prefix}.json and {prefix}.md ({turns} turns, {len(messages)} entries)')


if __name__ == '__main__':
    main()
