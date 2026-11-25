"""Pure data/decision logic. Input spans inclusive; GLiREL API output exclusive."""
import copy
import hashlib
import html
import json
import math
from collections import defaultdict


def validate_example(example):
    tokens, entities, labels = (example[k] for k in ('tokens', 'entities', 'labels'))
    if not tokens or len(tokens) > 512 or not all(isinstance(t, str) and t for t in tokens):
        raise ValueError('Cần 1–512 token không rỗng. Demo dùng token tách bằng khoảng trắng.')
    if not 2 <= len(entities) <= 20:
        raise ValueError('Cần từ 2 đến 20 thực thể.')
    if not 1 <= len(labels) <= 30 or len(set(labels)) != len(labels):
        raise ValueError('Cần 1–30 tên nhãn khác nhau.')
    if not all(isinstance(x, str) and x.strip() for x in labels):
        raise ValueError('Tên nhãn không được rỗng.')
    spans = set()
    for start, end, kind in entities:
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < len(tokens):
            raise ValueError('Span dùng chỉ số nguyên từ 0, bao gồm cả vị trí cuối; phải nằm trong câu.')
        if (start, end) in spans:
            raise ValueError('Hai thực thể có cùng span; hãy bỏ bản ghi lặp.')
        if any(not (end < a or start > b) for a, b in spans):
            raise ValueError('Demo này không nhận các span chồng nhau; sửa span trước khi suy luận.')
        spans.add((start, end))
    for h, label, t in example.get('expected', []):
        if type(h) is not int or type(t) is not int or not (0 <= h < len(entities) and 0 <= t < len(entities)) or h == t:
            raise ValueError('Gold cần [chỉ số thực thể head, tên quan hệ, chỉ số thực thể tail].')
        if not isinstance(label, str):
            raise ValueError('Nhãn gold phải là chuỗi; có thể vắng trong danh sách ứng viên.')
    return example


def signature(example):
    fields = {k: example[k] for k in ['tokens', 'entities', 'labels']}
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()


def entity_text(example, i):
    start, end, _ = example['entities'][i]
    return ' '.join(example['tokens'][start:end+1])


def prediction_key(p):
    return (tuple(p['head_pos']), p['label'], tuple(p['tail_pos']))


def gold_keys(example):
    return {((example['entities'][h][0], example['entities'][h][1]+1), label,
             (example['entities'][t][0], example['entities'][t][1]+1))
            for h, label, t in example.get('expected', [])}


def select(candidates, threshold, top_k=1):
    """top_k is per directed pair, NOT global. Strict > matches pinned API."""
    groups = defaultdict(list)
    for p in candidates:
        if float(p['score']) > threshold:
            groups[(tuple(p['head_pos']), tuple(p['tail_pos']))].append(p)
    chosen = []
    for group in groups.values():
        group = sorted(group, key=lambda p: -float(p['score']))
        chosen.extend(group if top_k == -1 else group[:top_k])
    return sorted(chosen, key=lambda p: -float(p['score']))


def evaluate(example, predictions):
    gold, pred = gold_keys(example), {prediction_key(p) for p in predictions}
    tp, fp, fn = len(gold & pred), len(pred-gold), len(gold-pred)
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.)


def diagnosis(example, candidates, predictions, threshold):
    gold = gold_keys(example)
    predicted = {prediction_key(p) for p in predictions}
    scores = {prediction_key(p): float(p['score']) for p in candidates}
    messages = []
    for key in sorted(gold-predicted):
        h, label, t = key
        if label not in example['labels']:
            reason = 'Nhãn đúng không có trong danh sách ứng viên: mô hình không được hỏi nhãn này.'
        elif (t, label, h) in predicted:
            reason = 'Có dự đoán đảo chiều head → tail so với gold.'
        elif key in scores and scores[key] <= threshold:
            reason = f'Điểm {scores[key]:.4f} không vượt ngưỡng {threshold:.2f}.'
        elif key in scores:
            reason = 'Nhãn khác trên cùng cặp có thứ hạng cao hơn; kiểm tra top-k.'
        else:
            reason = 'Không có ứng viên này trong bản ghi; chưa đủ dữ liệu để xác định nguyên nhân.'
        messages.append(('Bỏ sót', key, reason))
    for key in sorted(predicted-gold):
        messages.append(('Dự đoán dư', key, 'Có điểm vượt ngưỡng nhưng không nằm trong gold do người dùng khai báo. Đọc lại câu và gold trước khi kết luận lỗi mô hình.'))
    return messages


def highlighted(example):
    marks = {}
    for i, (start, end, _) in enumerate(example['entities']):
        for pos in range(start, end+1): marks[pos] = i
    return '<div class="sentence">' + ' '.join(
        f'<span class="token {"entity" if i in marks else ""}"><small>{i}</small>{html.escape(t)}'
        + (f'<em>E{marks[i]}</em>' if i in marks else '') + '</span>'
        for i, t in enumerate(example['tokens'])) + '</div>'


def validate_bundle(bundle):
    if not isinstance(bundle, dict) or not isinstance(bundle.get('cases'), list) or not 1 <= len(bundle['cases']) <= 100:
        raise ValueError('JSON cần trường cases gồm 1–100 ví dụ, theo định dạng demo_predictions.json.')
    bundle = copy.deepcopy(bundle)
    for case in bundle['cases']:
        # Accept the original Kaggle exporter schema as well as the new one.
        example = case['example']
        if 'entities' not in example and 'spans' in example:
            example['entities'] = example['spans']
        if 'expected' not in example and 'gold' in example:
            example['expected'] = example['gold']
        ex = validate_example(case['example'])
        spans = {(a, b+1) for a, b, _ in ex['entities']}
        candidates = case.get('candidates_before_display_thresholds', [])
        if len(candidates) > 20000: raise ValueError('Quá nhiều ứng viên trong một ví dụ.')
        keys = set()
        for p in candidates:
            if not math.isfinite(float(p['score'])) or not 0 <= float(p['score']) <= 1:
                raise ValueError('Điểm dự đoán phải hữu hạn trong [0,1].')
            if p['label'] not in ex['labels'] or tuple(p['head_pos']) not in spans or tuple(p['tail_pos']) not in spans:
                raise ValueError('Nhãn/span trong output không khớp input. Output dùng end-exclusive.')
            if p['head_pos'] == p['tail_pos']:
                raise ValueError('Demo không nhận cặp một thực thể với chính nó.')
            key = prediction_key(p)
            if key in keys: raise ValueError('Ứng viên cặp–nhãn bị trùng.')
            keys.add(key)
    return bundle


def sigmoid(z):
    return 1/(1+math.exp(-z))
