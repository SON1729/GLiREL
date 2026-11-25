"""Validate recorded tensor shapes/axes/logits without loading model weights."""
import hashlib
import json
import math
from pathlib import Path
from core import validate_bundle

ROOT=Path(__file__).resolve().parent

def audit_bundle(bundle):
    bundle=validate_bundle(bundle)
    cases=[]
    for case in bundle['cases']:
        e=case['example'];t=case['trace'];n=len(e['tokens']);m=len(e['labels'])
        units=t['prompt_and_text_units'][0]
        expected=[part for label in e['labels'] for part in ('[REL]',label)]+['[SEP]']+e['tokens']
        if units!=expected:raise ValueError('Prompt unit/order mismatch: '+e['name'])
        d=bundle['config']['hidden_size'];p=2*m+1
        checks={
            'prompt_units':len(units)==p+n,
            'tokenrep_shape':t['pooled_embeddings']['shape']==[1,p+n,d],
            'text_shape':(t.get('text_after_rnn') or t.get('contextual_text_after_rnn'))['shape']==[1,n,d],
            'label_shape':t['label_projection']['shape']==[1,m,d],
            'pair_scorer_shape':t['pair_projection']['shape']==t['scorer_pair_input']['shape'],
            'label_scorer_shape':t['label_projection']['shape']==t['scorer_label_input']['shape'],
        }
        entities=e['entities']
        axes=t.get('pair_axis') or [[h[:2],v[:2]] for i,h in enumerate(entities) for j,v in enumerate(entities) if i!=j]
        k=len(axes)
        checks['scorer_matrix_shape']=t['logits']['shape']==[1,k,m]
        scores={(tuple(x['head_pos']),x['label'],tuple(x['tail_pos'])):x['score'] for x in case['candidates_before_display_thresholds']}
        checks['complete_candidate_grid']=len(scores)==k*m
        errors=[]
        for flat,z in enumerate(t['logits']['first_values']):
            h,v=axes[flat//m]
            key=((h[0],h[1]+1),e['labels'][flat%m],(v[0],v[1]+1))
            errors.append(abs(1/(1+math.exp(-z))-scores[key]))
        checks['saved_logits_match_sigmoid']=bool(errors) and max(errors)<1e-6
        if not all(checks.values()):raise ValueError(f'Failed trace checks for {e["name"]}: {checks}')
        cases.append(dict(name=e['name'],tokens=n,entities=len(entities),labels=m,prompt_units=p+n,
                          pairs=k,logits_checked=len(errors),max_sigmoid_error=max(errors),checks=checks))
    return dict(source_commit='1f485a268c2144cdc0221bcf0b24152f93c0a155',
                scope='Recorded inference traces; not a model-weight or benchmark reproduction test.',
                checked_logits=sum(x['logits_checked'] for x in cases),cases=cases,
                missing_configuration=[x for x in ('span_marker_mode','max_entity_pair_distance','add_entity_markers','coref_classifier') if x not in bundle['config']])

if __name__=='__main__':
    raw=(ROOT/'data/recorded.json').read_bytes()
    report=audit_bundle(json.loads(raw));report['recorded_sha256']=hashlib.sha256(raw).hexdigest()
    out=ROOT/'data/pipeline_audit.json'
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
