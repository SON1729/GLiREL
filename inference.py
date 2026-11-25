"""Optional real GLiREL execution; not imported by offline demo."""
import threading
import os
from pathlib import Path
from core import validate_example

COMMIT = '1f485a268c2144cdc0221bcf0b24152f93c0a155'
REVISION = '02e48d6f3fa5fe88953500abc4a37a5428b77af8'
LOCK = threading.RLock()


def load_model(checkpoint, device):
    import torch
    from glirel import GLiREL
    if device == 'cpu':
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Máy chưa nhận CUDA. Chọn cpu hoặc chạy demo trên máy có GPU.')
    if checkpoint == 'jackboyla/glirel-large-v0':
        from huggingface_hub import snapshot_download
        checkpoint = snapshot_download(checkpoint, revision=REVISION,
                                       allow_patterns=['glirel_config.json','pytorch_model.bin'])
    path = Path(checkpoint).expanduser()
    if not all((path/name).is_file() for name in ['glirel_config.json','pytorch_model.bin']):
        raise ValueError('Cần đường dẫn thư mục chứa glirel_config.json và pytorch_model.bin.')
    model = GLiREL.from_pretrained(str(path), map_location='cpu', strict=True).to(device).eval()
    model.device = device
    return model


def model_config(model):
    return {k: getattr(model.config, k, None) for k in (
        'model_name', 'hidden_size', 'subtoken_pooling', 'label_embed_strategy',
        'span_marker_mode', 'rel_mode', 'scorer', 'refine_prompt', 'refine_relation',
        'max_entity_pair_distance', 'add_entity_markers', 'coref_classifier',
        'max_len', 'fixed_relation_types', 'dropout')}


def summarize(tensor):
    return {'shape':list(tensor.shape), 'first_values':tensor.detach().float().flatten()[:8].cpu().tolist()}


def predict(model, example):
    import torch
    validate_example(example)
    if len(example['tokens']) > getattr(model.config, 'max_len', 384):
        raise ValueError('Câu vượt giới hạn token của checkpoint; rút ngắn câu để không bị cắt mất thực thể.')
    units = [x for label in example['labels'] for x in ('[REL]', label)] + ['[SEP]'] + example['tokens']
    tokenizer = model.token_rep_layer.bert_layer.tokenizer
    ids = tokenizer([units], is_split_into_words=True, padding=False, truncation=False)['input_ids'][0]
    if len(ids) > 512:
        raise ValueError('Câu và tên nhãn vượt 512 subtoken; rút ngắn câu hoặc giảm nhãn.')
    trace, handles = {}, []
    def tensor_hook(key):
        def hook(module, args, output): trace[key] = summarize(output)
        return hook
    def encoder_hook(module, args, kwargs, output):
        ids = kwargs.get('input_ids', args[0] if args else None)
        trace['encoder'] = {'hidden_state':summarize(output.last_hidden_state)}
        if ids is not None:
            trace['encoder']['input_ids'] = ids.detach().cpu().tolist()
            trace['encoder']['subtokens'] = [embedding.tokenizer.convert_ids_to_tokens(row)
                                            for row in trace['encoder']['input_ids']]
    def pooling_hook(module, args, output):
        trace['prompt_and_text_units'] = args[0]
        trace['pooled_embeddings'] = summarize(output['embeddings'])
        trace['label_axis'] = args[0][0][1:2*len(example['labels']):2]
        aligned = tokenizer(args[0], is_split_into_words=True, padding=True, truncation=False)
        if aligned['input_ids'] == trace.get('encoder', {}).get('input_ids'):
            trace['encoder']['word_ids'] = [aligned.word_ids(batch_index=i) for i in range(len(args[0]))]
    def pair_hook(module, args, kwargs, output):
        trace['pair_projection'] = summarize(output)
        indices = kwargs.get('relations_idx')
        if indices is not None:
            trace['pair_axis'] = indices.detach().cpu().tolist()[0]
    def score_hook(module, args, output):
        trace['scorer_pair_input'], trace['scorer_label_input'] = summarize(args[0]), summarize(args[1])
        trace['logits'] = summarize(output)
        if args[0].numel() <= 100_000:
            trace['scorer_pair_vectors'] = args[0][0].detach().float().cpu().tolist()
            trace['scorer_label_vectors'] = args[1][0].detach().float().cpu().tolist()
            trace['logit_matrix'] = output[0].detach().float().cpu().tolist()
    ner = [[a,b,k,' '.join(example['tokens'][a:b+1])] for a,b,k in example['entities']]
    with LOCK:
        try:
            embedding = model.token_rep_layer.bert_layer
            handles.append(embedding.model.register_forward_hook(encoder_hook, with_kwargs=True))
            handles.append(model.token_rep_layer.register_forward_hook(pooling_hook))
            handles.append(model.span_rep_layer.register_forward_hook(pair_hook, with_kwargs=True))
            for key, module in [('text_after_rnn',model.rnn),
                                ('label_projection',model.prompt_rep_layer)]:
                handles.append(module.register_forward_hook(tensor_hook(key)))
            handles.append(model.scorer.register_forward_hook(score_hook))
            with torch.inference_mode():
                candidates = model.predict_relations(example['tokens'], example['labels'], ner=ner,
                                                     threshold=0.0, top_k=-1)
        finally:
            for handle in handles: handle.remove()
    for p in candidates:p['score'] = float(p['score'])
    return dict(example=example, ner=ner, trace=trace, candidates_before_display_thresholds=candidates)
