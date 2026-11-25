"""Optional real GLiREL execution; not imported by offline demo."""
import threading
from pathlib import Path
from core import validate_example

COMMIT = '1f485a268c2144cdc0221bcf0b24152f93c0a155'
REVISION = '02e48d6f3fa5fe88953500abc4a37a5428b77af8'
LOCK = threading.RLock()


def load_model(checkpoint, device):
    import torch
    from glirel import GLiREL
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


def summarize(tensor):
    return {'shape':list(tensor.shape), 'first_values':tensor.detach().float().flatten()[:8].cpu().tolist()}


def predict(model, example):
    import torch
    validate_example(example)
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
    def score_hook(module, args, output):
        trace['scorer_pair_input'], trace['scorer_label_input'] = summarize(args[0]), summarize(args[1])
        trace['logits'] = summarize(output)
    ner = [[a,b,k,' '.join(example['tokens'][a:b+1])] for a,b,k in example['entities']]
    with LOCK:
        try:
            embedding = model.token_rep_layer.bert_layer
            handles.append(embedding.model.register_forward_hook(encoder_hook, with_kwargs=True))
            handles.append(model.token_rep_layer.register_forward_hook(pooling_hook))
            for key, module in [('text_after_rnn',model.rnn),('pair_projection',model.span_rep_layer),
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
