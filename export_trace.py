"""Run qualitative GLiREL inference on hand-annotated English examples."""

import argparse
import json
from pathlib import Path


def case(name, sentence, entities, labels, expected):
    return dict(name=name, tokens=sentence.split(), entities=entities,
                labels=labels, expected=expected)


CASES = [
    case("01_single_fact", "Ada works for Acme .",
         [(0, 0, "PERSON"), (3, 3, "ORG")],
         ["employer", "founder", "headquarters location"], [(0, "employer", 1)]),
    case("02_multiple_pairs", "Ada founded North Labs , which is based in Paris .",
         [(0, 0, "PERSON"), (2, 3, "ORG"), (9, 9, "LOC")],
         ["founded by", "headquarters location", "employer"],
         [(1, "founded by", 0), (1, "headquarters location", 2)]),
    case("03_negation", "Ada never worked for Acme .",
         [(0, 0, "PERSON"), (4, 4, "ORG")],
         ["employer", "founder", "headquarters location"], []),
    case("04_missing_label", "Ada works for Acme .",
         [(0, 0, "PERSON"), (3, 3, "ORG")],
         ["founder", "headquarters location"], []),
    case("05_cross_sentence",
         "North Labs opened in 2010 . It is headquartered in Paris .",
         [(0, 1, "ORG"), (6, 6, "ORG"), (10, 10, "LOC")],
         ["headquarters location", "inception", "founded by"],
         [(1, "headquarters location", 2), (0, "headquarters location", 2)]),
]


def entity_spans(example):
    tokens = example["tokens"]
    return [[start, end, kind, " ".join(tokens[start:end + 1])]
            for start, end, kind in example["entities"]]


def expected_triples(example):
    spans = entity_spans(example)
    return {(tuple(spans[h][:2]), label, tuple(spans[t][:2]))
            for h, label, t in example["expected"]}


def tensor_summary(value):
    return {
        "shape": list(value.shape),
        "first_values": value.detach().flatten()[:6].float().cpu().tolist(),
    }


def predict_with_trace(model, example, ner):
    trace = {}
    embeddings = model.token_rep_layer.bert_layer

    def record_encoder(module, args, kwargs, output):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        trace["encoder"] = {"hidden_state": tensor_summary(output.last_hidden_state)}
        if ids is not None:
            token_ids = ids.detach().cpu().tolist()
            trace["encoder"]["input_ids"] = token_ids
            trace["encoder"]["subtokens"] = [
                embeddings.tokenizer.convert_ids_to_tokens(row) for row in token_ids
            ]

    def record_pooling(module, args, output):
        trace["prompt_and_text_units"] = args[0]
        trace["pooled_embeddings"] = tensor_summary(output["embeddings"])

    def record_tensor(name, module):
        def hook(_module, _args, output):
            trace[name] = tensor_summary(output)
        return module.register_forward_hook(hook)

    def record_scores(module, args, output):
        trace["scorer_pair_input"] = tensor_summary(args[0])
        trace["scorer_label_input"] = tensor_summary(args[1])
        trace["logits"] = tensor_summary(output)

    handles = [
        embeddings.model.register_forward_hook(record_encoder, with_kwargs=True),
        model.token_rep_layer.register_forward_hook(record_pooling),
        record_tensor("contextual_text_after_rnn", model.rnn),
        record_tensor("pair_projection", model.span_rep_layer),
        record_tensor("label_projection", model.prompt_rep_layer),
        model.scorer.register_forward_hook(record_scores),
    ]
    try:
        candidates = model.predict_relations(
            example["tokens"], example["labels"], ner=ner,
            threshold=0.0, top_k=-1,
        )
    finally:
        for handle in handles:
            handle.remove()
    return candidates, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", help="Path to your GLiREL checkpoint directory")
    parser.add_argument("--show-inputs", action="store_true", help="Print inputs without loading GLiREL")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.3, 0.5, 0.7])
    parser.add_argument("--output", type=Path, default=Path("demo_predictions.json"))
    args = parser.parse_args()

    if args.show_inputs:
        for example in CASES:
            print(example["name"], example["tokens"], entity_spans(example), example["labels"])
        return
    if args.checkpoint is None:
        parser.error("pass the checkpoint directory or use --show-inputs")
    checkpoint = Path(args.checkpoint).resolve()
    for filename in ("glirel_config.json", "pytorch_model.bin"):
        if not (checkpoint / filename).is_file():
            parser.error(f"missing {checkpoint / filename}")
    if any(not 0 <= threshold <= 1 for threshold in args.thresholds):
        parser.error("thresholds must be between 0 and 1")

    import torch
    from glirel import GLiREL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiREL.from_pretrained(str(checkpoint), map_location="cpu", strict=True).to(device).eval()
    results = []
    for example in CASES:
        ner = entity_spans(example)
        gold = expected_triples(example)
        with torch.inference_mode():
            candidates, trace = predict_with_trace(model, example, ner)
        for prediction in candidates:
            prediction["score"] = float(prediction["score"])
        views = []
        for threshold in args.thresholds:
            predictions = [p for p in candidates if p["score"] > threshold]
            predicted = {
                ((p["head_pos"][0], p["head_pos"][1] - 1), p["label"],
                 (p["tail_pos"][0], p["tail_pos"][1] - 1))
                for p in predictions
            }
            views.append(dict(threshold=threshold, tp=len(predicted & gold),
                              fp=len(predicted - gold), fn=len(gold - predicted),
                              predictions=sorted(predictions, key=lambda p: -p["score"])))
            print(f"{example['name']} threshold={threshold:g}: "
                  f"TP={views[-1]['tp']} FP={views[-1]['fp']} FN={views[-1]['fn']}")
        results.append(dict(example=example, ner=ner, expected=sorted(gold),
                            trace=trace, candidates_before_display_thresholds=candidates,
                            views=views))
    config = json.loads((checkpoint / "glirel_config.json").read_text(encoding="utf-8"))
    config_fields = ("model_name", "hidden_size", "subtoken_pooling", "label_embed_strategy",
                     "refine_prompt", "refine_relation", "scorer", "rel_mode")
    args.output.write_text(json.dumps(dict(checkpoint=checkpoint.name, device=device,
                                          config={key: config.get(key) for key in config_fields},
                                          cases=results), ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
