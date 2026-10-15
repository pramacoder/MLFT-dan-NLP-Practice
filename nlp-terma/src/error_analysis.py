import os
import json
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict
from .spans import extract_labeled_spans, normalize_bio
from .config import EXPERIMENTS_DIR

NEGATIONS_AND_INTENSIFIERS = {"tidak", "kurang", "belum", "bukan", "tanpa", "tak", "sangat", "banget", "sekali", "terlalu", "cukup", "lumayan", "agak"}

def categorize_errors(
    tokens_list: List[List[str]],
    y_true_list: List[List[str]],
    y_pred_list: List[List[str]]
) -> Dict[str, Any]:
    """
    Categorizes errors into 6 structured categories (NLP-T11):
    1. missed_aspect: Aspect in True, completely missing in Pred
    2. missed_opinion: Sentiment in True, completely missing in Pred
    3. aspect_sentiment_confusion: Entity type confusion at overlapping positions
    4. false_span: Span in Pred, completely absent in True
    5. boundary_error: Overlapping span of same type with differing start/end
    6. negation_modifier_error: Boundary or span error involving negations/intensifiers
    """
    error_counts = {
        "missed_aspect": 0,
        "missed_opinion": 0,
        "aspect_sentiment_confusion": 0,
        "false_span": 0,
        "boundary_error": 0,
        "negation_modifier_error": 0
    }

    examples_per_category = defaultdict(list)

    for sent_idx, (tokens, y_t, y_p) in enumerate(zip(tokens_list, y_true_list, y_pred_list)):
        true_spans = extract_labeled_spans(y_t)
        pred_spans = extract_labeled_spans(y_p)

        # Matched spans
        matched = true_spans.intersection(pred_spans)
        unmatched_true = list(true_spans - matched)
        unmatched_pred = list(pred_spans - matched)

        handled_true = set()
        handled_pred = set()

        # Check boundary errors and confusion between overlapping spans
        for t_ent, t_s, t_e in unmatched_true:
            for p_ent, p_s, p_e in unmatched_pred:
                # Check token overlap
                if max(t_s, p_s) <= min(t_e, p_e):
                    # There is overlap!
                    overlap_tokens = tokens[max(t_s, p_s):min(t_e, p_e) + 1]
                    has_marker = any(tok.lower() in NEGATIONS_AND_INTENSIFIERS for tok in tokens[min(t_s, p_s):max(t_e, p_e) + 1])

                    if t_ent != p_ent:
                        error_counts["aspect_sentiment_confusion"] += 1
                        examples_per_category["aspect_sentiment_confusion"].append({
                            "sent_idx": sent_idx,
                            "sentence": " ".join(tokens),
                            "true_span": f"{t_ent} ('{' '.join(tokens[t_s:t_e+1])}')",
                            "pred_span": f"{p_ent} ('{' '.join(tokens[p_s:p_e+1])}')"
                        })
                    else:
                        # Same entity, boundary error
                        if has_marker:
                            error_counts["negation_modifier_error"] += 1
                            examples_per_category["negation_modifier_error"].append({
                                "sent_idx": sent_idx,
                                "sentence": " ".join(tokens),
                                "true_span": f"{t_ent} ('{' '.join(tokens[t_s:t_e+1])}')",
                                "pred_span": f"{p_ent} ('{' '.join(tokens[p_s:p_e+1])}')",
                                "note": "Melibatkan penanda negasi / penguat opini"
                            })
                        else:
                            error_counts["boundary_error"] += 1
                            examples_per_category["boundary_error"].append({
                                "sent_idx": sent_idx,
                                "sentence": " ".join(tokens),
                                "true_span": f"{t_ent} ('{' '.join(tokens[t_s:t_e+1])}')",
                                "pred_span": f"{p_ent} ('{' '.join(tokens[p_s:p_e+1])}')"
                            })

                    handled_true.add((t_ent, t_s, t_e))
                    handled_pred.add((p_ent, p_s, p_e))

        # Check completely missed spans
        for t_ent, t_s, t_e in unmatched_true:
            if (t_ent, t_s, t_e) not in handled_true:
                span_text = " ".join(tokens[t_s:t_e+1])
                if t_ent == "ASPECT":
                    error_counts["missed_aspect"] += 1
                    examples_per_category["missed_aspect"].append({
                        "sent_idx": sent_idx,
                        "sentence": " ".join(tokens),
                        "missed_span": f"ASPECT ('{span_text}')"
                    })
                elif t_ent == "SENTIMENT":
                    error_counts["missed_opinion"] += 1
                    examples_per_category["missed_opinion"].append({
                        "sent_idx": sent_idx,
                        "sentence": " ".join(tokens),
                        "missed_span": f"SENTIMENT ('{span_text}')"
                    })

        # Check completely false spans
        for p_ent, p_s, p_e in unmatched_pred:
            if (p_ent, p_s, p_e) not in handled_pred:
                span_text = " ".join(tokens[p_s:p_e+1])
                error_counts["false_span"] += 1
                examples_per_category["false_span"].append({
                    "sent_idx": sent_idx,
                    "sentence": " ".join(tokens),
                    "false_span": f"{p_ent} ('{span_text}')"
                })

    return {
        "error_counts": error_counts,
        "sample_examples": {k: v[:5] for k, v in examples_per_category.items()}
    }

def analyze_crf_test_errors():
    pred_path = os.path.join(EXPERIMENTS_DIR, "crf_test_predictions.json")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predictions not found at {pred_path}. Run train_crf.py first.")

    with open(pred_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    tokens_list = [r["tokens"] for r in records]
    y_true_list = [r["y_true"] for r in records]
    y_pred_list = [r["y_pred"] for r in records]

    results = categorize_errors(tokens_list, y_true_list, y_pred_list)

    output_path = os.path.join(EXPERIMENTS_DIR, "crf_error_analysis.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=== [NLP-T11] Ringkasan Analisis Kesalahan Model CRF (Test Set) ===")
    for k, v in results["error_counts"].items():
        print(f"  - {k:<30}: {v} kasus")
    print(f"\nHasil lengkap tersimpan di: {output_path}")

if __name__ == "__main__":
    analyze_crf_test_errors()
