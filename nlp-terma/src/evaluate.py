from typing import List, Dict, Set, Tuple, Any
from .spans import extract_labeled_spans

def compute_span_metrics(
    true_labels_list: List[List[str]],
    pred_labels_list: List[List[str]]
) -> Dict[str, Any]:
    """
    Computes exact-match span-level Precision, Recall, and F1 score:
    - Global micro-averaged across all entities
    - Per entity type (ASPECT, SENTIMENT)
    
    Adheres strictly to the specification in NLP-T10-02.
    """
    counts = {
        "GLOBAL": {"TP": 0, "FP": 0, "FN": 0},
        "ASPECT": {"TP": 0, "FP": 0, "FN": 0},
        "SENTIMENT": {"TP": 0, "FP": 0, "FN": 0}
    }

    for y_true, y_pred in zip(true_labels_list, pred_labels_list):
        true_spans = extract_labeled_spans(y_true)
        pred_spans = extract_labeled_spans(y_pred)

        # Matched spans (TP)
        matched = true_spans.intersection(pred_spans)
        
        # False positives: in pred but not in true
        fps = pred_spans - true_spans
        
        # False negatives: in true but not in pred
        fns = true_spans - pred_spans

        counts["GLOBAL"]["TP"] += len(matched)
        counts["GLOBAL"]["FP"] += len(fps)
        counts["GLOBAL"]["FN"] += len(fns)

        for ent, s, e in matched:
            if ent in counts:
                counts[ent]["TP"] += 1

        for ent, s, e in fps:
            if ent in counts:
                counts[ent]["FP"] += 1

        for ent, s, e in fns:
            if ent in counts:
                counts[ent]["FN"] += 1

    # Calculate P, R, F1
    results = {}
    for key, c in counts.items():
        tp = c["TP"]
        fp = c["FP"]
        fn = c["FN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results[key] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }

    return results

def test_unit_exact_match_example():
    """
    Unit test with lecturer's example: TP=2, FP=1, FN=1, F1=2/3
    """
    # 5 tokens
    # True has: ASPECT at 0..1, SENTIMENT at 2..2, SENTIMENT at 4..4
    y_true = [["B-ASPECT", "I-ASPECT", "B-SENTIMENT", "O", "B-SENTIMENT"]]
    
    # Pred has: ASPECT at 0..1 (TP), SENTIMENT at 2..2 (TP), ASPECT at 3..3 (FP)
    # Missing SENTIMENT at 4..4 (FN)
    y_pred = [["B-ASPECT", "I-ASPECT", "B-SENTIMENT", "B-ASPECT", "O"]]

    res = compute_span_metrics(y_true, y_pred)
    g = res["GLOBAL"]

    assert g["TP"] == 2, f"Expected TP=2, got {g['TP']}"
    assert g["FP"] == 1, f"Expected FP=1, got {g['FP']}"
    assert g["FN"] == 1, f"Expected FN=1, got {g['FN']}"
    expected_f1 = 2.0 / 3.0
    assert abs(g["F1"] - expected_f1) < 1e-6, f"Expected F1=2/3, got {g['F1']}"
    print(f"Unit test PASSED: TP={g['TP']}, FP={g['FP']}, FN={g['FN']}, Precision={g['Precision']:.4f}, Recall={g['Recall']:.4f}, F1={g['F1']:.4f} (2/3)")

if __name__ == "__main__":
    test_unit_exact_match_example()
