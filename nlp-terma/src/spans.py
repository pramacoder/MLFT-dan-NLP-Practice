from typing import List, Tuple, Dict, Set

def normalize_bio(labels: List[str]) -> List[str]:
    """
    Normalizes BIO sequence (NLP-T10-01):
    - An 'I-X' tag that does not follow a valid 'B-X' or 'I-X' is converted to 'B-X'.
    - An 'I-X' tag following a 'B-Y' or 'I-Y' (where X != Y) is converted to 'B-X'.
    """
    normalized = []
    prev_label = "O"

    for lbl in labels:
        if lbl.startswith("I-"):
            entity_type = lbl.split("-")[1]
            valid_prev = {f"B-{entity_type}", f"I-{entity_type}"}
            if prev_label not in valid_prev:
                # Illegal transition corrected to B-X
                new_lbl = f"B-{entity_type}"
                normalized.append(new_lbl)
                prev_label = new_lbl
                continue
        normalized.append(lbl)
        prev_label = lbl

    return normalized

def extract_labeled_spans(labels: List[str]) -> Set[Tuple[str, int, int]]:
    """
    Extracts spans as a set of tuples: (entity_type, start_idx, end_idx_inclusive).
    Applies normalize_bio first to guarantee clean span extraction.
    """
    clean_labels = normalize_bio(labels)
    spans = set()
    
    current_entity = None
    start_idx = None

    for i, lbl in enumerate(clean_labels):
        if lbl.startswith("B-"):
            if current_entity is not None:
                spans.add((current_entity, start_idx, i - 1))
            current_entity = lbl.split("-")[1]
            start_idx = i
        elif lbl.startswith("I-"):
            entity = lbl.split("-")[1]
            if current_entity == entity:
                pass  # Continuation
            else:
                # Should not happen after normalization, but safe fallback
                if current_entity is not None:
                    spans.add((current_entity, start_idx, i - 1))
                current_entity = entity
                start_idx = i
        else:  # 'O'
            if current_entity is not None:
                spans.add((current_entity, start_idx, i - 1))
                current_entity = None
                start_idx = None

    if current_entity is not None:
        spans.add((current_entity, start_idx, len(clean_labels) - 1))

    return spans

def reconstruct_span_texts(tokens: List[str], labels: List[str]) -> Dict[str, List[Tuple[str, int, int]]]:
    """
    Returns dict mapping entity type to list of (surface_text, start_idx, end_idx).
    """
    spans = extract_labeled_spans(labels)
    result = {"ASPECT": [], "SENTIMENT": []}
    
    # Sort by start_idx
    sorted_spans = sorted(list(spans), key=lambda x: x[1])
    for entity, s, e in sorted_spans:
        surface = " ".join(tokens[s:e+1])
        result[entity].append((surface, s, e))
    return result
