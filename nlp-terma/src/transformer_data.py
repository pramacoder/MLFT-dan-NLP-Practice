from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer
from .config import MODEL_NAME, MAX_LEN, LABEL2ID

def create_tokenizer(model_name: str = MODEL_NAME):
    """
    Loads pretrained Fast Tokenizer for IndoBERT.
    """
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(
    tokens: List[str],
    labels: List[str],
    tokenizer: Any,
    label2id: Dict[str, int] = LABEL2ID,
    max_length: int = MAX_LEN
) -> Dict[str, Any]:
    """
    Subword tokenization and label alignment policy (NLP-T5-02):
    - First subword of a token receives the original label ID.
    - Subsequent subwords of the same token receive -100 (ignored in loss computation).
    - Special tokens ([CLS], [SEP], [PAD]) receive -100.
    """
    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_offsets_mapping=False
    )

    word_ids = tokenized_inputs.word_ids()
    aligned_labels = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            # Special token ([CLS], [SEP])
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            # First subword of this word: assign true label ID
            label_str = labels[word_idx]
            aligned_labels.append(label2id.get(label_str, 0))
        else:
            # Continuation subword: assign -100 to avoid double counting in loss
            aligned_labels.append(-100)
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

def align_predictions_back_to_words(
    word_ids: List[int],
    subword_preds: List[int],
    id2label: Dict[int, str]
) -> List[str]:
    """
    Maps subword-level model predictions back to the original word/token level (NLP-T5-03):
    Extracts the prediction corresponding to the first subword of each word.
    """
    word_preds = []
    previous_word_idx = None

    for word_idx, pred_id in zip(word_ids, subword_preds):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            word_preds.append(id2label.get(pred_id, "O"))
            previous_word_idx = word_idx

    return word_preds

def demonstrate_alignment():
    """
    Demonstrates subword alignment on 3 compound / inflected Indonesian words (NLP-T5-02):
    1. 'kamarnya' (kamar + nya)
    2. 'sangatlah' (sangat + lah)
    3. 'diperbaiki' (di + perbaiki)
    """
    print("Testing Indonesian subword alignment demonstration...")
    try:
        tokenizer = create_tokenizer()
    except Exception as e:
        print(f"Online tokenizer download failed ({e}), creating offline demonstration.")
        return

    words = ["kamarnya", "sangatlah", "diperbaiki"]
    labels = ["B-ASPECT", "B-SENTIMENT", "I-SENTIMENT"]

    encoded = tokenize_and_align_labels(words, labels, tokenizer)
    subwords = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    aligned = encoded["labels"]

    print("\n--- Subword Alignment Demonstration ---")
    print(f"{'Subword':<15} | {'Label ID':<10} | {'Interpretation'}")
    print("-" * 45)
    for sw, lbl in zip(subwords, aligned):
        interp = "Ignored (-100)" if lbl == -100 else f"First Subword ({labels[0] if lbl==LABEL2ID[labels[0]] else labels[1] if lbl==LABEL2ID[labels[1]] else labels[2]})"
        print(f"{sw:<15} | {lbl:<10} | {interp}")

if __name__ == "__main__":
    demonstrate_alignment()
