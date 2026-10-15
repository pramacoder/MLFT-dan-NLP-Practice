import re
from typing import List, Dict, Any

# Negation and Intensifier lists derived strictly from Train EDA (NLP-T3-04)
NEGATIONS = {"tidak", "kurang", "belum", "bukan", "tanpa", "tak"}
INTENSIFIERS = {"sangat", "banget", "sekali", "terlalu", "cukup", "lumayan", "agak"}

def word_shape(word: str) -> str:
    """
    Computes per-character word shape representation (e.g. 'Jokowi' -> 'Xxxxx', 'AC' -> 'XX', 'kamar' -> 'xxxxx', '200rb' -> 'dddxx'):
    """
    shape = []
    for char in word:
        if char.isupper():
            shape.append("X")
        elif char.islower():
            shape.append("x")
        elif char.isdigit():
            shape.append("d")
        else:
            shape.append("-")
    return "".join(shape)

def token2features(sent_tokens: List[str], i: int) -> Dict[str, Any]:
    """
    Extracts lexical, morphological, orthographic, shape, context (+-1),
    and domain-specific negation/intensifier features for token i.
    """
    word = sent_tokens[i]
    word_lower = word.lower()

    features = {
        "bias": 1.0,
        "word.lower()": word_lower,
        "word[-3:]": word_lower[-3:] if len(word_lower) >= 3 else word_lower,
        "word[-2:]": word_lower[-2:] if len(word_lower) >= 2 else word_lower,
        "word[:3]": word_lower[:3] if len(word_lower) >= 3 else word_lower,
        "word[:2]": word_lower[:2] if len(word_lower) >= 2 else word_lower,
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word.shape": word_shape(word),
        "is_negation": word_lower in NEGATIONS,
        "is_intensifier": word_lower in INTENSIFIERS,
    }

    # Previous token context (-1)
    if i > 0:
        prev_word = sent_tokens[i - 1]
        prev_lower = prev_word.lower()
        features.update({
            "-1:word.lower()": prev_lower,
            "-1:word.istitle()": prev_word.istitle(),
            "-1:word.isupper()": prev_word.isupper(),
            "-1:word.shape": word_shape(prev_word),
            "-1:is_negation": prev_lower in NEGATIONS,
            "-1:is_intensifier": prev_lower in INTENSIFIERS,
        })
    else:
        features["BOS"] = True

    # Next token context (+1)
    if i < len(sent_tokens) - 1:
        next_word = sent_tokens[i + 1]
        next_lower = next_word.lower()
        features.update({
            "+1:word.lower()": next_lower,
            "+1:word.istitle()": next_word.istitle(),
            "+1:word.isupper()": next_word.isupper(),
            "+1:word.shape": word_shape(next_word),
        })
    else:
        features["EOS"] = True

    return features

def sentence2features(sent_tokens: List[str]) -> List[Dict[str, Any]]:
    return [token2features(sent_tokens, i) for i in range(len(sent_tokens))]

def sentence2labels(sent_labels: List[str]) -> List[str]:
    return sent_labels

if __name__ == "__main__":
    test_sent = ["kamar", "mandi", "sangat", "bersih", "."]
    feats = sentence2features(test_sent)
    print(f"Features for sentence: {test_sent}\n")
    for tok, f in zip(test_sent, feats):
        print(f"--- Token: '{tok}' ---")
        print(f)
