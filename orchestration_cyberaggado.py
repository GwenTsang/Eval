#!/usr/bin/env python3
"""Orchestrateur EMOTYC — métriques par groupe sémantique sur les 4 golds XLSX."""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from emotyc_config import (
    ALL_LABELS, EMOTYC_LABEL2ID, LABEL_GROUPS, GROUP_DISPLAY_NAMES,
)

# ── Configuration ───────────────────────────────────────────────────────
XLSX_FILES = [
    ROOT / "golds/homophobie_annotations_gold_flat_updated.xlsx",
    ROOT / "golds/obésité_annotations_gold_flat_updated.xlsx",
    ROOT / "golds/religion_annotations_gold_flat_updated.xlsx",
    ROOT / "golds/racisme_annotations_gold_flat_updated.xlsx",
]
THRESHOLD  = 0.5
BATCH_SIZE = 128

GROUP_INDICES = {
    g: [EMOTYC_LABEL2ID[l] for l in labels]
    for g, labels in LABEL_GROUPS.items()
}


# ── Fonctions ───────────────────────────────────────────────────────────
def load_gold_xlsx(path):
    """Charge un xlsx gold → (sentences, gold_matrix int8)."""
    import pandas as pd

    df = pd.read_excel(path)
    sentences = df["TEXT"].astype(str).tolist()
    gold = np.zeros((len(df), 19), dtype=np.int8)
    for j, col in enumerate(ALL_LABELS):
        gold[:, j] = (pd.to_numeric(df[col], errors="coerce").fillna(0) >= 0.5).astype(np.int8)
    return sentences, gold


def format_inputs(tokenizer, sentences):
    """Formate chaque phrase avec son contexte (before/current/after)."""
    eos, n = tokenizer.eos_token, len(sentences)
    return [
        f"before:{sentences[i-1] if i > 0 else eos}{eos}"
        f"current: {sentences[i]}{eos}"
        f"after:{sentences[i+1] if i < n-1 else eos}{eos}"
        for i in range(n)
    ]


@torch.inference_mode()
def predict(tokenizer, model, device, texts):
    """Inférence par batch → matrice (N, 19) de probabilités."""
    parts = []
    for i in range(0, len(texts), BATCH_SIZE):
        enc = tokenizer(
            texts[i:i + BATCH_SIZE],
            return_tensors="pt", truncation=True,
            padding=True, max_length=512, add_special_tokens=False,
        ).to(device)
        parts.append(torch.sigmoid(model(**enc).logits).cpu().numpy())
    return np.vstack(parts)


# ── Main ────────────────────────────────────────────────────────────────
def main():
    missing = [p for p in XLSX_FILES if not p.exists()]
    if missing:
        sys.exit("Fichiers manquants :\n" + "\n".join(str(p) for p in missing))

    # Modèle (chargement unique)
    tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    model = (
        AutoModelForSequenceClassification
        .from_pretrained("TextToKids/CamemBERT-base-EmoTextToKids")
        .to(torch.device("cuda")).eval()
    )
    device = next(model.parameters()).device

    # Inférence sur chaque fichier, accumulation
    all_gold, all_pred = [], []
    for xlsx in XLSX_FILES:
        sentences, gold = load_gold_xlsx(xlsx)
        texts = format_inputs(tokenizer, sentences)
        probs = predict(tokenizer, model, device, texts)
        all_gold.append(gold)
        all_pred.append((probs >= THRESHOLD).astype(np.int8))
        print(f"  ✓ {xlsx.name} — {len(sentences)} phrases")

    gold = np.vstack(all_gold)
    pred = np.vstack(all_pred)
    print(f"\nTotal : {len(gold)} phrases, seuil = {THRESHOLD}\n")

    # Métriques par groupe sémantique
    for group, indices in GROUP_INDICES.items():
        g, p = gold[:, indices], pred[:, indices]
        print(f"── {GROUP_DISPLAY_NAMES[group]} ({', '.join(LABEL_GROUPS[group])}) ──")
        print(f"   Macro Rappel    : {recall_score(g, p, average='macro', zero_division=0):.3f}")
        print(f"   Macro Précision : {precision_score(g, p, average='macro', zero_division=0):.3f}")
        print(f"   Macro F1        : {f1_score(g, p, average='macro', zero_division=0):.3f}")
        print()


if __name__ == "__main__":
    main()
