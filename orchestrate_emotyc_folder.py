#!/usr/bin/env python3
"""Orchestrateur EMOTYC — charge le modèle une seule fois."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from emotyc_predict import (
    load_model,
    load_gold,
    format_input,
    predict_batch,
    compute_metrics,
)

DEFAULT_XLSX_FILES = [
    ROOT / "golds/homophobie_annotations_gold_flat_updated.xlsx",
    ROOT / "golds/obésité_annotations_gold_flat_updated.xlsx",
    ROOT / "golds/religion_annotations_gold_flat_updated.xlsx",
    ROOT / "golds/racisme_annotations_gold_flat_updated.xlsx",
]

OUT_DIR = ROOT / "results" / "orchestrated_emotyc"
BATCH_SIZE = 128
THRESHOLD = 0.06
USE_CONTEXT = True


def build_texts(tokenizer, sentences):
    """Construit les inputs BCA pour un fichier."""
    n = len(sentences)

    return [
        format_input(
            tokenizer,
            sentences[i],
            sentences[i - 1] if i > 0 and USE_CONTEXT else None,
            sentences[i + 1] if i < n - 1 and USE_CONTEXT else None,
            USE_CONTEXT,
        )
        for i in range(n)
    ]


def main() -> None:
    files = DEFAULT_XLSX_FILES

    missing = [p for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError("\n".join(str(p) for p in missing))

    print(f"{len(files)} fichiers sélectionnés")

    # Chargement unique du modèle
    tokenizer, model, device = load_model()

    # Inférence fichier par fichier
    all_gold, all_probs = [], []

    for xlsx in files:
        sentences, gold = load_gold(str(xlsx))

        texts = build_texts(tokenizer, sentences)

        probs = predict_batch(
            tokenizer,
            model,
            device,
            texts,
            batch_size=BATCH_SIZE,
        )

        all_gold.append(gold)
        all_probs.append(probs)

        print(f"  ✓ {xlsx.name} — {len(sentences)} phrases")

    # Métriques globales
    gold_cat = np.vstack(all_gold)
    pred_cat = (np.vstack(all_probs) >= THRESHOLD).astype(int)

    per_label, global_metrics = compute_metrics(gold_cat, pred_cat)

    # Export
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_files": len(files),
        "n_samples": len(gold_cat),
        "threshold": THRESHOLD,
        "use_context": USE_CONTEXT,
        "global_metrics": global_metrics,
        "per_label": per_label,
    }

    out_path = OUT_DIR / "emotyc_predictions_summary.json"

    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nMétriques globales exportées : {out_path}")


if __name__ == "__main__":
    main()
