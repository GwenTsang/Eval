#!/usr/bin/env python3
"""Orchestrateur EMOTYC — charge le modèle une seule fois, préserve les frontières BCA."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from emotyc_predict import load_model, load_gold, format_input, predict_batch, compute_metrics

DEFAULT_XLSX_FILES = [
    ROOT / "golds/homophobie_annotations_gold_flat_updated.xlsx",
    ROOT / "golds/obésité_annotations_gold_flat_updated.xlsx",
    ROOT / "golds/religion_annotations_gold_flat_updated.xlsx",
    ROOT / "golds/racisme_annotations_gold_flat_updated.xlsx",
]


def selected_files(args: argparse.Namespace) -> list[Path]:
    if args.xlsx_files:
        files = [(p if p.is_absolute() else ROOT / p) for p in args.xlsx_files]
    elif args.input_dir:
        d = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
        files = sorted(p for p in d.glob("*.xlsx") if not p.name.startswith("~$"))
    else:
        files = DEFAULT_XLSX_FILES

    missing = [p for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError("\n".join(str(p) for p in missing))
    return files


def build_texts(tokenizer, sentences, use_context):
    """Construit les inputs BCA pour UN fichier (frontières naturelles)."""
    N = len(sentences)
    return [
        format_input(
            tokenizer, sentences[i],
            sentences[i - 1] if (i > 0 and use_context) else None,
            sentences[i + 1] if (i < N - 1 and use_context) else None,
            use_context,
        )
        for i in range(N)
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--xlsx-files", nargs="+", type=Path)
    src.add_argument("--input-dir", type=Path)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "orchestrated_emotyc")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.06)
    p.add_argument("--no-context", dest="use_context", action="store_false")
    p.set_defaults(use_context=True)
    args = p.parse_args()

    files = selected_files(args)
    print(f"{len(files)} fichiers sélectionnés")

    # Chargement unique du modèle
    tokenizer, model, device = load_model()

    # Inférence fichier par fichier (frontières BCA préservées), accumulation
    all_gold, all_probs = [], []
    for xlsx in files:
        sentences, gold = load_gold(str(xlsx))
        texts = build_texts(tokenizer, sentences, args.use_context)
        probs = predict_batch(tokenizer, model, device, texts, batch_size=args.batch_size)
        all_gold.append(gold)
        all_probs.append(probs)
        print(f"  ✓ {xlsx.name} — {len(sentences)} phrases")

    # Métriques globales
    gold_cat = np.vstack(all_gold)
    pred_cat = (np.vstack(all_probs) >= args.threshold).astype(int)
    per_label, global_metrics = compute_metrics(gold_cat, pred_cat)

    # Export
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_files": len(files),
        "n_samples": len(gold_cat),
        "threshold": args.threshold,
        "use_context": args.use_context,
        "global_metrics": global_metrics,
        "per_label": per_label,
    }
    out_path = out_dir / "emotyc_predictions_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n Métriques globales exportées : {out_path}")


if __name__ == "__main__":
    main()