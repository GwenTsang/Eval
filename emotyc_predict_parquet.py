import json, os, numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from emotyc_config import (
    ALL_LABELS, EMOTYC_LABEL2ID, LABEL_GROUPS, GROUP_DISPLAY_NAMES,
)

PARQUET_PATH = os.path.join(os.path.dirname(__file__), "golds", "TTK_test.parquet")
OUT_DIR      = os.path.join(os.path.dirname(__file__), "results")
THRESHOLD    = 0.5
BATCH_SIZE   = 910

# Indices dans le vecteur 19-d pour chaque groupe
GROUP_INDICES = {
    group: [EMOTYC_LABEL2ID[l] for l in labels]
    for group, labels in LABEL_GROUPS.items()
}

# Mapping colonne parquet → labels EMOTYC qu'elle alimente
_PARQUET_COL_TO_LABELS = {
    "Emo":        ["Emo"],
    "types":      ["Base", "Complexe"],
    "modes":      ["Comportementale", "Designee", "Montree", "Suggeree"],
    "categories": [
        "Admiration", "Autre", "Colere", "Culpabilite", "Degout",
        "Embarras", "Fierte", "Jalousie", "Joie", "Peur",
        "Surprise", "Tristesse",
    ],
}


def load_model():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    model = (
        AutoModelForSequenceClassification
        .from_pretrained("TextToKids/CamemBERT-base-EmoTextToKids")
        .to(device).eval()
    )
    return tokenizer, model, device


def format_inputs(tokenizer, df):
    eos = tokenizer.eos_token
    texts = []
    for prev, sent, nxt in zip(
        df["previous_sentence"], df["target_sentence"], df["next_sentence"]
    ):
        prev = prev if isinstance(prev, str) else eos
        nxt  = nxt  if isinstance(nxt, str)  else eos
        texts.append(f"before:{prev}{eos}current: {sent}{eos}after:{nxt}{eos}")
    return texts


@torch.inference_mode()
def predict_batch(tokenizer, model, device, texts):
    all_probs = []
    for i in range(0, len(texts), BATCH_SIZE):
        enc = tokenizer(
            texts[i:i + BATCH_SIZE],
            return_tensors="pt", truncation=True,
            padding=True, max_length=512, add_special_tokens=False,
        ).to(device)
        all_probs.append(torch.sigmoid(model(**enc).logits).cpu().numpy())
    return np.vstack(all_probs)


def build_gold(df):
    N = len(df)
    gold = np.zeros((N, 19), dtype=np.int8)

    # Emo : chaîne "0"/"1"
    gold[:, EMOTYC_LABEL2ID["Emo"]] = df["Emo"].astype(int).values

    # Colonnes array (types, modes, categories)
    for col, labels in _PARQUET_COL_TO_LABELS.items():
        if col == "Emo":
            continue
        series = df[col]
        for label in labels:
            idx = EMOTYC_LABEL2ID[label]
            gold[:, idx] = series.apply(lambda arr: int(label in arr)).values

    return gold


# ── Métriques par groupe sémantique ─────────────────────────────────────
def compute_group_metrics(gold, pred):
    results = {}
    for group, indices in GROUP_INDICES.items():
        g = gold[:, indices]
        p = pred[:, indices]
        results[group] = {
            "display_name": GROUP_DISPLAY_NAMES[group],
            "labels": LABEL_GROUPS[group],
            "macro_f1":   round(f1_score(g, p, average="macro", zero_division=0), 3),
            "precision":  round(precision_score(g, p, average="macro", zero_division=0), 3),
            "recall":     round(recall_score(g, p, average="macro", zero_division=0), 3),
        }
    return results


# ── Main ────────────────────────────────────────────────────────────────
def main():
    # 1. Gold
    df = pd.read_parquet(PARQUET_PATH)
    gold = build_gold(df)
    N = len(df)
    print(f"Gold chargé : {N} phrases, 19 labels")

    # 2. Modèle
    tokenizer, model, device = load_model()

    # 3. Inputs formatés (contexte utilisé car présent dans le parquet)
    texts = format_inputs(tokenizer, df)

    # 4. Inférence
    print(f"Inférence sur {N} phrases (batch_size={BATCH_SIZE})…")
    probs = predict_batch(tokenizer, model, device, texts)
    pred = (probs >= THRESHOLD).astype(np.int8)
    print(f"Inférence terminée — shape: {probs.shape}")

    # 5. Métriques par groupe
    group_metrics = compute_group_metrics(gold, pred)

    for group, m in group_metrics.items():
        print(f"\n── {m['display_name']} ({', '.join(m['labels'])}) ──")
        print(f"   Macro F1  : {m['macro_f1']}")
        print(f"   Precision : {m['precision']}")
        print(f"   Recall    : {m['recall']}")

    # 6. Export JSON
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {
        "source": os.path.basename(PARQUET_PATH),
        "n_samples": N,
        "threshold": THRESHOLD,
        "group_metrics": group_metrics,
    }
    out = os.path.join(OUT_DIR, "emotyc_parquet_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Résumé exporté : {out}")


if __name__ == "__main__":
    main()