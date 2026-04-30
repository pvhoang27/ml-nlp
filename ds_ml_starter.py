"""
Basic DS/ML project starter (single file).

What it does:
- Train: read a CSV with text + label columns, split, TF-IDF, LogisticRegression
- Evaluate: accuracy + weighted F1 + classification report
- Save: persist the whole sklearn pipeline to a .joblib file
- Predict: load the saved model and run predictions on texts or an input CSV

Examples:
  python ds_ml_starter.py train --data data.csv --text-col text --label-col label --model-out artifacts/model.joblib
  python ds_ml_starter.py predict --model artifacts/model.joblib --text "toi rat thich mon nay"
  python ds_ml_starter.py predict --model artifacts/model.joblib --input-csv data.csv --text-col text --out-csv preds.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


def _toy_dataset() -> pd.DataFrame:
    # Small dataset so the script runs out-of-the-box.
    return pd.DataFrame(
        {
            "text": [
                "san pham rat tot",
                "chat luong kem",
                "toi rat hai long",
                "that te",
                "khong dung nhu mo ta",
                "gia tri tuyet voi",
                "toi se mua lai",
                "khong hai long",
            ],
            "label": ["pos", "neg", "pos", "neg", "neg", "pos", "pos", "neg"],
        }
    )


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    return df


def _validate_columns(df: pd.DataFrame, text_col: str, label_col: str | None) -> None:
    missing: list[str] = []
    if text_col not in df.columns:
        missing.append(text_col)
    if label_col is not None and label_col not in df.columns:
        missing.append(label_col)
    if missing:
        raise ValueError(f"Missing columns in data: {missing}. Available: {list(df.columns)}")


def _build_pipeline(max_features: int | None) -> Pipeline:
    vectorizer_kwargs: dict[str, Any] = {"ngram_range": (1, 2), "lowercase": True}
    if max_features is not None:
        vectorizer_kwargs["max_features"] = max_features

    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**vectorizer_kwargs)),
            ("clf", LogisticRegression(max_iter=500, solver="liblinear")),
        ]
    )


def _dump_model(model: Pipeline, path: Path) -> None:
    if joblib is None:
        raise RuntimeError("joblib is not available. Install it or use a different persistence method.")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def _load_model(path: Path) -> Pipeline:
    if joblib is None:
        raise RuntimeError("joblib is not available. Install it or use a different persistence method.")
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    obj = joblib.load(path)
    if not isinstance(obj, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline, got: {type(obj)}")
    return obj


def cmd_train(args: argparse.Namespace) -> int:
    if args.data is None:
        df = _toy_dataset()
        print("No --data provided; using a small toy dataset.", file=sys.stderr, flush=True)
    else:
        df = _load_csv(Path(args.data))

    _validate_columns(df, args.text_col, args.label_col)

    X = df[args.text_col].astype(str)
    y = df[args.label_col]

    stratify = y if y.nunique(dropna=False) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    model = _build_pipeline(max_features=args.max_features)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"accuracy={acc:.4f}")
    print(f"f1_weighted={f1:.4f}")
    # zero_division avoids warnings when a class is not predicted (common on tiny toy data).
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    if args.model_out is not None:
        out_path = Path(args.model_out)
        _dump_model(model, out_path)
        print(f"saved_model={out_path}")

    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model = _load_model(Path(args.model))

    texts: list[str] = []
    if args.text:
        texts.extend(args.text)

    df_in: pd.DataFrame | None = None
    if args.input_csv is not None:
        df_in = _load_csv(Path(args.input_csv))
        _validate_columns(df_in, args.text_col, None)
        texts.extend(df_in[args.text_col].astype(str).tolist())

    if not texts:
        raise ValueError("Provide --text and/or --input-csv.")

    preds = model.predict(texts)

    if args.out_csv is not None:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df = pd.DataFrame({"text": texts, "pred": preds})
        out_df.to_csv(out_path, index=False)
        print(f"saved_predictions={out_path}")
        return 0

    # stdout: one prediction per line
    for p in preds:
        print(p)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ds_ml_starter.py", add_help=True)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train + evaluate + (optionally) save model")
    t.add_argument("--data", type=str, default=None, help="Path to CSV. If omitted, uses a toy dataset.")
    t.add_argument("--text-col", type=str, default="text", help="Text column name in CSV")
    t.add_argument("--label-col", type=str, default="label", help="Label column name in CSV")
    t.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    t.add_argument("--seed", type=int, default=42, help="Random seed")
    t.add_argument("--max-features", type=int, default=None, help="TF-IDF max_features (optional)")
    t.add_argument("--model-out", type=str, default="artifacts/model.joblib", help="Where to save the model")
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="Predict with a saved model")
    pr.add_argument("--model", type=str, required=True, help="Path to saved .joblib model")
    pr.add_argument("--text", type=str, nargs="*", default=None, help="One or more texts")
    pr.add_argument("--input-csv", type=str, default=None, help="CSV to predict on")
    pr.add_argument("--text-col", type=str, default="text", help="Text column name in input CSV")
    pr.add_argument("--out-csv", type=str, default=None, help="Write predictions to CSV")
    pr.set_defaults(func=cmd_predict)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
