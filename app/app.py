# app/app.py
# AdmitRank — Train • Predict • Explain
# UI aligned with user's screenshot.
# Features:
# - Train tab: upload CSV, choose target/features, select model family, evaluate on 10% split
# - Predict tab: upload prediction CSV, optional ZIP of PDFs, fusion alpha, Top-K (overall)
# - Tolerant ZIP parsing: names with/without .pdf, allow LOR/LOR1/LOR2, skip bad names with warning

import io
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# optional XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# PDF reading
from PyPDF2 import PdfReader
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None


# -------------------- Helpers --------------------

def detect_task(y: pd.Series) -> str:
    """Heuristic: numeric -> regression unless few unique/binary; non-numeric -> classification."""
    if pd.api.types.is_numeric_dtype(y):
        nuniq = y.dropna().nunique()
        if nuniq <= 5 or set(y.dropna().unique()).issubset({0, 1}):
            return "classification"
        return "regression"
    return "classification"


def split_features(df: pd.DataFrame, features: List[str]) -> Tuple[List[str], List[str]]:
    num = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in features if c not in num]
    return num, cat


def build_preprocess(df: pd.DataFrame,
                     features: List[str],
                     num_strategy: str = "median",
                     cat_strategy: str = "most_frequent") -> ColumnTransformer:
    num_cols, cat_cols = split_features(df, features)
    transformers = []
    if num_cols:
        transformers.append(
            ("num", SimpleImputer(strategy=num_strategy), num_cols)
        )
    if cat_cols:
        transformers.append(
            ("cat", Pipeline(steps=[
                ("imp", SimpleImputer(strategy=cat_strategy)),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        )
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return pre


def make_model(task: str, family: str, random_state: int = 42):
    """Factory: return estimator per task/family with sensible defaults."""
    if task == "classification":
        if family == "Linear/Simple":
            return LogisticRegression(max_iter=300)
        if family == "RandomForest":
            return RandomForestClassifier(n_estimators=300, random_state=random_state)
        if family == "XGBoost":
            if HAS_XGB:
                return XGBClassifier(
                    n_estimators=400, learning_rate=0.05, max_depth=5,
                    subsample=0.9, colsample_bytree=0.9, random_state=random_state,
                    eval_metric="logloss"
                )
            st.info("XGBoost not available, using RandomForest.")
            return RandomForestClassifier(n_estimators=300, random_state=random_state)
    else:
        if family == "Linear/Simple":
            return LinearRegression()
        if family == "RandomForest":
            return RandomForestRegressor(n_estimators=300, random_state=random_state)
        if family == "XGBoost":
            if HAS_XGB:
                return XGBRegressor(
                    n_estimators=400, learning_rate=0.05, max_depth=5,
                    subsample=0.9, colsample_bytree=0.9, random_state=random_state
                )
            st.info("XGBoost not available, using RandomForest.")
            return RandomForestRegressor(n_estimators=300, random_state=random_state)
    # fallback
    return LogisticRegression(max_iter=300) if task == "classification" else LinearRegression()


def evaluate(task: str,
             y_true: np.ndarray,
             y_pred: np.ndarray,
             y_proba: Optional[np.ndarray] = None) -> Dict:
    """Simple metrics for display."""
    if task == "classification":
        out = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="weighted"),
        }
        # Try ROC-AUC for binary
        try:
            if y_proba is not None:
                if y_proba.ndim == 1:
                    out["roc_auc"] = roc_auc_score(y_true, y_proba)
                elif y_proba.shape[1] == 2:
                    out["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            pass
        return out
    else:
        return {
            "r2": r2_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))
        }


# -------------------- PDF & ZIP (tolerant) --------------------

# Accept <KEY>_(SOP|CV|LOR|LOR1|LOR2)(.pdf optional), case-insensitive
PDF_PAT = re.compile(
    r"^(?P<key>[\w\-]+)_(?P<kind>SOP|CV|LOR\d?)"
    r"(?:\.(?P<ext>pdf))?$",
    re.IGNORECASE
)


def build_doc_index_from_zip(zip_file, key_regex: re.Pattern = PDF_PAT):
    """
    Build { key: {"SOP":[bytes], "CV":[bytes], "LOR":[bytes]} } from a user ZIP.
    Tolerant:
      - Names may omit .pdf
      - Accept LOR, LOR1, LOR2 (mapped to LOR)
      - Non-matching files skipped with a warning
    """
    skipped = []
    idx: Dict[str, Dict[str, List[bytes]]] = {}

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_file.read()))
    except zipfile.BadZipFile:
        raise ValueError("The uploaded file is not a valid ZIP.")

    for info in zf.infolist():
        if info.is_dir():
            continue
        name = Path(info.filename).name
        m = key_regex.match(name) or key_regex.match(Path(name).stem)
        if not m:
            skipped.append(name)
            continue

        key = m.group("key")
        kind = m.group("kind").upper()
        kind_base = "LOR" if kind.startswith("LOR") else kind

        try:
            data = zf.read(info)
        except KeyError:
            skipped.append(name)
            continue

        entry = idx.setdefault(key, {"SOP": [], "CV": [], "LOR": []})
        entry[kind_base].append(data)

    if skipped:
        st.warning(
            "Some files were skipped (bad name). Use `<KEY>_SOP.pdf`, `<KEY>_CV.pdf`, "
            "`<KEY>_LOR.pdf` (or `_LOR1.pdf`, `_LOR2.pdf`).\n"
            f"Skipped: {', '.join(skipped[:10])}{' …' if len(skipped) > 10 else ''}"
        )

    if not idx:
        raise ValueError(
            "No valid documents were found in the ZIP. Make sure names look like "
            "`380_SOP.pdf`, `380_LOR1.pdf`, `380_CV.pdf` (extension optional)."
        )
    return idx


def _pdf_bytes_to_text(b: bytes) -> str:
    txt = []
    try:
        with io.BytesIO(b) as bio:
            reader = PdfReader(bio)
            for pg in reader.pages:
                try:
                    txt.append(pg.extract_text() or "")
                except Exception:
                    continue
    except Exception:
        return ""
    return "\n".join(txt)


def score_text_simple(text: str) -> float:
    """Heuristic [0,1] score based on length and sentiment (if VADER available)."""
    if not text:
        return 0.0
    n_words = max(1, len(text.split()))
    len_score = np.tanh(n_words / 300.0)  # saturate ~300 words
    sent_score = 0.5
    if _VADER is not None:
        try:
            s = _VADER.polarity_scores(text)["compound"]
            sent_score = 0.5 + 0.5 * s  # [-1,1] -> [0,1]
        except Exception:
            pass
    raw = 0.7 * len_score + 0.3 * sent_score
    return float(np.clip(raw, 0.0, 1.0))


def score_docs_index(idx: Dict[str, Dict[str, List[bytes]]]) -> Dict[str, float]:
    """Aggregate per-key document scores (mean over SOP/LOR/CV files)."""
    out = {}
    for key, groups in idx.items():
        scores = []
        for kind in ("SOP", "LOR", "CV"):
            for b in groups.get(kind, []):
                t = _pdf_bytes_to_text(b)
                scores.append(score_text_simple(t))
        out[key] = float(np.mean(scores)) if scores else 0.0
    return out


# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="AdmitRank — Train • Predict • Explain", layout="wide")

st.title("AdmitRank — Train • Predict • Explain")
st.caption(
    "Upload any historical CSV to train, then predict on a new CSV. "
    "Optionally add a ZIP of SOP/LOR/CV PDFs named like "
    "`1234_SOP.pdf`, `1234_LOR1.pdf`, `1234_CV.pdf` where `1234` matches a key column (e.g., *Serial No.*)."
)

tabs = st.tabs(["Train", "Predict"])

# Session state
if "trained" not in st.session_state:
    st.session_state.trained = False
    st.session_state.pipe = None
    st.session_state.task = None
    st.session_state.target = None
    st.session_state.features = None
    st.session_state.metrics = None
    st.session_state.family = None


# --------------- TRAIN TAB ---------------
with tabs[0]:
    st.header("Train a model")
    up = st.file_uploader("Upload historical training CSV", type=["csv"], key="train_csv")
    if up:
        df_hist = pd.read_csv(up)
        st.write(df_hist.head())

        target = st.selectbox("Target column", options=list(df_hist.columns))
        default_feats = [c for c in df_hist.columns if c != target]
        features = st.multiselect("Feature columns", options=list(df_hist.columns), default=default_feats)

        if features:
            task = detect_task(df_hist[target])
            st.info(f"Detected task: **{task}**")
        else:
            task = "classification"

        family = st.selectbox(
            "Model family",
            options=["Linear/Simple", "RandomForest", "XGBoost" if HAS_XGB else "Linear/Simple"],
            index=1 if task == "classification" else 0
        )

        colA, colB, colC = st.columns(3)
        with colA:
            num_strategy = st.selectbox("Numeric impute", ["median", "mean", "most_frequent"], index=0)
        with colB:
            cat_strategy = st.selectbox("Categorical impute", ["most_frequent", "constant"], index=0)
        with colC:
            test_size = st.slider("Hold-out test size", 0.05, 0.3, 0.10, step=0.01)

        if st.button("Train"):
            if not features:
                st.error("Select at least one feature.")
            else:
                X = df_hist[features].copy()
                y = df_hist[target].copy()

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42,
                    stratify=y if detect_task(y) == "classification" else None
                )

                pre = build_preprocess(df_hist, features, num_strategy, cat_strategy)
                est = make_model(task, family)
                pipe = Pipeline(steps=[("pre", pre), ("model", est)])
                pipe.fit(X_train, y_train)

                # Evaluate
                if task == "classification":
                    y_pred = pipe.predict(X_test)
                    try:
                        y_proba = pipe.predict_proba(X_test)
                    except Exception:
                        y_proba = None
                    metrics = evaluate(task, y_test, y_pred, y_proba)
                    msg = f"Accuracy: {metrics['accuracy']:.3f} | F1: {metrics['f1']:.3f}"
                    if "roc_auc" in metrics:
                        msg += f" | AUC: {metrics['roc_auc']:.3f}"
                    st.success(msg)
                else:
                    y_pred = pipe.predict(X_test)
                    metrics = evaluate(task, y_test, y_pred)
                    st.success(f"R²: {metrics['r2']:.3f} | MAE: {metrics['mae']:.3f} | RMSE: {metrics['rmse']:.3f}")

                # Save
                st.session_state.trained = True
                st.session_state.pipe = pipe
                st.session_state.task = task
                st.session_state.target = target
                st.session_state.features = features
                st.session_state.metrics = metrics
                st.session_state.family = family

                st.write("Model trained with features:", features)
                st.write("Metrics:", metrics)


# --------------- PREDICT TAB ---------------
with tabs[1]:
    st.header("Predict on a new applicants CSV")

    if not st.session_state.trained:
        st.info("Train a model first in the **Train** tab.")
    else:
        # Fusion
        st.markdown("**Fusion weight α (0 → documents only, 1 → tabular only)**")
        alpha = st.slider("α", 0.0, 1.0, 0.70, step=0.05, label_visibility="collapsed")

        # Top-K only from prediction CSV
        top_k = st.number_input("Top-K (overall)", min_value=1, max_value=100, value=5, step=1)

        pred_file = st.file_uploader("Upload applicants CSV for prediction", type=["csv"], key="pred_csv")
        doc_zip = st.file_uploader("Optional: upload ZIP of SOP/LOR/CV PDFs", type=["zip"], key="doc_zip")

        if pred_file is not None:
            # STRICTLY use only prediction CSV rows for ranking
            df_pred = pd.read_csv(pred_file)
            st.write(df_pred.head())

            missing = [c for c in st.session_state.features if c not in df_pred.columns]
            if missing:
                st.error(f"Prediction CSV is missing columns used by the model: {missing}")
            else:
                pipe = st.session_state.pipe
                task = st.session_state.task

                Xp = df_pred[st.session_state.features].copy()

                # p_tabular
                if task == "classification":
                    try:
                        proba = pipe.predict_proba(Xp)
                        p_tab = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
                    except Exception:
                        # fallback: map predicted label to 1 for top class heuristic
                        y_pred = pipe.predict(Xp)
                        mode = pd.Series(y_pred).mode().iloc[0]
                        p_tab = (y_pred == mode).astype(float)
                else:
                    yhat = pipe.predict(Xp).astype(float)
                    lo, hi = np.nanmin(yhat), np.nanmax(yhat)
                    p_tab = (yhat - lo) / (hi - lo + 1e-9)

                df_out = df_pred.copy()
                df_out["p_tabular"] = p_tab

                # doc_score (key-based)
                if doc_zip is not None:
                    key_col = st.selectbox(
                        "Select key column that matches PDF names (e.g., 'Serial No.')",
                        options=list(df_pred.columns)
                    )
                    try:
                        idx = build_doc_index_from_zip(doc_zip)
                        p_doc_by_key = score_docs_index(idx)  # key->score
                        df_out["doc_score"] = df_pred[key_col].astype(str).map(
                            lambda k: p_doc_by_key.get(str(k), np.nan)
                        )
                    except Exception as e:
                        st.error(f"Could not read ZIP: {e}")
                        df_out["doc_score"] = np.nan
                else:
                    df_out["doc_score"] = np.nan

                # fusion (if doc missing, fall back to tabular for that row)
                doc = df_out["doc_score"].fillna(df_out["p_tabular"])
                df_out["p_final"] = alpha * df_out["p_tabular"] + (1 - alpha) * doc

                # ---- Top-K applicants (overall) — ONLY from prediction df_out ----
                st.markdown("## Top-K applicants (overall)")
                top_overall = df_out.sort_values("p_final", ascending=False).head(int(top_k))
                st.dataframe(top_overall, use_container_width=True)

                # download
                st.download_button(
                    "Download predictions CSV",
                    data=df_out.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv"
                )
