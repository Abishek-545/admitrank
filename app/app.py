# -*- coding: utf-8 -*-
# AdmitRank — Train • Predict • Explain
# Two screens (tabs), richer visuals, strict Top-K from prediction CSV only,
# tolerant PDF-ZIP parsing.

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
    confusion_matrix, roc_curve, auc,
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

# pdf
from PyPDF2 import PdfReader
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None

import matplotlib.pyplot as plt


# ------------------ Utilities ------------------

def detect_task(y: pd.Series) -> str:
    """Heuristic: numeric -> regression unless very few unique or binary; else classification."""
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
    """Factory returns estimator per task/family with sensible defaults."""
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
            st.info("XGBoost not available; using RandomForest instead.")
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
            st.info("XGBoost not available; using RandomForest instead.")
            return RandomForestRegressor(n_estimators=300, random_state=random_state)
    return LogisticRegression(max_iter=300) if task == "classification" else LinearRegression()


def evaluate(task: str,
             y_true: np.ndarray,
             y_pred: np.ndarray,
             y_proba: Optional[np.ndarray] = None) -> Dict:
    if task == "classification":
        out = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="weighted"),
        }
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


# ------------------ Tolerant PDF ZIP ------------------

PDF_PAT = re.compile(
    r"^(?P<key>[\w\-]+)_(?P<kind>SOP|CV|LOR\d?)"
    r"(?:\.(?P<ext>pdf))?$",
    re.IGNORECASE
)


def build_doc_index_from_zip(zip_file, key_regex: re.Pattern = PDF_PAT):
    """
    Build { key: {"SOP":[bytes], "CV":[bytes], "LOR":[bytes]} } from a ZIP.
    Tolerant: allow names without .pdf, accept LOR/LOR1/LOR2. Skip bad names with a warning.
    """
    idx: Dict[str, Dict[str, List[bytes]]] = {}
    skipped = []

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_file.read()))
    except zipfile.BadZipFile:
        raise ValueError("Uploaded file is not a valid ZIP.")

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
            b = zf.read(info)
        except Exception:
            skipped.append(name)
            continue

        entry = idx.setdefault(key, {"SOP": [], "CV": [], "LOR": []})
        entry[kind_base].append(b)

    if skipped:
        st.warning(
            "Some files were skipped due to unrecognized names. Use `<KEY>_SOP.pdf`, "
            "`<KEY>_CV.pdf`, `<KEY>_LOR.pdf` (or `_LOR1.pdf`, `_LOR2.pdf`).\n"
            f"Skipped: {', '.join(skipped[:10])}{' …' if len(skipped) > 10 else ''}"
        )

    if not idx:
        raise ValueError("No valid SOP/LOR/CV PDFs were found in the ZIP.")
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
    """Heuristic [0,1] score based on length + sentiment (if VADER present)."""
    if not text:
        return 0.0
    n_words = max(1, len(text.split()))
    len_score = np.tanh(n_words / 300.0)
    sent_score = 0.5
    if _VADER is not None:
        try:
            s = _VADER.polarity_scores(text)["compound"]
            sent_score = 0.5 + 0.5 * s
        except Exception:
            pass
    raw = 0.7 * len_score + 0.3 * sent_score
    return float(np.clip(raw, 0.0, 1.0))


def score_docs_index(idx: Dict[str, Dict[str, List[bytes]]]) -> Dict[str, float]:
    out = {}
    for key, groups in idx.items():
        scores = []
        for kind in ("SOP", "LOR", "CV"):
            for b in groups.get(kind, []):
                t = _pdf_bytes_to_text(b)
                scores.append(score_text_simple(t))
        out[key] = float(np.mean(scores)) if scores else 0.0
    return out


# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="AdmitRank — Train • Predict • Explain", layout="wide")
st.title("AdmitRank — Train • Predict • Explain")
st.caption(
    "Upload any historical CSV to train, then predict on a new CSV. "
    "Optionally add a ZIP of SOP/LOR/CV PDFs named like `1234_SOP.pdf`, "
    "`1234_LOR1.pdf`, `1234_CV.pdf` (extension optional) where `1234` matches a key column."
)

tabs = st.tabs(["Train", "Predict"])

# Session variables
if "trained" not in st.session_state:
    st.session_state.trained = False
    st.session_state.pipe = None
    st.session_state.task = None
    st.session_state.features = None
    st.session_state.target = None
    st.session_state.metrics = None
    st.session_state.family = None
    st.session_state.train_df_preview = None


# ------------------ TRAIN TAB ------------------
with tabs[0]:
    st.header("Train a model")
    up = st.file_uploader("Upload historical training CSV", type=["csv"], key="train_csv")
    if up:
        df_hist = pd.read_csv(up)
        st.session_state.train_df_preview = df_hist.head(10)
        st.dataframe(st.session_state.train_df_preview, use_container_width=True)

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
            ["Linear/Simple", "RandomForest", "XGBoost" if HAS_XGB else "Linear/Simple"],
            index=1 if task == "classification" else 0
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            num_strategy = st.selectbox("Numeric impute", ["median", "mean", "most_frequent"], index=0)
        with c2:
            cat_strategy = st.selectbox("Categorical impute", ["most_frequent", "constant"], index=0)
        with c3:
            test_size = st.slider("Hold-out test size", 0.05, 0.30, 0.10, step=0.01)

        if st.button("Train"):
            if not features:
                st.error("Please choose at least one feature.")
            else:
                X = df_hist[features].copy()
                y = df_hist[target].copy()

                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=test_size, random_state=42,
                    stratify=y if detect_task(y) == "classification" else None
                )

                pre = build_preprocess(df_hist, features, num_strategy, cat_strategy)
                est = make_model(task, family)
                pipe = Pipeline(steps=[("pre", pre), ("model", est)])
                pipe.fit(X_tr, y_tr)

                # Evaluate
                st.subheader("Hold-out evaluation")
                if task == "classification":
                    y_pred = pipe.predict(X_te)
                    try:
                        y_proba = pipe.predict_proba(X_te)
                    except Exception:
                        y_proba = None
                    metrics = evaluate(task, y_te, y_pred, y_proba)
                    ms = f"Accuracy: {metrics['accuracy']:.3f} | F1: {metrics['f1']:.3f}"
                    if "roc_auc" in metrics:
                        ms += f" | AUC: {metrics['roc_auc']:.3f}"
                    st.success(ms)

                    # Confusion matrix
                    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                    cm = confusion_matrix(y_te, y_pred)
                    ax[0].imshow(cm, cmap="Blues")
                    ax[0].set_title("Confusion matrix")
                    ax[0].set_xlabel("Predicted")
                    ax[0].set_ylabel("True")
                    for (i, j), v in np.ndenumerate(cm):
                        ax[0].text(j, i, str(v), ha="center", va="center")

                    # ROC curve (binary only)
                    if y_proba is not None and (
                        (y_proba.ndim == 1) or (y_proba.ndim == 2 and y_proba.shape[1] == 2)
                    ):
                        if y_proba.ndim == 2:
                            scores = y_proba[:, 1]
                        else:
                            scores = y_proba
                        fpr, tpr, _ = roc_curve(y_te, scores)
                        roc_auc = auc(fpr, tpr)
                        ax[1].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
                        ax[1].plot([0, 1], [0, 1], "--", color="gray")
                        ax[1].set_title("ROC curve")
                        ax[1].set_xlabel("FPR")
                        ax[1].set_ylabel("TPR")
                        ax[1].legend()
                    else:
                        ax[1].axis("off")

                    st.pyplot(fig)

                else:
                    y_pred = pipe.predict(X_te)
                    metrics = evaluate(task, y_te, y_pred)
                    st.success(f"R²: {metrics['r2']:.3f} | MAE: {metrics['mae']:.3f} | RMSE: {metrics['rmse']:.3f}")

                    # Residuals
                    res = y_te - y_pred
                    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                    ax[0].scatter(y_pred, res, alpha=0.6)
                    ax[0].axhline(0, color="red", ls="--")
                    ax[0].set_title("Residuals vs Prediction")
                    ax[0].set_xlabel("Prediction")
                    ax[0].set_ylabel("Residual")

                    ax[1].hist(res, bins=20, color="#5b9bd5")
                    ax[1].set_title("Residuals histogram")
                    st.pyplot(fig)

                # Feature importance / coefficients
                st.subheader("Global feature importance")
                try:
                    model = pipe.named_steps["model"]
                    # If tree-based
                    if hasattr(model, "feature_importances_"):
                        # Get final feature names after preprocessing
                        cols = []
                        pre: ColumnTransformer = pipe.named_steps["pre"]
                        for name, trans, cols_sel in pre.transformers_:
                            if name == "cat":
                                ohe = trans.named_steps["ohe"]
                                cols.extend(list(ohe.get_feature_names_out(cols_sel)))
                            elif name == "num":
                                cols.extend(cols_sel)
                        imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)[:20]
                        st.bar_chart(imp)
                    elif hasattr(model, "coef_"):
                        coef = np.ravel(model.coef_)
                        pre: ColumnTransformer = pipe.named_steps["pre"]
                        cols = []
                        for name, trans, cols_sel in pre.transformers_:
                            if name == "cat":
                                ohe = trans.named_steps["ohe"]
                                cols.extend(list(ohe.get_feature_names_out(cols_sel)))
                            elif name == "num":
                                cols.extend(cols_sel)
                        imp = pd.Series(coef, index=cols).sort_values(key=np.abs, ascending=False)[:20]
                        st.bar_chart(imp)
                    else:
                        st.info("This model does not expose a standard importance/coefficient interface.")
                except Exception as e:
                    st.info(f"Feature importance unavailable: {e}")

                # Save to session
                st.session_state.trained = True
                st.session_state.pipe = pipe
                st.session_state.task = task
                st.session_state.features = features
                st.session_state.target = target
                st.session_state.metrics = metrics
                st.session_state.family = family


# ------------------ PREDICT TAB ------------------
with tabs[1]:
    st.header("Predict on a new applicants CSV")

    if not st.session_state.trained:
        st.info("Train a model first in the **Train** tab.")
    else:
        # Fusion and Top-K
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("**Fusion weight α (0 → documents only, 1 → tabular only)**")
            alpha = st.slider("α", 0.0, 1.0, 0.70, step=0.05, label_visibility="collapsed")
        with c2:
            top_k = st.number_input("Top-K (overall)", min_value=1, max_value=100, value=5, step=1)

        pred_file = st.file_uploader("Upload applicants CSV (prediction set)", type=["csv"], key="pred_csv")
        doc_zip = st.file_uploader("Optional: ZIP of SOP/LOR/CV PDFs", type=["zip"], key="doc_zip")

        if pred_file is not None:
            # IMPORTANT: df_pred comes ONLY from the uploaded prediction CSV
            df_pred = pd.read_csv(pred_file)
            st.dataframe(df_pred.head(10), use_container_width=True)

            missing = [c for c in st.session_state.features if c not in df_pred.columns]
            if missing:
                st.error(f"Prediction CSV is missing columns used by the model: {missing}")
            else:
                pipe = st.session_state.pipe
                task = st.session_state.task
                Xp = df_pred[st.session_state.features].copy()

                # Tabular probability/score
                if task == "classification":
                    try:
                        proba = pipe.predict_proba(Xp)
                        p_tab = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
                    except Exception:
                        y_pred = pipe.predict(Xp)
                        mode = pd.Series(y_pred).mode().iloc[0]
                        p_tab = (y_pred == mode).astype(float)
                else:
                    yhat = pipe.predict(Xp).astype(float)
                    lo, hi = np.nanmin(yhat), np.nanmax(yhat)
                    p_tab = (yhat - lo) / (hi - lo + 1e-9)

                df_out = df_pred.copy()
                df_out["p_tabular"] = p_tab

                # Document score by key (ONLY map to rows present in df_pred)
                if doc_zip is not None:
                    key_col = st.selectbox(
                        "Select key column matching PDF names (e.g., 'Serial No.')",
                        options=list(df_pred.columns),
                        help="Only rows in this prediction CSV are used."
                    )
                    try:
                        idx = build_doc_index_from_zip(doc_zip)
                        by_key = score_docs_index(idx)  # dict
                        df_out["doc_score"] = df_pred[key_col].astype(str).map(
                            lambda k: by_key.get(str(k), np.nan)
                        )
                    except Exception as e:
                        st.error(f"Could not parse ZIP: {e}")
                        df_out["doc_score"] = np.nan
                else:
                    df_out["doc_score"] = np.nan

                # Fusion per-row (if doc missing for a row, use tabular for that row)
                doc_fill = df_out["doc_score"].fillna(df_out["p_tabular"])
                df_out["p_final"] = alpha * df_out["p_tabular"] + (1 - alpha) * doc_fill

                # ------- Visuals for prediction set -------
                st.subheader("Prediction score distribution")
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(df_out["p_final"], bins=20, color="#5b9bd5")
                ax.set_xlabel("p_final")
                ax.set_ylabel("Count")
                ax.set_title("Histogram of predicted scores")
                st.pyplot(fig)

                # ------- Strict Top-K from prediction CSV only -------
                st.subheader("Top-K applicants (overall)")
                top_overall = df_out.sort_values("p_final", ascending=False).head(int(top_k))
                st.dataframe(top_overall, use_container_width=True)

                st.download_button(
                    "Download predictions CSV",
                    data=df_out.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv"
                )
