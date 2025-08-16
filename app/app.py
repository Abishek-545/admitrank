# app/app.py
# AdmitRank — Train • Predict • Explain
# Two-screen UI with Train & Predict tabs
# - Train: fit a model on any historical CSV, evaluate on 10% holdout, save model in session
# - Predict: upload a new CSV, (optionally) upload a ZIP of PDFs (SOP/LOR/CV), compute tabular prob,
#            doc score (optional), fuse with alpha, and show Top-K strictly from this prediction set.
# - Robust to missing PDFs: if a PDF is missing, tabular score is used as-is.

import io
import re
import zipfile
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import LinearSVR, LinearSVC

# PDF parsing (tolerant if unavailable on some environments)
try:
    from PyPDF2 import PdfReader  # type: ignore
    PYPDF_AVAILABLE = True
except Exception:
    PYPDF_AVAILABLE = False

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AdmitRank — Train • Predict • Explain",
    page_icon="🎓",
    layout="wide",
)

# --------------- helpers -----------------------------------------------------------------


def detect_task_type(y: pd.Series) -> str:
    """
    Decide whether this is a regression or classification target.
    - If numeric with many unique values -> regression
    - If numeric/category with <= 10 unique values -> classification
    """
    if y.dtype.kind in "ifu":
        nun = y.dropna().nunique()
        if nun <= 10:
            return "classification"
        return "regression"
    else:
        # non-numeric -> classification
        return "classification"


def build_preprocessor(
    df: pd.DataFrame, features: List[str]
) -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
    """Create a ColumnTransformer for numeric, categorical and text features."""
    numeric_cols, cat_cols, text_cols = [], [], []
    for c in features:
        if df[c].dtype.kind in "ifu":
            numeric_cols.append(c)
        elif df[c].dtype == "object":
            # crude heuristic: treat long average length columns as text
            if df[c].dropna().astype(str).str.len().mean() > 20:
                text_cols.append(c)
            else:
                cat_cols.append(c)
        else:
            cat_cols.append(c)

    tfms = []
    if numeric_cols:
        tfms.append(
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                   ("scaler", StandardScaler())]), numeric_cols)
        )
    if cat_cols:
        tfms.append(
            ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                   ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
        )
    if text_cols:
        tfms.append(
            ("txt", Pipeline(steps=[("tfidf", TfidfVectorizer(max_features=4000, ngram_range=(1, 2)))]),
             # ColumnTransformer expects 1D input for Tfidf: use 'passthrough' later via custom selector
             # We'll use a wrapper that picks .values.ravel() later via a FunctionTransformer,
             # but a simpler route is to combine text columns into one string.
             # So here we just remember text cols, and we will concatenate in a new column before fitting.
             text_cols)
        )

    preprocessor = ColumnTransformer(tfms, remainder="drop", verbose_feature_names_out=False)
    return preprocessor, numeric_cols, cat_cols, text_cols


def concat_text_columns(df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
    """Concatenate multiple text columns into a single '___TEXT___' column for TF-IDF."""
    if not text_cols:
        return df
    joined = df[text_cols].astype(str).apply(lambda r: " | ".join(r.values), axis=1)
    out = df.copy()
    out["___TEXT___"] = joined
    # swap text_cols list to the single combined column
    return out


def choose_model(task: str, family: str):
    """
    Pick a light-weight scikit-learn model family.
    family in {"Linear", "Tree", "SVM"}.
    """
    if task == "regression":
        if family == "Linear":
            return LinearRegression()
        if family == "SVM":
            # keep it simple, bounded complexity
            return LinearSVR(random_state=42)
        # default tree
        return RandomForestRegressor(n_estimators=150, random_state=42)

    # classification
    if family == "Linear":
        # liblinear supports small/medium datasets well
        return LogisticRegression(max_iter=1000, solver="liblinear")
    if family == "SVM":
        # use linear svc; no probability => we will calibrate to decision_function
        return LinearSVC()
    # default tree
    return RandomForestClassifier(n_estimators=200, random_state=42)


def preds_to_probability(task: str, model, X) -> np.ndarray:
    """
    Produce probabilities in [0, 1].
    - classification: prefer predict_proba if available; else map decision_function to (0,1) via logistic.
    - regression: min-max normalize predictions row-wise to [0,1] (per batch).
    """
    if task == "classification":
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.shape[1] == 1:
                # weird edge-case: single column, treat as positive prob
                return proba.ravel()
            # assume positive class is the last column by convention
            return proba[:, -1]
        # fallback: decision_function -> logistic squashing
        if hasattr(model, "decision_function"):
            s = model.decision_function(X)
            return 1 / (1 + np.exp(-s))
        # last resort: predict labels, map to {0,1}
        labs = model.predict(X)
        # try to coerce strings like "YES"/"NO"
        if labs.dtype.kind in "OUS":
            return np.where(labs.astype(str).str.upper().isin(["1", "YES", "Y", "TRUE"]), 1.0, 0.0)
        return (labs == 1).astype(float)

    # regression
    yhat = model.predict(X).astype(float)
    # min-max scale within this batch (avoid /0)
    mn, mx = np.nanmin(yhat), np.nanmax(yhat)
    if mx - mn < 1e-9:
        return np.clip((yhat - mn), 0, 1)
    return np.clip((yhat - mn) / (mx - mn), 0, 1)


def parse_pdf_text(file_bytes: bytes) -> str:
    """Extract text from a PDF bytes object. Return empty string if parsing fails or library missing."""
    if not PYPDF_AVAILABLE:
        return ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = []
        for page in reader.pages:
            try:
                text.append(page.extract_text() or "")
            except Exception:
                pass
        return "\n".join(text).strip()
    except Exception:
        return ""


def simple_doc_score(text: str) -> float:
    """
    Very light heuristic scoring for docs:
    - normalize length
    - reward presence of positive keywords
    """
    if not text:
        return np.nan
    l = len(text)
    # length score
    length_score = np.tanh(l / 5000.0)  # saturates ~1 for long docs

    textU = text.lower()
    kws = ["research", "publication", "internship", "award", "project", "paper"]
    kw_score = min(sum(1 for k in kws if k in textU) / 6.0, 1.0)

    score = 0.7 * length_score + 0.3 * kw_score
    return float(np.clip(score, 0.0, 1.0))


def build_doc_index_from_zip(zip_bytes: bytes) -> dict:
    """
    Read a ZIP and index files by ID & type (SOP/LOR/CV). We accept case-insensitive names like:
        1234_SOP.pdf, 1234_LOR.pdf, 1234_LOR1.pdf, 1234_CV.pdf
    Extension is optional in the regex, but we'll only actually parse PDFs.
    Returns: { id -> {"SOP": [bytes,...], "LOR": [bytes,...], "CV": [bytes,...]} }
    """
    index = {}
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for n in z.namelist():
                base = n.split("/")[-1]  # ignore folder levels
                m = re.match(r"^(.+?)_(SOP|LOR\d*|LOR|CV)(?:\.[A-Za-z0-9]+)?$", base, flags=re.IGNORECASE)
                if not m:
                    continue
                applicant_id = str(m.group(1))
                typ = m.group(2).upper()
                if typ.startswith("LOR"):
                    typ = "LOR"
                # only accept PDFs or files we can read as bytes -> parse_pdf_text will handle empties
                try:
                    data = z.read(n)
                except Exception:
                    continue
                index.setdefault(applicant_id, {}).setdefault(typ, []).append(data)
    except Exception:
        return {}
    return index


def score_docs_for_row(applicant_id: str, doc_index: dict) -> float:
    """Average simple_doc_score over all available docs for that applicant. Return NaN if none."""
    if doc_index is None or applicant_id not in doc_index:
        return np.nan
    texts = []
    for typ in ("SOP", "LOR", "CV"):
        for b in doc_index[applicant_id].get(typ, []):
            txt = parse_pdf_text(b)
            if txt:
                texts.append(txt)
    if not texts:
        return np.nan
    ss = [simple_doc_score(t) for t in texts if t]
    if not ss:
        return np.nan
    return float(np.nanmean(ss))


def fuse_scores(tab: float, doc: float, alpha: float) -> float:
    """If doc is NaN -> return tab. Else weighted fusion."""
    if np.isnan(doc):
        return float(tab)
    return float(alpha * tab + (1.0 - alpha) * doc)


def small_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """Light-weight feature importance (for tree) or coefficients (for linear)."""
    try:
        if hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
            return pd.DataFrame({"feature": feature_names, "importance": vals}).sort_values(
                "importance", ascending=False
            )
        if hasattr(model, "coef_"):
            co = model.coef_
            if co.ndim > 1:
                co = co.ravel()
            vals = np.abs(co)
            return pd.DataFrame({"feature": feature_names, "importance": vals}).sort_values(
                "importance", ascending=False
            )
    except Exception:
        pass
    return pd.DataFrame(columns=["feature", "importance"])


# --------------- session state ------------------------------------------------------------

if "trained" not in st.session_state:
    st.session_state.trained = False
if "model" not in st.session_state:
    st.session_state.model = None
if "task" not in st.session_state:
    st.session_state.task = None
if "features" not in st.session_state:
    st.session_state.features = []
if "target" not in st.session_state:
    st.session_state.target = None
if "id_col" not in st.session_state:
    st.session_state.id_col = None
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = None
if "train_metrics" not in st.session_state:
    st.session_state.train_metrics = {}
if "train_cols_text" not in st.session_state:
    st.session_state.train_cols_text = []
if "train_feature_names_out" not in st.session_state:
    st.session_state.train_feature_names_out = []
if "doc_index_train" not in st.session_state:
    st.session_state.doc_index_train = None

# --------------- UI ----------------------------------------------------------------------

st.title("AdmitRank — Train • Predict • Explain")

st.write(
    """
Upload any **historical CSV** to **train**, then predict on a **new CSV**.  
Optionally add a **ZIP of SOP/LOR/CV PDFs** named like `1234_SOP.pdf`, `1234_LOR1.pdf`, `1234_CV.pdf`
(**extension is optional** in the match), where `1234` matches a key column of your CSV.  
Missing PDFs are OK — the system will simply use the tabular model.
"""
)

tab_train, tab_predict = st.tabs(["Train", "Predict"])

# ------------------------------------------------------------------------------------------
# Train tab
# ------------------------------------------------------------------------------------------
with tab_train:
    st.subheader("Train a model")

    up = st.file_uploader("Upload historical training CSV", type=["csv"], key="train_csv")

    if up is not None:
        df_hist = pd.read_csv(up)
        st.dataframe(df_hist.head(12), use_container_width=True)

        # choose target & features
        cols = df_hist.columns.tolist()
        target_col = st.selectbox("Target column", cols, index=len(cols) - 1)
        # id col guess
        id_guess = None
        for k in ["applicant_id", "Serial No.", "Serial No", "Serial", "id", "ID"]:
            if k in cols:
                id_guess = k
                break
        id_col = st.selectbox("ID column (for docs matching)", [None] + cols, index=([None] + cols).index(id_guess) if id_guess else 0)

        default_features = [c for c in cols if c != target_col]
        feat_cols = st.multiselect("Feature columns", default_features, default=default_features)

        family = st.selectbox("Model family", ["Tree", "Linear", "SVM"], index=0)
        test_size = st.slider("Test size (holdout)", 0.05, 0.4, 0.1, 0.05)

        # Build training DF for TF-IDF: if multiple short text cols, we will concatenate them.
        # We'll detect text columns and build preprocessor; but first we concatenate text columns.
        pre, num_cols, cat_cols, text_cols = build_preprocessor(df_hist, feat_cols)
        df_hist_aug = concat_text_columns(df_hist, text_cols)
        # If we concatenated, swap text cols list to single column
        if text_cols:
            text_cols = ["___TEXT___"]
            # need to rebuild preprocessor with the single text column
            tfms = []
            if num_cols:
                tfms.append(
                    ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                           ("scaler", StandardScaler())]), num_cols)
                )
            if cat_cols:
                tfms.append(
                    ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                           ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
                )
            tfms.append(("txt", Pipeline(steps=[("tfidf", TfidfVectorizer(max_features=4000, ngram_range=(1, 2)))]), text_cols))
            pre = ColumnTransformer(tfms, remainder="drop", verbose_feature_names_out=False)

        X = df_hist_aug[feat_cols if "___TEXT___" not in df_hist_aug.columns else (list(set(feat_cols) - set(text_cols)) + text_cols)]
        y = df_hist[target_col]
        task = detect_task_type(y)

        # split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if task == "classification" else None)

        model = choose_model(task, family)
        pipe = Pipeline([("pre", pre), ("model", model)])

        # fit
        pipe.fit(X_tr, y_tr)

        # evaluate
        if task == "classification":
            yhat = pipe.predict(X_te)
            acc = accuracy_score(y_te, yhat)
            f1 = f1_score(y_te, yhat, average="binary" if y.dropna().nunique() == 2 else "macro")
            st.success(f"Classification — Accuracy: **{acc:.3f}**, F1: **{f1:.3f}** on {len(y_te)} rows")
            st.caption("Tip: feature importances / coefficients (if available) shown below.")
        else:
            pred = pipe.predict(X_te)
            rmse = mean_squared_error(y_te, pred, squared=False)
            mae = mean_absolute_error(y_te, pred)
            r2 = r2_score(y_te, pred)
            st.success(f"Regression — RMSE: **{rmse:.3f}**, MAE: **{mae:.3f}**, R²: **{r2:.3f}** on {len(y_te)} rows")

        # feature importance (if available)
        try:
            # names out:
            try:
                f_out = pipe[:-1].get_feature_names_out()
            except Exception:
                f_out = feat_cols
            imp = small_feature_importance(pipe[-1], list(f_out))
            if not imp.empty:
                st.write("Top features / coefficients")
                st.dataframe(imp.head(20), use_container_width=True)
        except Exception:
            pass

        # cache into session
        st.session_state.trained = True
        st.session_state.model = pipe
        st.session_state.task = task
        st.session_state.features = feat_cols
        st.session_state.target = target_col
        st.session_state.id_col = id_col

        st.success("Model trained and stored in session. Switch to the **Predict** tab to score a new CSV.")

# ------------------------------------------------------------------------------------------
# Predict tab
# ------------------------------------------------------------------------------------------
with tab_predict:
    st.subheader("Predict on a new applicants CSV")

    if not st.session_state.trained:
        st.info("Train a model first in the **Train** tab.")
    else:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            pred_up = st.file_uploader("Upload NEW applicants CSV", type=["csv"], key="pred_csv")

        with col_right:
            alpha = st.slider("Fusion weight α (tabular vs docs)", 0.0, 1.0, 0.7, 0.05)
            k_overall = st.number_input("Top-K overall", min_value=1, max_value=1000, value=5, step=1)

        # Optional ZIP of PDFs
        docs_zip = st.file_uploader(
            "Optional: ZIP of SOP/LOR/CV PDFs (e.g., 1234_SOP.pdf, 1234_LOR1.pdf, 1234_CV.pdf)",
            type=["zip"],
            key="pred_docs_zip",
        )

        doc_index = None
        if docs_zip is not None:
            doc_index = build_doc_index_from_zip(docs_zip.getvalue())
        # else: None -> scoring will skip docs

        if pred_up is not None:
            df_pred = pd.read_csv(pred_up)

            # check features presence
            missing_feats = [c for c in st.session_state.features if c not in df_pred.columns]
            # allow text concatenation column if used
            text_needed = "___TEXT___" in getattr(st.session_state.model["pre"], "feature_names_in_", [])
            if text_needed and "___TEXT___" not in df_pred.columns:
                # We'll re-concat if any text cols existed in training -> assume same in predict
                # We don't know the original text cols list here, so best-effort:
                obj_cols = [c for c in df_pred.columns if df_pred[c].dtype == "object"]
                if obj_cols:
                    df_pred = concat_text_columns(df_pred, obj_cols)
                    if "___TEXT___" not in df_pred.columns:
                        st.warning("Could not reconstruct text column for TF-IDF. Predictions may be off.")
                else:
                    st.warning("No object columns to rebuild TF-IDF. Predictions may be off.")

            if missing_feats:
                st.error(f"Prediction data is missing required features: {missing_feats}")
                st.stop()

            st.write("Preview")
            st.dataframe(df_pred.head(10), use_container_width=True)

            id_col = st.session_state.id_col
            if id_col is None or id_col not in df_pred.columns:
                st.warning("No ID column configured or present; doc matching by ID will be skipped.")

            # compute tabular probabilities
            model = st.session_state.model
            task = st.session_state.task

            Xp = df_pred[st.session_state.features if "___TEXT___" not in df_pred.columns else (list(set(st.session_state.features) - {"___TEXT___"}) | {"___TEXT___"})]
            try:
                p_tab = preds_to_probability(task, model, Xp)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            out = df_pred.copy()
            out["p_tabular"] = p_tab

            # doc scoring (optional)
            doc_scores = []
            if id_col and doc_index:
                for _, r in out.iterrows():
                    doc_scores.append(score_docs_for_row(str(r[id_col]), doc_index))
            else:
                doc_scores = [np.nan] * len(out)
            out["doc_score"] = doc_scores

            # fuse
            out["p_final"] = [
                fuse_scores(t, d, alpha) for t, d in zip(out["p_tabular"].values, out["doc_score"].values)
            ]

            # small histogram viz
            import matplotlib.pyplot as plt

            st.markdown("#### Score distribution (p_final)")
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.hist(out["p_final"].values, bins=12)
            ax.set_xlabel("p_final")
            ax.set_ylabel("count")
            st.pyplot(fig, use_container_width=True)

            # --------------------------- TOP-K (strictly from THIS PREDICTION CSV) ----------------------
            # overall
            st.markdown("### Top-K applicants (overall)")
            top_overall = out.sort_values("p_final", ascending=False).head(int(k_overall))
            # show main columns + scores
            show_cols = []
            # prefer common numeric columns
            for c in ["Serial No.", "Serial No", "Serial", "applicant_id", "ID", "id"]:
                if c in out.columns:
                    show_cols.append(c)
                    break
            score_cols = [c for c in out.columns if c not in ["p_tabular", "doc_score", "p_final"] and out[c].dtype.kind in "ifu"]
            # choose a short set to show
            base_cols = [c for c in score_cols if c not in show_cols][:8]
            show = list(dict.fromkeys(show_cols + base_cols + ["p_tabular", "doc_score", "p_final"]))
            st.dataframe(top_overall[show], use_container_width=True)

            # Group-wise Top-K (optional)
            st.markdown("### Optional: Top-K per group")
            grp_col = st.selectbox("Group by column (optional)", [None] + out.columns.tolist(), index=0)
            k_group = st.number_input("K per group", min_value=1, max_value=1000, value=1, step=1, key="k_group")
            if grp_col:
                gb = []
                for g, df_g in out.groupby(grp_col):
                    df_g = df_g.sort_values("p_final", ascending=False).head(int(k_group))
                    gb.append(df_g)
                if gb:
                    per_group = pd.concat(gb, axis=0)
                    st.dataframe(per_group[show], use_container_width=True)
                else:
                    st.info("No groups found.")

            st.caption("Top-K tables above are derived **only from the uploaded prediction CSV** (never from training rows).")
