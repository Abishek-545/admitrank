# app.py
# AdmitRank – Train • Predict • Explain (final)

import io
import re
import zipfile
from pathlib import PurePath
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

try:
    from xgboost import XGBClassifier, XGBRegressor  # optional
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    from PyPDF2 import PdfReader
    PDF_OK = True
except Exception:
    PDF_OK = False


# ------------------------------
# ---------- Utilities ----------
# ------------------------------
st.set_page_config(page_title="AdmitRank — Train • Predict • Explain", layout="wide")

def read_table(upload) -> pd.DataFrame:
    """Robust CSV/Excel reader for Streamlit UploadedFile with buffer rewind."""
    if upload is None:
        return pd.DataFrame()
    # Always rewind the buffer
    try:
        upload.seek(0)
    except Exception:
        pass

    # Try CSV first
    try:
        df = pd.read_csv(upload, engine="python")
        return df
    except pd.errors.EmptyDataError:
        # Fall through to Excel attempt
        pass
    except Exception:
        pass

    # Rewind and try Excel
    try:
        upload.seek(0)
        df = pd.read_excel(upload)
        return df
    except Exception:
        return pd.DataFrame()


def is_binary_series(s: pd.Series) -> bool:
    vals = pd.Series(s.dropna().unique())
    if len(vals) <= 1:
        return False
    if vals.dtype.kind in "ifb":
        # treat as binary if only two distinct numbers (e.g., 0/1)
        return len(vals) == 2
    # strings like YES/NO
    v = vals.astype(str).str.upper().str.strip()
    return len(v.unique()) == 2


def infer_task(y: pd.Series) -> str:
    """Return 'classification' or 'regression'."""
    y_clean = y.dropna()
    # Common patterns for classification
    if y_clean.dtype.kind in "b":
        return "classification"
    if y_clean.dtype.kind in "if" and len(y_clean.unique()) <= 10:
        # small finite set of numbers often indicates classes
        return "classification"
    if y_clean.dtype.kind in "O":
        return "classification"
    # otherwise
    return "regression"


def build_estimator(task: str, family: str):
    """Return an (estimator, needs_scaling) tuple."""
    family = family.lower()
    if task == "classification":
        if family == "linear":
            return LogisticRegression(max_iter=200, n_jobs=None), True
        if family == "tree":
            return RandomForestClassifier(n_estimators=300, random_state=42), False
        if family == "svm":
            return SVC(probability=True, kernel="rbf"), True
        if family == "xgboost" and XGB_OK:
            return XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric="logloss",
            ), False
        # fallback
        return LogisticRegression(max_iter=200), True

    # regression
    if family == "linear":
        return LinearRegression(), True
    if family == "tree":
        return RandomForestRegressor(n_estimators=300, random_state=42), False
    if family == "svm":
        return SVR(kernel="rbf"), True
    if family == "xgboost" and XGB_OK:
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        ), False
    return RandomForestRegressor(n_estimators=300, random_state=42), False


def split_train_eval(
    df: pd.DataFrame,
    features: list,
    target: str,
    family: str,
    test_size: float = 0.1,
    random_state: int = 42,
):
    """Train/test split + pipeline training + basic metrics + permutation importances."""
    y = df[target]
    task = infer_task(y)

    # Prepare label encoder if classification & non-numeric labels
    le = None
    y_train = y.copy()
    if task == "classification" and y_train.dtype.kind not in "ifb":
        le = LabelEncoder()
        y_train = le.fit_transform(y_train.astype(str))

    X = df[features].copy()
    # Identify types
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    est, needs_scaling = build_estimator(task, family)

    num_pipe = [("imputer", SimpleImputer(strategy="median"))]
    if needs_scaling:
        num_pipe.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_pipe)

    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )

    pipe = Pipeline(
        [
            ("pre", pre),
            ("model", est),
        ]
    )

    stratify = y_train if task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_train, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    metrics = {}
    if task == "classification":
        # proba if available
        try:
            proba = pipe.predict_proba(X_te)[:, -1]
        except Exception:
            proba = None
        metrics["Accuracy"] = accuracy_score(y_te, y_pred)
        if proba is not None and len(np.unique(y_te)) == 2:
            metrics["ROC-AUC"] = roc_auc_score(y_te, proba)
        metrics["F1"] = f1_score(y_te, y_pred, average="weighted")
    else:
        metrics["R2"] = r2_score(y_te, y_pred)
        metrics["MAE"] = mean_absolute_error(y_te, y_pred)
        metrics["RMSE"] = np.sqrt(mean_squared_error(y_te, y_pred))

    # Permutation importance
    imp_df = pd.DataFrame()
    try:
        perm = permutation_importance(pipe, X_te, y_te, n_repeats=5, random_state=42)
        # Build feature names after OHE automatically; but it's tricky to map.
        # Instead, compute importance on original features by permuting columns manually:
        # Simpler: show model-agnostic "drop-column" style using permutation_importance on original data
        # However ColumnTransformer makes mapping complex. We'll just show raw importances from permutation.
        # We'll aggregate importances per original column by permuting each col alone:
        agg_importance = []
        for col in X.columns:
            X_te_perm = X_te.copy()
            X_te_perm[col] = np.random.permutation(X_te_perm[col].values)
            score_base = pipe.score(X_te, y_te)
            score_perm = pipe.score(X_te_perm, y_te)
            agg_importance.append((col, score_base - score_perm))
        imp_df = pd.DataFrame(agg_importance, columns=["feature", "importance"]).sort_values(
            "importance", ascending=False
        )
    except Exception:
        pass

    # For returning p scaling in regression
    y_min, y_max = None, None
    if task == "regression":
        y_min = float(pd.Series(y).min())
        y_max = float(pd.Series(y).max())

    model_pack = {
        "pipeline": pipe,
        "task": task,
        "family": family,
        "label_encoder": le,
        "features": features,
        "target": target,
        "y_min": y_min,
        "y_max": y_max,
    }
    return model_pack, metrics, imp_df


def ensure_model() -> bool:
    ok = all(k in st.session_state for k in ("model", "fitted"))
    return ok


def safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def extract_id_from_name(name: str) -> Optional[str]:
    """
    Accept <id>_<TYPE>.pdf (extension optional).
    Returns the <id> as string if found, else None.
    """
    base = PurePath(name).name
    # allow ".pdf" optional
    m = re.match(r"^([^_]+)_(SOP|LOR\d*|CV)(?:\.[pP][dD][fF])?$", base)
    if not m:
        return None
    return m.group(1)


def pdf_to_text(pdf_bytes: bytes) -> str:
    if not PDF_OK:
        return ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        txt = []
        for page in reader.pages:
            try:
                txt.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(txt)
    except Exception:
        return ""


def simple_doc_score(text: str) -> float:
    """
    Very lightweight quality proxy in [0, 1]:
      - length (characters),
      - vocabulary richness (unique words / total words).
    """
    if not text:
        return 0.0
    chars = len(text)
    words = re.findall(r"[A-Za-z']+", text)
    total_words = max(1, len(words))
    unique_words = len(set(w.lower() for w in words))
    richness = unique_words / total_words  # 0..1

    # Length normalization (capped at ~1500 chars)
    length_score = min(1.0, chars / 1500.0)

    # Combine
    score = 0.6 * length_score + 0.4 * richness
    return float(np.clip(score, 0, 1))


def build_doc_index_from_zip(upload_zip) -> Dict[str, float]:
    """
    Parse ZIP of PDFs and compute mean doc score per id.
    Missing/invalid files are ignored gracefully.
    """
    if upload_zip is None:
        return {}
    try:
        upload_zip.seek(0)
    except Exception:
        pass

    try:
        with zipfile.ZipFile(upload_zip) as zf:
            scores: Dict[str, list] = {}
            for name in zf.namelist():
                id_ = extract_id_from_name(name)
                if id_ is None:
                    continue
                try:
                    data = zf.read(name)
                except Exception:
                    continue
                text = pdf_to_text(data)
                s = simple_doc_score(text)
                scores.setdefault(id_, []).append(s)
            # average per id
            return {k: float(np.mean(v)) for k, v in scores.items() if v}
    except Exception:
        return {}


def proba_from_model(model_pack: dict, df_pred: pd.DataFrame) -> np.ndarray:
    """
    Return an array of probabilities/scores in [0, 1] for the *prediction set only*.
    Binary classification: positive class probability.
    Regression: min-max scale predictions using train target range.
    """
    pipe = model_pack["pipeline"]
    features = model_pack["features"]
    task = model_pack["task"]
    y_min = model_pack.get("y_min", None)
    y_max = model_pack.get("y_max", None)

    Xp = df_pred[features].copy()
    if task == "classification":
        try:
            proba = pipe.predict_proba(Xp)[:, -1]
        except Exception:
            # fallback: decision_function -> sigmoid-ish
            try:
                d = pipe.decision_function(Xp)
                proba = 1 / (1 + np.exp(-d))
            except Exception:
                proba = pipe.predict(Xp)
                # force to [0,1]
                proba = (proba - np.min(proba)) / (np.ptp(proba) + 1e-9)
        return np.clip(proba, 0, 1)
    else:
        pred = pipe.predict(Xp).astype(float)
        # normalize to 0..1 using training range if available
        if y_min is None or y_max is None or y_max <= y_min:
            mn, mx = float(np.min(pred)), float(np.max(pred))
        else:
            mn, mx = float(y_min), float(y_max)
        proba = (pred - mn) / (max(1e-9, mx - mn))
        return np.clip(proba, 0, 1)


# ------------------------------
# --------- Page Header --------
# ------------------------------
st.markdown(
    """
    <h1 style="margin-bottom:0">AdmitRank — Train • Predict • Explain</h1>
    <p style="color:#aaa">
      Upload any <b>historical CSV</b> to <b>train</b>, then predict on a <b>new CSV</b>.
      Optionally add a ZIP of SOP/LOR/CV PDFs named like
      <code>1234_SOP.pdf</code>, <code>1234_LOR1.pdf</code>, <code>1234_CV.pdf</code>
      (extension is optional).
      <br/>Missing PDFs are OK — the system will simply use the tabular model.
    </p>
    """,
    unsafe_allow_html=True,
)

tab_train, tab_predict = st.tabs(["Train", "Predict"])

# ------------------------------
# ----------- TRAIN ------------
# ------------------------------
with tab_train:
    st.subheader("Train a model")

    hist_upload = st.file_uploader(
        "Upload historical training CSV",
        type=["csv", "xlsx", "xls"],
        key="hist_csv",
        help="This is the dataset used to fit the model.",
    )

    if hist_upload is not None:
        df_hist = read_table(hist_upload)
        if df_hist.empty:
            st.error("Could not read the uploaded file. Please check the format.")
        else:
            st.dataframe(df_hist.head(), use_container_width=True)
            # Choose target and features
            cols = list(df_hist.columns)
            target = st.selectbox("Target column", options=cols, index=len(cols) - 1)
            feature_default = [c for c in cols if c != target]
            features = st.multiselect(
                "Feature columns",
                options=cols,
                default=feature_default,
                help="Choose the columns you want to use as features.",
            )
            st.caption("Tip: You can include categorical columns; they'll be one-hot encoded automatically.")

            colA, colB, colC = st.columns([1, 1, 1])
            with colA:
                family = st.selectbox("Model family", options=["Linear", "Tree", "SVM", "XGBoost" if XGB_OK else "Linear"])
            with colB:
                test_size = st.slider("Test size (hold-out)", 0.05, 0.3, 0.10, 0.05)
            with colC:
                random_state = st.number_input("Random seed", 0, 9999, 42)

            if st.button("Train", type="primary", use_container_width=False):
                if not features:
                    st.warning("Please select at least one feature.")
                else:
                    model_pack, metrics, imp_df = split_train_eval(
                        df_hist, features, target, family, test_size=test_size, random_state=random_state
                    )
                    st.session_state["model"] = model_pack
                    st.session_state["fitted"] = True

                    st.success("Model trained and stored in session. You can switch to the Predict tab now.")

                    st.markdown("### Validation metrics")
                    mcols = st.columns(len(metrics) or 1)
                    for (k, v), c in zip(metrics.items(), mcols):
                        with c:
                            st.metric(k, f"{v:.4f}")

                    # Simple target distribution
                    st.markdown("### Target distribution")
                    with st.container():
                        if infer_task(df_hist[target]) == "classification":
                            counts = df_hist[target].value_counts()
                            st.bar_chart(counts, use_container_width=True)
                        else:
                            st.histogram(df_hist[target], bins=30, use_container_width=True)

                    # Feature importances
                    if not imp_df.empty:
                        st.markdown("### Feature importance (permutation) — higher is better")
                        st.dataframe(imp_df, use_container_width=True)
                        st.bar_chart(
                            imp_df.set_index("feature").head(15),
                            use_container_width=True,
                        )


# ------------------------------
# ---------- PREDICT -----------
# ------------------------------
with tab_predict:
    st.subheader("Predict on a new applicants CSV")

    if not ensure_model():
        st.info("Train a model in the **Train** tab first.")
    else:
        model_pack = st.session_state["model"]
        features = model_pack["features"]
        target = model_pack["target"]

        st.markdown(
            f"**This model expects the following features:** `{', '.join(features)}`"
        )

        pred_upload = st.file_uploader(
            "Upload new applicants CSV",
            type=["csv", "xlsx", "xls"],
            key="pred_csv",
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            zip_upload = st.file_uploader(
                "Optional: Upload a ZIP with SOP/LOR/CV PDFs (named like 1234_SOP.pdf, 1234_LOR1.pdf, 1234_CV.pdf). Missing files are fine.",
                type=["zip"],
                key="docs_zip",
            )
        with col2:
            alpha = st.slider("Fusion weight α (tabular vs doc)", 0.0, 1.0, 0.3, 0.05,
                             help="p_final = α·p_tabular + (1−α)·doc_score")

        # Read once & reuse
        df_pred = read_table(pred_upload) if pred_upload is not None else pd.DataFrame()

        if not df_pred.empty:
            st.markdown("#### Preview (first 10 rows)")
            st.dataframe(df_pred.head(10), use_container_width=True)

            # Identify id column for doc matching
            # Heuristics: choose a numeric/integer column or an 'id' look-alike
            id_candidates = [c for c in df_pred.columns if re.search(r"id|serial", c, re.I)]
            if not id_candidates:
                # try to find a unique-ish numeric integer column
                for c in df_pred.columns:
                    if pd.api.types.is_integer_dtype(df_pred[c]) and df_pred[c].nunique() == len(df_pred):
                        id_candidates.append(c)
            id_col = st.selectbox(
                "Key column for matching docs (the <id> part in 1234_SOP.pdf)",
                options=id_candidates if id_candidates else list(df_pred.columns),
            )

            # Group for optional Top-K per group
            group_options = [""] + list(df_pred.columns)
            k_group = st.selectbox("Optional: return Top-K per group (choose a column or leave blank)", options=group_options)
            group_col = None if k_group == "" else k_group

            k_value = st.number_input("K per group (or overall if no group)", 1, 500, 5)

            # Compute tabular proba strictly from prediction set
            try:
                p_tabular = proba_from_model(model_pack, df_pred)
            except Exception as e:
                st.error(f"Could not score the prediction data with the trained model. {e}")
                st.stop()

            # Compute doc index if ZIP provided
            doc_index = build_doc_index_from_zip(zip_upload) if zip_upload is not None else {}

            # Derive doc score vector on df_pred order (tolerant to missing)
            ids_as_str = df_pred[id_col].astype(str).fillna("")
            doc_scores = np.array([doc_index.get(_id, np.nan) for _id in ids_as_str])
            doc_scores_nan_to_zero = np.nan_to_num(doc_scores, nan=np.nan)  # keep NaN for display, but later handle fusion

            # Fusion
            # If doc score is NaN -> use tabular only for that row: p_final = p_tabular
            p_final = p_tabular.copy()
            if np.isfinite(doc_scores).any():
                ds = np.where(np.isfinite(doc_scores), doc_scores, p_tabular)
                p_final = alpha * p_tabular + (1 - alpha) * ds

            # Build result frame (prediction set only)
            out = df_pred.copy()
            out["p_tabular"] = p_tabular
            out["doc_score"] = [None if not np.isfinite(v) else float(v) for v in doc_scores]
            out["p_final"] = p_final

            # ----------------- Visualizations -----------------
            st.markdown("### Probability distribution on prediction set")
            st.histogram(out["p_final"], bins=20, use_container_width=True)

            # ----------------- Top-K selection -----------------
            st.markdown("### Top-K applicants")

            if group_col is None:
                # Overall Top-K using p_final
                topk = out.sort_values("p_final", ascending=False).head(int(k_value))
                st.dataframe(topk, use_container_width=True)
            else:
                # Per group
                def take_topk(g):
                    return g.sort_values("p_final", ascending=False).head(int(k_value))

                grp = out.groupby(group_col, group_keys=False)
                topk = grp.apply(take_topk)
                st.dataframe(topk, use_container_width=True)

            # Summaries by group (optional)
            st.markdown("### Optional: Mean p_final by group")
            if group_col is None:
                mean_pf = out["p_final"].mean()
                st.write(f"Mean p_final (overall): **{mean_pf:.4f}**")
            else:
                gp = out.groupby(group_col)["p_final"].mean().sort_values(ascending=False)
                st.bar_chart(gp, use_container_width=True)

        else:
            st.info("Upload a prediction CSV to score applicants.")


# ------------------------------
# --------- Footer note --------
# ------------------------------
st.caption(
    "Built to be robust and simple: tolerant to missing PDFs, uses only the prediction set for Top-K, "
    "and provides lightweight, fast visualizations."
)
