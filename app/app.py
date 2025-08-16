# app/app.py
import io
import re
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Optional XGBoost if available
try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# PDFs
try:
    import PyPDF2
    HAS_PYPDF2 = True
except Exception:
    HAS_PYPDF2 = False


# ---------- Page config ----------
st.set_page_config(
    page_title="AdmitRank — Train • Predict • Explain",
    layout="wide",
    page_icon="🎓",
)

# ---------- Helpers ----------

def read_csv(upload) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()
    return pd.read_csv(upload)

def _avg_len(s: pd.Series) -> float:
    try:
        return float(np.nanmean(s.astype(str).str.len()))
    except Exception:
        return 0.0

def detect_id_column(df: pd.DataFrame) -> str:
    cand = ["Serial No.", "serial_no", "applicant_id", "id", "ID"]
    for c in cand:
        if c in df.columns:
            return c
    # fallback: first column
    return df.columns[0]

def detect_task(y: pd.Series) -> Tuple[str, bool]:
    """
    Returns ("classification" or "regression", is_binary_bool)
    """
    # string targets -> treat as classification
    if y.dtype == "object":
        uniq = y.dropna().unique()
        return ("classification", len(uniq) == 2)
    # numeric
    uniq = pd.unique(y.dropna())
    if len(uniq) <= 10:
        return ("classification", len(uniq) == 2)
    return ("regression", False)

def split_feature_types(df: pd.DataFrame, features: List[str]) -> Tuple[List[str], List[str], Optional[str]]:
    """
    Separate numeric, categorical, text.
    We auto-detect "long" text columns (avg length > ~25 or very high cardinality)
    and collapse them into a single TEXT_ALL column.
    """
    num_cols = []
    cat_cols = []
    text_cols = []
    for c in features:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            # treat as categorical for simplicity
            cat_cols.append(c)
        else:
            avglen = _avg_len(df[c])
            nunique = df[c].nunique(dropna=True)
            # heuristic: longer strings or very high cardinality -> text
            if avglen >= 25 or nunique > max(30, 0.5 * len(df[c])):
                text_cols.append(c)
            else:
                cat_cols.append(c)

    text_all_name = None
    if len(text_cols) > 0:
        text_all_name = "TEXT_ALL"
    return num_cols, cat_cols, text_all_name

def add_text_all(df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
    if not text_cols:
        return df
    dfx = df.copy()
    dfx["TEXT_ALL"] = dfx[text_cols].astype(str).agg(" . ".join, axis=1)
    return dfx

def build_preprocess_and_model(
    df: pd.DataFrame,
    target: str,
    id_col: str,
    model_family: str = "RandomForest",
    num_impute: str = "median",
    cat_impute: str = "most_frequent",
) -> Tuple[Pipeline, Dict]:
    """
    Builds and fits a pipeline on df (train split inside),
    stores metadata (feature types, label encoder when needed).
    """
    # features = all columns except id_col + target
    base_features = [c for c in df.columns if c not in [id_col, target]]

    num_cols, cat_cols, text_all_name = split_feature_types(df, base_features)
    text_cols = []
    # if have text_all
    if text_all_name:
        # identify source text columns
        text_cols = [
            c for c in base_features
            if (c not in num_cols and c not in cat_cols)
        ]
        df = add_text_all(df, text_cols)

    X = df[[c for c in base_features if c in (num_cols + cat_cols)]].copy()
    if text_all_name:
        X[text_all_name] = df[text_all_name]

    y = df[target].copy()

    task, is_binary = detect_task(y)

    # encode y for classification if categorical
    label_encoder = None
    if task == "classification" and y.dtype == "object":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.astype(str))

    # build preprocess
    transformers = []
    if num_cols:
        transformers.append(
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy=num_impute)),
                ("scaler", StandardScaler()),
            ]), num_cols)
        )
    if cat_cols:
        transformers.append(
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy=cat_impute)),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols)
        )
    if text_all_name:
        transformers.append(
            ("txt", Pipeline([
                # missing -> empty string
                ("imp", SimpleImputer(strategy="constant", fill_value="")),
                ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=5000)),
            ]), text_all_name)
        )

    pre = ColumnTransformer(transformers, remainder="drop")

    # choose estimator
    if task == "classification":
        if model_family == "Linear":
            est = LogisticRegression(max_iter=2000, n_jobs=None)
        elif model_family == "XGBoost" and HAS_XGB:
            est = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                eval_metric="logloss",
            )
        else:
            est = RandomForestClassifier(n_estimators=400, random_state=42)
    else:  # regression
        if model_family == "Linear":
            est = LinearRegression()
        elif model_family == "XGBoost" and HAS_XGB:
            est = XGBRegressor(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
            )
        else:
            est = RandomForestRegressor(n_estimators=400, random_state=42)

    pipe = Pipeline([("pre", pre), ("est", est)])

    # train/val split for a quick score
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y if task == "classification" else None
    )
    pipe.fit(X_train, y_train)

    # metrics
    metrics = {}
    if task == "classification":
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        try:
            metrics["ROC_AUC"] = float(roc_auc_score(y_test, prob))
        except Exception:
            metrics["ROC_AUC"] = float("nan")
        metrics["Accuracy"] = float(accuracy_score(y_test, pred))
        metrics["F1"] = float(f1_score(y_test, pred))
    else:
        pred = pipe.predict(X_test)
        metrics["MAE"] = float(mean_absolute_error(y_test, pred))
        metrics["RMSE"] = float(np.sqrt(mean_squared_error(y_test, pred)))
        metrics["R2"] = float(r2_score(y_test, pred))

    meta = dict(
        id_col=id_col,
        target=target,
        task=task,
        is_binary=is_binary,
        num_cols=num_cols,
        cat_cols=cat_cols,
        text_cols=text_cols,    # original long text cols used to build TEXT_ALL
        text_all_name="TEXT_ALL" if text_cols else None,
        model_family=model_family,
        label_encoder=label_encoder,
        metrics=metrics,
    )
    return pipe, meta


def pdf_to_text(file_bytes: bytes) -> str:
    if not HAS_PYPDF2:
        return ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        chunks = []
        for page in reader.pages:
            chunks.append(page.extract_text() or "")
        return "\n".join(chunks)
    except Exception:
        return ""

def score_text_simple(text: str) -> float:
    """
    Very light doc score in [0,1]:
    - normalize length
    - bonus for keywords
    """
    if not text:
        return 0.0
    t = text.strip()
    n = len(t)
    # normalize length ~2k chars good enough
    length_score = min(n / 2000.0, 1.0)
    kw = ["research", "intern", "publication", "project", "award", "experience"]
    kw_hits = sum(int(k in t.lower()) for k in kw)
    kw_score = min(kw_hits / 5.0, 1.0)
    return 0.7 * length_score + 0.3 * kw_score

PDF_REGEX = re.compile(r"^(\d+)[^/\\]*_(SOP|LOR\d*|CV)(?:\.\w+)?$", re.IGNORECASE)

def build_doc_index_from_zip(zip_file: Optional[io.BytesIO]) -> Dict[str, float]:
    """
    Returns {id_str: score_in_[0,1]}.
    Tolerant to any missing PDFs; unknown files ignored.
    """
    if zip_file is None:
        return {}
    idx: Dict[str, List[float]] = {}
    try:
        with zipfile.ZipFile(zip_file) as zf:
            for name in zf.namelist():
                base = name.split("/")[-1]
                m = PDF_REGEX.match(base)
                if not m:
                    continue
                key = m.group(1)  # applicant id as string
                data = zf.read(name)
                txt = pdf_to_text(data)
                sc = score_text_simple(txt)
                idx.setdefault(key, []).append(sc)
    except Exception:
        return {}
    # average per id
    out: Dict[str, float] = {}
    for k, vals in idx.items():
        if len(vals) == 0:
            continue
        out[k] = float(np.mean(vals))
    return out


def predict_on_new(
    pipe: Pipeline,
    meta: Dict,
    df_pred_raw: pd.DataFrame,
    doc_idx: Dict[str, float],
    alpha: float,
    group_col: Optional[str],
    top_k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (scored_prediction_dataframe, top_k_dataframe)
    Scoring uses only df_pred_raw (no training rows).
    """
    id_col = meta["id_col"]
    target = meta["target"]
    task = meta["task"]
    text_cols = meta["text_cols"]
    text_all_name = meta["text_all_name"]

    # Remove target if user accidentally included it in prediction file
    df_pred = df_pred_raw.copy()
    if target in df_pred.columns:
        df_pred = df_pred.drop(columns=[target])

    # Build TEXT_ALL if we used text during training
    if text_all_name:
        to_join = [c for c in text_cols if c in df_pred.columns]
        df_pred = add_text_all(df_pred, to_join)

    # X columns = everything except id_col (and no target col since removed)
    X_cols = [c for c in df_pred.columns if c != id_col]

    # tabular score
    if meta["task"] == "classification":
        p_tab = pipe.predict_proba(df_pred[X_cols])[:, 1]
    else:
        # For regression, normalize to [0,1] via min-max on predictions
        y_hat = pipe.predict(df_pred[X_cols])
        # guard degenerate case
        y_min, y_max = float(np.min(y_hat)), float(np.max(y_hat))
        if y_max > y_min:
            p_tab = (y_hat - y_min) / (y_max - y_min)
        else:
            p_tab = np.zeros_like(y_hat, dtype=float)

    # doc score from PDFs if available
    ids_as_str = df_pred[id_col].astype(str)
    p_doc = np.array([doc_idx.get(s, np.nan) for s in ids_as_str], dtype=float)

    # fusion
    p_final = p_tab.copy()
    mask = ~np.isnan(p_doc)
    p_final[mask] = alpha * p_tab[mask] + (1 - alpha) * p_doc[mask]

    scored = df_pred.copy()
    scored["p_tabular"] = p_tab
    scored["doc_score"] = p_doc
    scored["p_final"]  = p_final

    # sort by p_final (descending)
    scored_sorted = scored.sort_values("p_final", ascending=False)

    # Top-K overall or by group (always from prediction df only)
    if group_col and group_col in scored_sorted.columns:
        topk = (
            scored_sorted
            .groupby(group_col, group_keys=False)
            .head(top_k)
        )
    else:
        topk = scored_sorted.head(top_k)

    return scored_sorted, topk


# ---------- UI ----------

st.title("AdmitRank — Train • Predict • Explain")
st.write(
    "Upload any **historical CSV** to **train**, then predict on a **new CSV**. "
    "Optionally add a **ZIP of SOP/LOR/CV PDFs** named like "
    "`1234_SOP.pdf`, `1234_LOR1.pdf`, `1234_CV.pdf` "
    "(**extension is optional**). Missing PDFs are fine — we simply use the tabular score."
)

tab_train, tab_predict = st.tabs(["Train", "Predict"])


# --------- TRAIN TAB ----------
with tab_train:
    st.subheader("Train a model")

    train_csv = st.file_uploader("Upload historical training CSV", type=["csv"], key="train_csv")
    df_train = read_csv(train_csv)

    if not df_train.empty:
        st.caption("Preview (first 6 rows)")
        st.dataframe(df_train.head(6), use_container_width=True)

        id_col_default = detect_id_column(df_train)
        id_col = st.selectbox("ID column (matches PDFs)", options=list(df_train.columns), index=list(df_train.columns).index(id_col_default))

        target = st.selectbox("Target column", options=[c for c in df_train.columns if c != id_col])

        # Feature selection (default = all except id and target)
        default_feats = [c for c in df_train.columns if c not in [id_col, target]]
        features = st.multiselect("Feature columns", options=[c for c in df_train.columns if c not in [id_col, target]],
                                  default=default_feats)

        col1, col2, col3 = st.columns(3)
        with col1:
            model_family = st.selectbox("Model family", ["RandomForest", "Linear"] + (["XGBoost"] if HAS_XGB else []))
        with col2:
            num_strategy = st.selectbox("Numeric imputation", ["median", "mean"])
        with col3:
            cat_strategy = st.selectbox("Categorical imputation", ["most_frequent", "constant"])

        if st.button("Train", type="primary"):
            # keep only selected columns
            df_sub = df_train[[id_col, target] + features].copy()

            with st.spinner("Training..."):
                pipe, meta = build_preprocess_and_model(
                    df_sub,
                    target=target,
                    id_col=id_col,
                    model_family=model_family,
                    num_impute=num_strategy,
                    cat_impute=cat_strategy,
                )

            # keep in session
            st.session_state.model = pipe
            st.session_state.meta = meta

            st.success("Model trained and stored in session.")
            st.write("**Detected task:**", meta["task"])
            st.write("**Validation metrics (10% split):**")
            st.json(meta["metrics"])

            # Quick visualizations (train split metrics already above)
            st.markdown("### Feature snapshot")
            st.bar_chart(df_train[features].select_dtypes(include=np.number).head(200))

    else:
        st.info("Upload a training CSV to begin.")


# --------- PREDICT TAB ----------
with tab_predict:
    st.subheader("Predict on a new applicants CSV")

    if "model" not in st.session_state or "meta" not in st.session_state:
        st.warning("Train a model on the **Train** tab first.")
        st.stop()

    model = st.session_state.model
    meta = st.session_state.meta
    id_col_trained = meta["id_col"]

    pred_csv = st.file_uploader("Upload prediction CSV", type=["csv"], key="pred_csv")
    zip_docs = st.file_uploader("Optional: ZIP of SOP/LOR/CV PDFs", type=["zip"])

    alpha = st.slider("Fusion weight α (tabular vs doc)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    top_k = st.number_input("Top-K overall", min_value=1, max_value=1000, value=5, step=1)
    group_col = st.selectbox("Optional: return Top-K per group (choose a column or leave blank)", options=[""] + ([] if pred_csv is None else list(read_csv(pred_csv).columns)))
    group_col = None if group_col == "" else group_col

    if pred_csv is not None:
        df_pred = read_csv(pred_csv)

        if id_col_trained not in df_pred.columns:
            st.error(f"Your prediction CSV must contain the ID column used during training: **{id_col_trained}**.")
            st.stop()

        st.caption("Prediction file (first 6 rows)")
        st.dataframe(df_pred.head(6), use_container_width=True)

        if st.button("Run prediction", type="primary"):
            with st.spinner("Scoring..."):
                # Build doc index (tolerant to missing/partial ZIP)
                doc_idx = build_doc_index_from_zip(zip_docs)

                scored, topk = predict_on_new(
                    model, meta, df_pred, doc_idx, alpha, group_col, int(top_k)
                )

            # Small distribution viz
            st.markdown("### Score distribution (p_final)")
            st.bar_chart(pd.Series(np.round(scored["p_final"], 2)).value_counts().sort_index())

            st.markdown("### Predictions (with fusion & Top-K)")
            st.dataframe(topk[[c for c in topk.columns if c not in ["TEXT_ALL"]]], use_container_width=True)

            st.markdown("### Download scored predictions")
            st.download_button(
                label="Download full scored CSV",
                data=scored.to_csv(index=False).encode("utf-8"),
                file_name="scored_predictions.csv",
                mime="text/csv",
            )
    else:
        st.info("Upload a prediction CSV to score new applicants.")
