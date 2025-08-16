# app/app.py
import os, io, re, zipfile
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# ==== sklearn / ML ====
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, r2_score, mean_squared_error,
    confusion_matrix, roc_curve, auc
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
    XGB_OK = True
except Exception:
    XGB_OK = False

# ==== NLP for docs ====
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# ==== PDF extraction (graceful fallback) ====
_PYPDF2_OK = True
try:
    from PyPDF2 import PdfReader
except Exception:
    _PYPDF2_OK = False

st.set_page_config(page_title="AdmitRank — Train • Predict • Explain", layout="wide")
st.title("AdmitRank — Train • Predict • Explain")
st.caption(
    "Train on any historical CSV (numeric + categorical + text). Predict on a new CSV. "
    "Optionally add a ZIP with **SOP / LOR / CV PDFs** named like `<KEY>_SOP.pdf`, `<KEY>_LOR1.pdf`, `<KEY>_CV.pdf` "
    "where `<KEY>` matches a column you choose (e.g., `Serial No.`)."
)

# =========================================================
# Helpers
# =========================================================
def read_csv_any(file) -> pd.DataFrame:
    encodings = ("utf-8", "cp1252", "latin1")
    if hasattr(file, "read"):
        data = file.read()
        for enc in encodings:
            try:
                return pd.read_csv(io.StringIO(data.decode(enc)))
            except UnicodeDecodeError:
                continue
        return pd.read_csv(io.StringIO(data.decode("latin1", errors="ignore")))
    for enc in encodings:
        try:
            return pd.read_csv(file, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(file, encoding="latin1", errors="ignore")

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df

# ---- CSV feature typing (for automatic preprocessing) ----
def looks_like_multilabel(series: pd.Series) -> bool:
    if series.dtype == object:
        s = series.dropna().astype(str)
        if len(s) == 0: return False
        return s.str.contains(r"[;,|]").mean() >= 0.2
    return False

def looks_like_longtext(series: pd.Series) -> bool:
    if series.dtype == object:
        s = series.dropna().astype(str)
        if len(s) == 0: return False
        return (s.str.len().mean() >= 40) and (not looks_like_multilabel(series))
    return False

class MultiLabelBinarizerCol(BaseEstimator, TransformerMixin):
    def __init__(self, sep_pattern=r"[;,|]", lowercase=True, strip=True, min_freq=1, prefix=None):
        self.sep_pattern = sep_pattern; self.lowercase = lowercase; self.strip = strip
        self.min_freq = min_freq; self.prefix = prefix; self.classes_ = None
    def fit(self, X, y=None):
        col = pd.Series(X.iloc[:,0] if isinstance(X, pd.DataFrame) else np.array(X).ravel()).fillna("")
        tokens=[]
        for val in col.astype(str):
            parts=re.split(self.sep_pattern,val)
            for p in parts:
                t=p.strip() if self.strip else p
                if self.lowercase: t=t.lower()
                if t: tokens.append(t)
        if tokens:
            vc=pd.Series(tokens).value_counts()
            self.classes_=vc[vc>=self.min_freq].index.tolist()
        else:
            self.classes_=[]
        return self
    def transform(self, X):
        col = pd.Series(X.iloc[:,0] if isinstance(X, pd.DataFrame) else np.array(X).ravel()).fillna("")
        out = np.zeros((len(col), len(self.classes_)), dtype=float)
        m={c:i for i,c in enumerate(self.classes_)}
        for i,val in enumerate(col.astype(str)):
            parts=re.split(self.sep_pattern,val)
            for p in parts:
                t=p.strip().lower()
                if t in m: out[i,m[t]]=1.0
        return out
    def get_feature_names_out(self, input_features=None):
        prefix = self.prefix or (input_features[0] if input_features else "ml")
        return np.array([f"{prefix}__{c}" for c in (self.classes_ or [])], dtype=object)

class ColumnSelector1D(BaseEstimator, TransformerMixin):
    def __init__(self, col_name: str): self.col_name=col_name
    def fit(self, X, y=None): return self
    def transform(self, X):
        if isinstance(X,pd.DataFrame): return X[self.col_name].astype(str).fillna("").values
        arr=np.array(X); 
        return arr[:,0].astype(str) if (arr.ndim==2 and arr.shape[1]>=1) else arr.astype(str)

def build_preprocessor(df: pd.DataFrame, feature_cols, num_strategy="median", cat_strategy="most_frequent"):
    X = df[feature_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = [c for c in X.columns if c not in num_cols]
    multi_cols = [c for c in obj_cols if looks_like_multilabel(X[c])]
    long_cols  = [c for c in obj_cols if looks_like_longtext(X[c])]
    cat_cols   = [c for c in obj_cols if c not in set(multi_cols)|set(long_cols)]

    transformers=[]
    if num_cols: transformers.append(("num", SimpleImputer(strategy=num_strategy), num_cols))
    if cat_cols: transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse=True), cat_cols))
    for c in multi_cols:
        transformers.append((f"mlb__{c}", Pipeline([("imp",SimpleImputer(strategy="constant",fill_value="")),("mlb",MultiLabelBinarizerCol(prefix=c))]), [c]))
    for c in long_cols:
        transformers.append((f"tfidf__{c}", Pipeline([("sel",ColumnSelector1D(c)),("tfidf",TfidfVectorizer(max_features=5000,ngram_range=(1,2),min_df=2))]), [c]))
    if not transformers: transformers.append(("fallback", SimpleImputer(strategy=num_strategy), feature_cols))
    ct = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    return ct, {"num":num_cols,"cat":cat_cols,"multi":multi_cols,"long":long_cols}

# ---- Task detection & label encoding ----
def detect_task_and_encode(y_raw: pd.Series):
    """
    Decide binary / multiclass / regression robustly.
    - Two unique labels of any type => binary (map to 0/1)
    - 3..15 discrete unique labels => multiclass
    - Otherwise => regression
    """
    y = pd.Series(y_raw)
    # Try numeric conversion
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.notna().mean() == 1.0:
        uniq = np.unique(y_num.dropna())
        if len(uniq) == 2:             # binary numeric (e.g., 0/1 or 1/2)
            lo, hi = float(np.min(uniq)), float(np.max(uniq))
            y_bin = (y_num == hi).astype(int)
            return "binary", y_bin, {"mapping": (lo, hi)}
        if np.all(np.equal(np.mod(uniq,1),0)) and 3 <= len(uniq) <= 15:
            le = LabelEncoder()
            y_enc = le.fit_transform(y_num.astype(int))
            return "multiclass", pd.Series(y_enc), {"label_encoder": le}
        return "regression", y_num.astype(float), {}
    else:
        # Non-numeric labels (e.g., YES/NO, Admit/Reject, A/B/C)
        uniq = pd.Series(y.dropna().astype(str)).unique()
        if len(uniq) == 2:              # binary categorical
            le = LabelEncoder()
            y_enc = le.fit_transform(y.astype(str))
            return "binary", pd.Series(y_enc), {"label_encoder": le}
        # multiclass categorical (limit classes)
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        if len(le.classes_) <= 15:
            return "multiclass", pd.Series(y_enc), {"label_encoder": le}
        # too many classes -> treat as regression fallback (rare)
        y_num2 = pd.to_numeric(y, errors="coerce").fillna(0.0)
        return "regression", y_num2, {}

# ---- Models ----
ALGOS = ["RandomForest", "Linear", "XGBoost", "SVM", "KNN", "GradientBoosting"]

def build_estimator(family: str, task: str, random_state: int = 42):
    is_class = task in ("binary","multiclass")
    if family == "RandomForest":
        return RandomForestClassifier(n_estimators=300, random_state=random_state) if is_class \
               else RandomForestRegressor(n_estimators=300, random_state=random_state)
    if family == "Linear":
        return LogisticRegression(max_iter=2000, multi_class="auto") if is_class else Ridge(alpha=1.0)
    if family == "XGBoost" and XGB_OK:
        return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                             colsample_bytree=0.9, eval_metric="mlogloss" if task=="multiclass" else "logloss") \
               if is_class else \
               XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                            colsample_bytree=0.9, objective="reg:squarederror")
    if family == "SVM":
        # IMPORTANT: probability=True for proba in classification
        return SVC(kernel="rbf", probability=True) if is_class else SVR(kernel="rbf")
    if family == "KNN":
        return KNeighborsClassifier(n_neighbors=7) if is_class else KNeighborsRegressor(n_neighbors=7)
    if family == "GradientBoosting":
        return GradientBoostingClassifier(random_state=random_state) if is_class \
               else GradientBoostingRegressor(random_state=random_state)
    return LogisticRegression(max_iter=2000, multi_class="auto") if is_class else Ridge(alpha=1.0)

def evaluate_split(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, task: str):
    if task == "binary":
        proba = pipe.predict_proba(X_test)[:,1]
        y_pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        auc_val = roc_auc_score(y_test, proba)
        return {"task":"binary", "accuracy":acc, "roc_auc":auc_val, "y_pred":y_pred, "proba":proba}
    if task == "multiclass":
        proba = pipe.predict_proba(X_test)
        y_pred = np.argmax(proba, axis=1)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        return {"task":"multiclass", "accuracy":acc, "f1_macro":f1m, "y_pred":y_pred, "proba":proba}
    # regression
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {"task":"regression", "r2":r2, "rmse":rmse, "y_pred":y_pred}

def train_and_evaluate(df, features, target, family, num_strategy, cat_strategy, test_size=0.1, random_state=42):
    X = df[features].copy()
    task, y_enc, label_info = detect_task_and_encode(df[target])

    preproc, schema = build_preprocessor(df, features, num_strategy, cat_strategy)
    if family == "XGBoost" and not XGB_OK:
        st.warning("XGBoost not installed; using Linear model instead."); family = "Linear"
    est = build_estimator(family, task)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state,
        stratify=y_enc if task in ("binary","multiclass") else None
    )

    pipe = Pipeline([("preproc", preproc), ("model", est)])
    pipe.fit(X_tr, y_tr)
    metrics = evaluate_split(pipe, X_te, y_te, task)
    metrics["y_test"] = y_te.values

    pipe_full = Pipeline([("preproc", preproc), ("model", est)])
    pipe_full.fit(X, y_enc)

    return pipe_full, task, schema, metrics, family, label_info

# ---- Probability-like output for fusion ----
def predict_prob_like(model, task, X_df):
    if task == "binary":
        proba = model.predict_proba(X_df)[:,1]
        return proba.astype(float)
    if task == "multiclass":
        P = model.predict_proba(X_df)
        return np.max(P, axis=1).astype(float)  # confidence of predicted class
    preds = model.predict(X_df).astype(float)
    return np.clip(preds, 0.0, 1.0)

# ---- Explainability (best-effort) ----
def get_feature_names_from_ct(ct: ColumnTransformer):
    names=[]
    try:
        for name, trans, cols in ct.transformers_:
            if name=="remainder" and trans=="drop": continue
            if hasattr(trans,"get_feature_names_out"):
                base=cols if isinstance(cols,list) else []
                try: out = trans.get_feature_names_out(base)
                except Exception: out = trans.get_feature_names_out()
                names.extend(list(out))
            elif isinstance(trans,Pipeline):
                last=trans.steps[-1][1]
                if hasattr(last,"get_feature_names_out"):
                    try: out = last.get_feature_names_out()
                    except Exception: out = last.get_feature_names_out(None)
                    names.extend([f"{name}__{t}" for t in out])
                else:
                    if isinstance(cols,list): names.extend(cols)
            else:
                if isinstance(cols,list): names.extend(cols)
    except Exception: return None
    return names or None

def model_feature_importance(pipe: Pipeline):
    try:
        ct: ColumnTransformer = pipe.named_steps["preproc"]
        est = pipe.named_steps["model"]
        fn = get_feature_names_from_ct(ct)
        if hasattr(est,"feature_importances_"):
            vals = est.feature_importances_
            return (fn, vals) if fn is not None and len(fn)==len(vals) else None
        if hasattr(est,"coef_"):
            coef = est.coef_.ravel()
            vals = np.abs(coef)
            return (fn, vals) if fn is not None and len(fn)==len(vals) else None
    except Exception:
        pass
    return None

# =========================================================
# Documents (SOP/LOR/CV) — PDF ONLY with key-column matching
# =========================================================

# File name patterns: <key>_SOP.pdf, <key>_LOR1.pdf, <key>_CV.pdf (case-insensitive)
SOP_RE = re.compile(r"^([a-z0-9_\-\.]+)_sop\.pdf$", re.IGNORECASE)
LOR_RE = re.compile(r"^([a-z0-9_\-\.]+)_lor(\d+)\.pdf$", re.IGNORECASE)
CV_RE  = re.compile(r"^([a-z0-9_\-\.]+)_cv\.pdf$",  re.IGNORECASE)

def normalize_key(x: object) -> str:
    """Normalize any value to a safe key so filenames and CSV values match more robustly."""
    if pd.isna(x): return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", "_", s)                  # spaces -> underscores
    s = re.sub(r"[^a-z0-9_\-\.]+", "", s)       # keep alnum/_/-/.
    return s

def pdf_bytes_to_text(b: bytes) -> str:
    if not _PYPDF2_OK:
        return ""
    try:
        reader = PdfReader(io.BytesIO(b))
        texts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            texts.append(txt)
        return "\n".join(texts)
    except Exception:
        return ""

def _read_text_from_zip(zf, name) -> str:
    raw = zf.read(name)
    return pdf_bytes_to_text(raw)

def build_doc_index_from_zip(uploaded_zip_file):
    """
    Returns: dict[key] = {"sop": str|None, "lors": [str], "cv": str|None}
    Where key is the normalized key derived from file names.
    """
    idx={}
    buf=io.BytesIO(uploaded_zip_file.read())
    with zipfile.ZipFile(buf) as zf:
        for name in zf.namelist():
            if name.endswith("/"): continue
            base=Path(name).name; low=base.lower()
            m_sop=SOP_RE.match(low); m_lor=LOR_RE.match(low); m_cv=CV_RE.match(low)
            if not (m_sop or m_lor or m_cv): continue
            if m_sop:
                key,_=m_sop.groups()
                key = normalize_key(key)
                idx.setdefault(key,{"sop":None,"lors":[], "cv":None}); idx[key]["sop"]=_read_text_from_zip(zf,name)
            elif m_lor:
                key,_,_=m_lor.groups()
                key = normalize_key(key)
                idx.setdefault(key,{"sop":None,"lors":[], "cv":None}); idx[key]["lors"].append(_read_text_from_zip(zf,name))
            else:
                key,_=m_cv.groups()
                key = normalize_key(key)
                idx.setdefault(key,{"sop":None,"lors":[], "cv":None}); idx[key]["cv"]=_read_text_from_zip(zf,name)
    return idx

DOC_WEIGHTS={"words":0.10,"paras":0.05,"avg_sentence_len":0.10,"flesch":0.20,"fkgrade":0.05,"sentiment":0.10,"rep_rate":-0.10}
def sop_lor_features(text:str)->dict:
    t=" ".join(text.split()); words=t.split(); n=len(words)
    paras=max(1,text.count("\n\n")+1); avg=textstat.avg_sentence_length(t) if n else 0.0
    fle=textstat.flesch_reading_ease(t) if n else 0.0; fk=textstat.flesch_kincaid_grade(t) if n else 0.0
    sent=analyzer.polarity_scores(t)["compound"] if n else 0.0
    toks=re.findall(r"[a-zA-Z]+",t.lower()); rep=0.0
    if len(toks)>2:
        c=Counter(list(zip(toks,toks[1:])))
        rep=max(c.values())/max(1,len(toks)-1)
    return {"words":n,"paras":paras,"avg_sentence_len":avg,"flesch":fle,"fkgrade":fk,"sentiment":sent,"rep_rate":rep}
def score_from_features(f:dict)->float:
    return float(np.clip(sum(DOC_WEIGHTS[k]*f.get(k,0.0) for k in DOC_WEIGHTS),0.0,1.0))
def fuse_scores(p_tab,p_doc,alpha): return float(p_tab) if np.isnan(p_doc) else float(alpha*p_tab+(1-alpha)*p_doc)

# =========================================================
# Plot helpers (matplotlib)
# =========================================================
def plot_confusion(y_true, y_pred, normalize=False, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title); plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for (i,j),z in np.ndenumerate(cm):
        ax.text(j, i, f"{z:.2f}" if normalize else int(z), ha="center", va="center", fontsize=9)
    st.pyplot(fig, use_container_width=True)

def plot_roc(y_true, proba):
    fpr, tpr, _ = roc_curve(y_true, proba)
    AUC = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, lw=2, label=f"ROC AUC={AUC:.3f}")
    ax.plot([0,1], [0,1], "--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right"); ax.set_title("ROC curve")
    st.pyplot(fig, use_container_width=True)

def plot_hist(values, title, bins=30, xlabel="value"):
    fig, ax = plt.subplots(figsize=(5,4))
    ax.hist(values, bins=bins, edgecolor="black")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("count")
    st.pyplot(fig, use_container_width=True)

def plot_parity(y_true, y_pred, title="Parity plot: ŷ vs y"):
    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(y_true, y_pred, alpha=0.6)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=2)
    ax.set_xlabel("True"); ax.set_ylabel("Predicted"); ax.set_title(title)
    st.pyplot(fig, use_container_width=True)

def plot_bar(names, values, title, top=25):
    idx = np.argsort(values)[::-1][:top]
    fig, ax = plt.subplots(figsize=(6, max(3, int(top/2))))
    ax.barh(np.array(names)[idx][::-1], np.array(values)[idx][::-1])
    ax.set_title(title); ax.set_xlabel("importance"); ax.set_ylabel("feature")
    st.pyplot(fig, use_container_width=True)

# =========================================================
# UI: Train vs Predict
# =========================================================
page = st.sidebar.radio("Pages", ["Train Model", "Predict"])

# ====================== TRAIN ======================
if page == "Train Model":
    st.subheader("Train a model on your historical dataset")
    up = st.file_uploader("Upload Historical CSV", type=["csv"])
    if up:
        df = clean_dataframe(read_csv_any(up))
        st.write("Preview (first 6 rows):"); st.dataframe(df.head(6), use_container_width=True)

        target = st.selectbox("Target column", df.columns)
        features = st.multiselect("Feature columns", [c for c in df.columns if c != target],
                                  help="Numeric, categorical, multi-label (comma-separated), and long text supported.")
        family = st.selectbox("Algorithm", ALGOS, index=0)
        num_strategy = st.selectbox("Numeric imputation", ["median","mean"])
        cat_strategy = st.selectbox("Categorical imputation", ["most_frequent","constant"])

        if st.button("Train"):
            if not features:
                st.error("Select at least one feature.")
            else:
                model, task, schema, metrics, used_family, label_info = train_and_evaluate(
                    df, features, target, family, num_strategy, cat_strategy, test_size=0.1, random_state=42
                )
                st.session_state.bundle = {
                    "model": model, "features": features, "task": task,
                    "schema": schema, "family": used_family, "label_info": label_info
                }
                os.makedirs("data", exist_ok=True)
                joblib.dump(st.session_state.bundle, "data/admit_model.pkl")

                st.success("✅ Trained on 90%, evaluated on 10% hold-out. Saved to data/admit_model.pkl")

                # Metrics summary
                if task == "binary":
                    auc_txt = f"{metrics['roc_auc']:.3f}"
                    st.markdown(f"**Task:** Binary &nbsp;&nbsp; **Accuracy:** {metrics['accuracy']:.3f} &nbsp;&nbsp; **ROC-AUC:** {auc_txt}")
                elif task == "multiclass":
                    st.markdown(f"**Task:** Multiclass &nbsp;&nbsp; **Accuracy:** {metrics['accuracy']:.3f} &nbsp;&nbsp; **F1-macro:** {metrics['f1_macro']:.3f}")
                else:
                    st.markdown(f"**Task:** Regression &nbsp;&nbsp; **R²:** {metrics['r2']:.3f} &nbsp;&nbsp; **RMSE:** {metrics['rmse']:.3f}")

                # Visualizations
                st.markdown("### Visualizations")
                if task == "binary":
                    plot_roc(metrics["y_test"], metrics["proba"])
                    plot_hist(metrics["proba"], "Predicted probabilities (test)", bins=25, xlabel="p(class=1)")
                    plot_confusion(metrics["y_test"], metrics["y_pred"], normalize=False, title="Confusion matrix (counts)")
                    plot_confusion(metrics["y_test"], metrics["y_pred"], normalize=True,  title="Confusion matrix (normalized)")
                elif task == "multiclass":
                    y_pred = metrics["y_pred"]
                    plot_confusion(metrics["y_test"], y_pred, normalize=False, title="Confusion matrix (counts)")
                    plot_confusion(metrics["y_test"], y_pred, normalize=True,  title="Confusion matrix (normalized)")
                    plot_hist(metrics["y_test"], "Class distribution (test)", bins=len(np.unique(metrics["y_test"])), xlabel="class")
                else:
                    y_pred = metrics["y_pred"]
                    plot_parity(metrics["y_test"], y_pred, title="Parity plot on test split")
                    residuals = y_pred - metrics["y_test"]
                    plot_hist(residuals, "Residuals histogram (test)", bins=30, xlabel="ŷ - y")

                # Global feature importance
                imp = model_feature_importance(model)
                if imp is not None:
                    names, vals = imp
                    st.markdown("### Global feature importance")
                    plot_bar(names, vals, "Top feature importances", top=min(30, len(vals)))
                else:
                    st.info("Global importance not available for this model / pipeline.")

# ====================== PREDICT ======================
else:
    st.subheader("Predict workflow")
    bundle = st.session_state.get("bundle")
    if bundle is None and os.path.exists("data/admit_model.pkl"):
        bundle = joblib.load("data/admit_model.pkl"); st.session_state.bundle = bundle

    if bundle is None:
        st.info("Please train a model on the **Train Model** page first.")
    else:
        model = bundle["model"]; features = bundle["features"]; task = bundle["task"]

        tab1, tab2, tab3, tab4 = st.tabs(["1) Applicants CSV","2) SOP/LOR/CV PDFs (optional)","3) Results & Top-K","4) Explainability"])

        # -- Tab 1: CSV
        with tab1:
            c1,c2 = st.columns(2)
            with c1:
                alpha = st.slider("Fusion weight α (tabular vs doc)", 0.0, 1.0, 0.7, 0.05,
                                  help="α=1 uses only tabular model; α=0 uses only document score.")
            with c2:
                k_top = st.number_input("Top-K (overall)", min_value=1, max_value=1000, value=5, step=1)
            pred_file = st.file_uploader("Upload Applicants CSV", type=["csv"], key="pred_csv")
            if pred_file:
                dfp = clean_dataframe(read_csv_any(pred_file))
                st.dataframe(dfp.head(8), use_container_width=True)
                X = dfp.copy()
                for f in features:
                    if f not in X.columns: X[f] = np.nan
                X = X[features]
                p_tab = predict_prob_like(model, task, X).astype(float)
                dfp["p_tabular"] = p_tab
                st.session_state.pred_df = dfp; st.session_state.alpha = alpha; st.session_state.k_top = int(k_top)

        # -- Tab 2: Docs (PDFs) with selectable KEY column
        with tab2:
            if "pred_df" not in st.session_state:
                st.info("Upload Applicants CSV in tab 1 first.")
            else:
                if not _PYPDF2_OK:
                    st.warning("PyPDF2 is not installed. Run:  pip install PyPDF2  to enable PDF parsing.")
                key_col = st.selectbox(
                    "Key column for document matching",
                    options=list(st.session_state.pred_df.columns),
                    help="We match `<KEY>_SOP.pdf` etc. using values from this column (after lowercasing & cleanup: spaces→_, keep a-z0-9_-.)."
                )
                st.caption(
                    f"Name your files like `<{key_col}>_SOP.pdf`, `<{key_col}>_LOR1.pdf`, `<{key_col}>_CV.pdf`. "
                    "Example: if key is `Serial No.` and value is `A-1029`, file becomes `a-1029_SOP.pdf`."
                )
                w1,w2,w3 = st.columns(3)
                with w1: w_sop = st.slider("SOP weight", 0.0, 1.0, 0.5, 0.05)
                with w2: w_lor = st.slider("LORs weight", 0.0, 1.0, 0.3, 0.05)
                with w3: w_cv  = st.slider("CV weight",  0.0, 1.0, 0.2, 0.05)
                zip_up = st.file_uploader("SOP/LOR/CV ZIP (optional, PDFs only)", type=["zip"], key="doc_zip")
                if zip_up:
                    dfp = st.session_state.pred_df.copy()
                    idx = build_doc_index_from_zip(zip_up)
                    tot = max(1e-6, w_sop + w_lor + w_cv)
                    w_sop, w_lor, w_cv = w_sop/tot, w_lor/tot, w_cv/tot
                    doc_scores=[]
                    for _,row in dfp.iterrows():
                        key_val = normalize_key(row.get(key_col, ""))
                        rec=idx.get(key_val)
                        if not rec: doc_scores.append(np.nan); continue
                        s = score_from_features(sop_lor_features(rec["sop"])) if rec["sop"] else np.nan
                        lt = "\n\n".join(rec["lors"]) if rec["lors"] else ""
                        l = score_from_features(sop_lor_features(lt)) if lt else np.nan
                        c = score_from_features(sop_lor_features(rec["cv"])) if rec.get("cv") else np.nan
                        parts,weights=[],[]
                        if not np.isnan(s): parts.append(s); weights.append(w_sop)
                        if not np.isnan(l): parts.append(l); weights.append(w_lor)
                        if not np.isnan(c): parts.append(c); weights.append(w_cv)
                        if parts:
                            wsum=sum(weights); doc_scores.append(float(np.dot(parts, np.array(weights)/wsum)))
                        else: doc_scores.append(np.nan)
                    dfp["doc_score"]=doc_scores; st.session_state.pred_df = dfp

        # -- Tab 3: Results & Top-K
        with tab3:
            if "pred_df" not in st.session_state:
                st.info("Please upload Applicants CSV in tab 1.")
            else:
                dfp = st.session_state.pred_df.copy()
                alpha = st.session_state.get("alpha",0.7); k_top = int(st.session_state.get("k_top",5))
                if "doc_score" not in dfp.columns: dfp["doc_score"] = np.nan
                dfp["p_final"] = [fuse_scores(t,d,alpha) for t,d in zip(dfp["p_tabular"].astype(float), dfp["doc_score"].astype(float))]

                st.markdown("**Distribution of final scores**")
                plot_hist(dfp["p_final"].values, "p_final histogram", bins=30, xlabel="p_final")

                df_sorted = dfp.sort_values("p_final", ascending=False)
                topk = df_sorted.head(k_top)

                st.subheader(f"Top-{k_top} applicants (overall)")
                st.dataframe(topk, use_container_width=True)

                st.markdown("---")
                st.markdown("**Optional: Top-K per group**")
                group_col = st.selectbox("Group by column", ["(none)"] + [c for c in dfp.columns if c not in ["p_tabular","doc_score","p_final"]])
                if group_col != "(none)":
                    per_k = st.number_input("K per group", 1, 100, min(k_top, 5))
                    grouped = df_sorted.groupby(group_col, group_keys=True).head(int(per_k))
                    st.dataframe(grouped, use_container_width=True)
                    means = df_sorted.groupby(group_col)["p_final"].mean().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(6, max(3, int(len(means)/2))))
                    ax.barh(means.index[::-1], means.values[::-1])
                    ax.set_title("Mean p_final by group"); ax.set_xlabel("mean p_final")
                    st.pyplot(fig, use_container_width=True)

                st.download_button("Download Top-K CSV", data=topk.to_csv(index=False).encode("utf-8"),
                                   file_name="admitrank_topk.csv", mime="text/csv")

                st.markdown("---")
                st.markdown("**Optional segmentation (KMeans on `p_final`)**")
                if st.checkbox("Cluster applicants by `p_final`"):
                    ncl = st.slider("Number of clusters", 2, 8, 3, 1)
                    km = KMeans(n_clusters=ncl, n_init="auto", random_state=42)
                    dfc = df_sorted.copy(); dfc["cluster"]=km.fit_predict(dfc[["p_final"]])
                    st.dataframe(dfc.sort_values(["cluster","p_final"], ascending=[True,False]), use_container_width=True, height=350)

        # -- Tab 4: Explainability
        with tab4:
            if "pred_df" not in st.session_state:
                st.info("Train & predict first.")
            else:
                imp = model_feature_importance(model)
                if imp is not None:
                    n,v = imp
                    plot_bar(n, v, "Global feature importance", top=min(30, len(v)))
                else:
                    st.info("Global importance unavailable for this model / pipeline.")
