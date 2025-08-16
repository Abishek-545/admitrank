
# AdmitRank — Train • Predict • Explain

📌 [Live App Link](#)  https://admitrank-dacysgdughauxleskuuend.streamlit.app/
## Overview
AdmitRank is the **first-ever novel business solution** for universities to automate student admission ranking.  
Instead of manually screening thousands of applications, universities can quickly predict the **top-K students** based on both **tabular data** (scores, ratings, CGPA, etc.) and **document insights** (SOPs, LORs, CVs).

## Dataset
- **Training:** Upload any historical CSV dataset.  
- **Prediction:** Upload a new CSV dataset with the same or extended features.  
- **Optional:** Add ZIP of SOP/LOR/CV PDFs linked by student IDs.  
- The system dynamically supports **any number of features** and **different ML models**.

## Key Features
- 📊 **Flexible CSVs** – Train with N features, predict on dynamic data.  
- 🤖 **Pluggable ML models** – Any regression/classifier can be used.  
- 📑 **Document integration** – Link SOP/LOR/CV PDFs to enhance predictions.  
- 🔝 **Top-K Ranking** – Universities see the best candidates instantly.  
- 📈 **Visualization** – Probability distributions & student ranking charts.  

## Alpha Fusion
We combine:
- **p_tabular** → Score from tabular ML model  
- **p_doc** → Score from NLP-based document model  
- **p_final** = α·p_tabular + (1-α)·p_doc  

This **fusion weight (α)** balances tabular vs. document signals for a robust admission decision.

## Business Value 
- 🚀 Saves **time** and **manual effort** for admission teams.  
- 🎯 Provides **objective, consistent, and explainable** rankings.  
- 🌍 Scalable across **all universities** — from small colleges to global institutions.
sequenceDiagram
    participant User
    participant Trainer as Trainer (app)
    participant Model as Trained Model
    participant Docs as PDF Parser
    participant Ranker as Top-K & Viz

    User->>Trainer: Upload training.csv\n+ select target/features
    Trainer->>Trainer: Split with Random Seed\n(hold-out test)
    Trainer->>Model: Fit pipeline (impute/encode/scale + estimator)
    Model-->>Trainer: Metrics (acc/ROC-AUC/RMSE...)
    Trainer-->>User: Train report\n+ saved pipeline & schema

    User->>Trainer: Upload predict.csv\n(+ optional ZIP PDFs)
    Trainer->>Model: Predict p_tabular
    Trainer->>Docs: Extract doc_score_i (missing → None)
    Docs-->>Trainer: doc_score_i per row
    Trainer->>Trainer: Fuse p_final = α·p_tabular + (1−α)·doc_score
    Trainer->>Ranker: Filter to prediction set only\nthen sort by p_final
    Ranker-->>User: Top-K table + charts\n+ downloadable CSV


## For quick testing use the dataset available in data folder( data\samples\sample classification dataset or  data\samples\sample regression dataset) 
- Use Admit_train_test to train the model 
- Use Admit_predict to predict applicant's rank


---



🔗 This project demonstrates how **AI + automation** can modernize the admission process and deliver **faster, fairer, and smarter decisions**.



\## Run locally



```bash

python -m venv .venv

\# Windows

.venv\\Scripts\\activate

\# macOS/Linux

source .venv/bin/activate



pip install -r requirements.txt

streamlit run app/app.py



