
# AdmitRank — Train • Predict • Explain
flowchart LR
    A[Upload training CSV] --> B[Select target & features]
    B --> C[Choose model family\n(SVM / RF / XGB / Linear...)]
    C --> D[Set test size & Random Seed]
    D --> E[Train & evaluate\n(hold-out metrics)]
    E --> F[Save trained pipeline & schema]

subgraph Prediction
        G[Upload prediction CSV] --> H[Optional ZIP of PDFs\nSOP/LOR/CV]
        H --> I[Parse PDFs → doc_score_i\n(missing ok)]
        G --> J[Tabular model → p_tabular]
        I --> K[Fusion: p_final = α·p_tabular + (1−α)·doc_score]
        J --> K
end

 F -. uses schema & model .-> J
 K --> L[Top-K per group (or overall)]
    L --> M[Visualize\n(distributions, feature importances)]
    M --> N[Export results (CSV)]

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



