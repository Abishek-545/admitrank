\# AdmitRank — Train • Predict • Explain



A simple end-to-end admissions ranking web app:

\- Train on \*\*any CSV\*\* (numeric, categorical, text)

\- Predict on a new CSV

\- Optional \*\*SOP / LOR / CV PDFs\*\* fusion (name files `<KEY>\_SOP.pdf`, `<KEY>\_LOR1.pdf`, `<KEY>\_CV.pdf`)

\- Choose the \*\*Key column\*\* in the UI to match PDFs with rows

\- Auto-detects \*\*binary / multiclass / regression\*\*, uses \*\*probabilities\*\* for classification

\- Top-K overall + Top-K per group + charts \& feature importance



\## Run locally



```bash

python -m venv .venv

\# Windows

.venv\\Scripts\\activate

\# macOS/Linux

source .venv/bin/activate



pip install -r requirements.txt

streamlit run app/app.py



