import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib
import os

# Step 1: Download dataset automatically
print("Downloading Kaggle dataset...")
path = kagglehub.dataset_download("mohansacharya/graduate-admissions")
print("Path to dataset files:", path)

# Step 2: Load dataset
csv_path = os.path.join(path, "Admission_Predict.csv")  # file name from Kaggle
df = pd.read_csv(csv_path)

# Step 3: Clean column names (remove spaces)
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# Step 4: Create binary label
df["admit"] = (df["Chance_of_Admit"] >= 0.5).astype(int)

# Step 5: Features to use
features = ["GRE_Score", "TOEFL_Score", "CGPA", "Research", "SOP", "LOR", "University_Rating"]

# Step 6: Train/test split
X = df[features]
y = df["admit"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Step 7: Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"Validation AUC: {auc:.3f}")

# Step 9: Save model
os.makedirs("data", exist_ok=True)
joblib.dump(model, "data/admit_model_rf.pkl")
print("Model saved to data/admit_model_rf.pkl")
