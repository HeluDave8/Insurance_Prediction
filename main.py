import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Prevent Tcl/Tk warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# ==============================
# 1. Create results folder
# ==============================
os.makedirs("results", exist_ok=True)

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv("data/TravelInsurancePrediction.csv")
print("Columns:", df.columns)

# ==============================
# 3. Data Understanding
# ==============================
with open("results/data_shape.txt", "w") as f:
    f.write(f"Dataset shape: {df.shape}\n\n")
    f.write(f"Columns:\n{df.columns.tolist()}\n\n")
    f.write(f"Missing values:\n{df.isnull().sum()}\n\n")
    f.write(f"Data types:\n{df.dtypes}\n\n")
    f.write(f"Summary statistics:\n{df.describe()}\n")

# ==============================
# 4. Preprocessing
# ==============================

# Encode categorical columns to 0/1
categorical_cols = ["GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad"]
for col in categorical_cols:
    df[col] = df[col].map({"No": 0, "Yes": 1})

# Encode Employment Type if exists
if "Employment Type" in df.columns:
    df["Employment Type"] = LabelEncoder().fit_transform(df["Employment Type"])

# Features & Target
X = df[["Age", "GraduateOrNot", "AnnualIncome", "FamilyMembers",
        "ChronicDiseases", "FrequentFlyer", "EverTravelledAbroad"]]
y = df["TravelInsurance"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with open("results/train_test_split.txt", "w") as f:
    f.write(f"X_train: {X_train.shape}, X_test: {X_test.shape}\n")
    f.write(f"y_train: {y_train.shape}, y_test: {y_test.shape}\n")

# ==============================
# 5. Exploratory Data Analysis
# ==============================

# Age vs TravelInsurance
plt.figure(figsize=(8,6))
sns.histplot(data=df, x="Age", hue="TravelInsurance", multiple="stack", bins=20)
plt.title("Age vs Travel Insurance", fontsize=14)
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/age_vs_travelinsurance.png", dpi=300)
plt.close()

# AnnualIncome vs TravelInsurance
plt.figure(figsize=(8,6))
sns.histplot(data=df, x="AnnualIncome", hue="TravelInsurance", multiple="stack", bins=20)
plt.title("AnnualIncome vs Travel Insurance", fontsize=14)
plt.xlabel("Annual Income")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/income_vs_travelinsurance.png", dpi=300)
plt.close()

# FamilyMembers vs TravelInsurance
plt.figure(figsize=(8,6))
sns.histplot(data=df, x="FamilyMembers", hue="TravelInsurance", multiple="stack", bins=10)
plt.title("FamilyMembers vs Travel Insurance", fontsize=14)
plt.xlabel("Family Members")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/family_vs_travelinsurance.png", dpi=300)
plt.close()

# Correlation Heatmap (Portfolio-ready)
corr = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.savefig("results/correlation_heatmap.png", dpi=300)
plt.close()

# ==============================
# 6. Train Models
# ==============================

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

metrics = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{name} Confusion Matrix", fontsize=14)
    plt.savefig(f"results/{name.replace(' ', '_').lower()}_confusion_matrix.png", dpi=300)
    plt.close()

# ==============================
# 7. Save Model Metrics
# ==============================

metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv("results/model_metrics.csv")

# ==============================
# 8. Save Best Model
# ==============================

best_model_name = metrics_df["F1"].idxmax()
best_model = models[best_model_name]
joblib.dump(best_model, "results/best_model.pkl")

print(f"Project finished. Best model: {best_model_name}")
print("All results saved in /results folder.")