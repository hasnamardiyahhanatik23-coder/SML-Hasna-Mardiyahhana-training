import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# =====================
# 1. Setup MLflow Local
# =====================
# Set tracking URI ke folder lokal agar tidak perlu server DagsHub
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Loan Prediction Experiment")

# Aktifkan Autolog! 
# Ini otomatis mencatat: Parameter model, Metrik akurasi, dan File Model (.pkl)
mlflow.sklearn.autolog()

def train():
    # =====================
    # 2. Load Dataset (Dari Repo 1)
    # =====================
    print("Loading preprocessed data...")
    # Pastikan file ini ada (hasil download dari GitHub Action workflow sebelumnya)
    if not os.path.exists("clean_dataset/train_clean.csv"):
        print("Error: File dataset tidak ditemukan. Pastikan GitHub Action sudah mendownloadnya.")
        return

    train_df = pd.read_csv("clean_dataset/train_clean.csv")
    test_df = pd.read_csv("dataset/test_clean.csv")

    X_train = train_df.drop("Loan_Status", axis=1)
    y_train = train_df["Loan_Status"]
    X_test  = test_df.drop("Loan_Status", axis=1)
    y_test  = test_df["Loan_Status"]

    # =====================
    # 3. Training Loop
    # =====================
    with mlflow.start_run(run_name="Logistic_Regression_Autolog"):
        print("Training Model...")
        
        # Kita pakai LogisticRegression sesuai request
        # Tidak perlu StandardScaler lagi karena sudah dilakukan di Repo 1
        model = LogisticRegression(max_iter=200, random_state=42)
        
        # Saat .fit() dipanggil, MLflow otomatis mencatat semuanya
        model.fit(X_train, y_train)

        # =====================
        # 4. Manual Print (Opsional)
        # =====================
        # Autolog sudah simpan di background, tapi kita print biar muncul di log GitHub Action
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print("\n=== Evaluation Results ===")
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print("==========================")
        print("Model & Metrics automatically logged by MLflow.")

if __name__ == "__main__":
    train()