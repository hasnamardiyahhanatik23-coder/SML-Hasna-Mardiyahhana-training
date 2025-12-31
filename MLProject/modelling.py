import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =====================
# MLflow setup (local)
# =====================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Loan Prediction Experiment")

mlflow.sklearn.autolog(
    log_models=True,
    log_input_examples=True,
    log_model_signatures=True
)

def train():
    # Path data dari root repo (ingat: modelling.py ada di folder MLProject)
    train_path = os.path.join("..", "dataset_preprocessed", "train_clean.csv")
    test_path  = os.path.join("..", "dataset_preprocessed", "test_clean.csv")

    print("Loading preprocessed data...")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan.\n"
            f"- {train_path}\n"
            f"- {test_path}\n"
            f"Pastikan workflow preprocessing sudah menghasilkan file ini."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop("Loan_Status", axis=1)
    y_train = train_df["Loan_Status"]

    X_test = test_df.drop("Loan_Status", axis=1)
    y_test = test_df["Loan_Status"]

    with mlflow.start_run(run_name="Logistic_Regression_Autolog"):
        print("Training Model...")
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Karena target sudah 0/1, pos_label harus 1
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        # Autolog sudah log banyak hal, tapi metric manual ini enak buat bukti
        mlflow.log_metric("accuracy_manual", acc)
        mlflow.log_metric("precision_manual", prec)
        mlflow.log_metric("recall_manual", rec)
        mlflow.log_metric("f1_manual", f1)

        print("\n=== Evaluation Results ===")
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print("==========================")

if __name__ == "__main__":
    train()
