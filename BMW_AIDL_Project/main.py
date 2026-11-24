from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def main():
    print("=" * 72)
    print("BMW – INNOVATION IN AUTOMOTIVE TECHNOLOGY")
    print("AIDL Predictive Maintenance Project")
    print("=" * 72)
    print("\n1. Loading the UCI AI4I 2020 dataset...")
    dataset = fetch_ucirepo(id=601)
    X = dataset.data.features.copy()
    y = dataset.data.targets["Machine failure"].astype(int)

    print("Original feature shape:", X.shape)
    print("\nTarget distribution:\n", y.value_counts().sort_index())
    print("\nMissing values in features:", int(X.isnull().sum().sum()))

    # Remove identifiers and failure-mode columns to avoid target leakage.
    drop_cols = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")

    categorical_features = ["Type"]
    numeric_features = [c for c in X.columns if c not in categorical_features]

    print("\n2. Preprocessing data...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # Convert sparse matrices to dense arrays for TensorFlow ANN.
    if hasattr(X_train_p, "toarray"):
        X_train_p = X_train_p.toarray()
        X_test_p = X_test_p.toarray()
    X_train_p = np.asarray(X_train_p, dtype=np.float32)
    X_test_p = np.asarray(X_test_p, dtype=np.float32)

    print("Training shape:", X_train_p.shape)
    print("Testing shape:", X_test_p.shape)

    # Save useful dataset summaries.
    summary = {
        "dataset": "UCI AI4I 2020 Predictive Maintenance Dataset",
        "records": int(len(X)),
        "features_after_exclusions": list(X.columns),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "target_counts": {str(k): int(v) for k, v in y.value_counts().items()},
        "missing_feature_values": int(X.isnull().sum().sum()),
        "note": "Synthetic industrial dataset; not BMW vehicle or factory data."
    }
    save_json(summary, OUTPUT_DIR / "dataset_summary.json")

    print("\n3. Training machine-learning models...")
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        )
    }

    results = []
    predictions = {}

    def evaluate_model(name, y_true, y_pred):
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-score": f1_score(y_true, y_pred, zero_division=0)
        }
        results.append(metrics)
        predictions[name] = y_pred
        print(f"\n{name}")
        print(classification_report(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion matrix:\n", cm)
        save_json(
            {"labels": ["No Failure (0)", "Failure (1)"], "matrix": cm.tolist()},
            OUTPUT_DIR / (name.lower().replace(" ", "_") + "_confusion_matrix.json")
        )
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Failure", "Failure"],
                    yticklabels=["No Failure", "Failure"])
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / (name.lower().replace(" ", "_") + "_confusion_matrix.png"), dpi=150)
        plt.close()

    for name, model in models.items():
        model.fit(X_train_p, y_train)
        pred = model.predict(X_test_p)
        evaluate_model(name, y_test, pred)
        joblib.dump(model, MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib"))

    print("\n4. Building and training the Artificial Neural Network...")
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    ann = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_p.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    ann.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = ann.fit(
        X_train_p, y_train.to_numpy(),
        validation_split=0.20,
        epochs=30,
        batch_size=32,
        verbose=1
    )
    ann_prob = ann.predict(X_test_p, verbose=0).ravel()
    ann_pred = (ann_prob >= 0.5).astype(int)
    evaluate_model("Artificial Neural Network", y_test, ann_pred)
    ann.save(MODEL_DIR / "artificial_neural_network.keras")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")

    # Training history plot.
    plt.figure(figsize=(7, 4))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.title("ANN Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ann_training_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.title("ANN Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ann_training_accuracy.png", dpi=150)
    plt.close()

    print("\n5. Comparing all models...")
    results_df = pd.DataFrame(results).sort_values("F1-score", ascending=False)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print(results_df.to_string(index=False))
    best_model = results_df.iloc[0]["Model"]
    print("\nBest model by test F1-score:", best_model)

    plt.figure(figsize=(8, 4))
    plot_df = results_df.melt(
        id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
        var_name="Metric", value_name="Score"
    )
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric")
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.title("Model Performance Comparison")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_performance_comparison.png", dpi=150)
    plt.close()

    # Illustrative prediction using the first test record.
    probability = float(ann_prob[0])
    predicted_class = int(probability >= 0.5)
    prediction_result = {
        "predicted_failure_probability_score": round(probability, 4),
        "predicted_class": predicted_class,
        "prediction": "Failure" if predicted_class == 1 else "No failure",
        "threshold": 0.5,
        "warning": "Educational model score on synthetic data; not a certified BMW or vehicle safety prediction."
    }
    save_json(prediction_result, OUTPUT_DIR / "illustrative_prediction.json")

    print("\n6. Illustrative ANN prediction")
    print("Predicted failure probability score:", round(probability, 4))
    print("Predicted class:", prediction_result["prediction"])
    print("\nCompleted. Check the outputs/ and models/ folders.")
    print("IMPORTANT: This is an educational prototype, not a validated vehicle safety system.")

if __name__ == "__main__":
    main()
