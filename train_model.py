"""
train_model.py
----------------
Script untuk melatih model Machine Learning yang memprediksi hasil
pertandingan babak gugur Piala Dunia FIFA.

Alur:
1. Load dataset ranking & pertandingan
2. Preprocessing & cleaning
3. Feature engineering
4. Split data (80% train, 20% test)
5. Training model (default: Random Forest, mudah diganti ke XGBoost)
6. Evaluasi model
7. Simpan model ke model/model.pkl menggunakan joblib

Jalankan dengan:
    python train_model.py
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from preprocessing import load_ranking_data, load_match_data
from feature_engineering import build_training_features, FEATURE_COLUMNS

# ============================================================
# KONFIGURASI
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
RANKING_PATH = BASE_DIR / "data" / "ranking.csv"
MATCHES_PATH = BASE_DIR / "data" / "matches.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "model" / "model.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def get_model():
    """
    Mengembalikan objek model yang akan dilatih.

    CATATAN UNTUK PENGEMBANGAN SELANJUTNYA:
    Untuk mengganti algoritma menjadi XGBoost, cukup ubah fungsi ini, misalnya:

        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=RANDOM_STATE
        )

    Seluruh alur training/evaluasi/penyimpanan lainnya TIDAK perlu diubah,
    karena sudah dirancang model-agnostic (tidak bergantung algoritma tertentu).
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train():
    """Fungsi utama untuk menjalankan seluruh proses training model."""

    print("1. Memuat dataset...")
    ranking_df = load_ranking_data(RANKING_PATH)
    matches_df = load_match_data(MATCHES_PATH)
    print(f"   - Dataset ranking : {ranking_df.shape[0]} tim")
    print(f"   - Dataset matches  : {matches_df.shape[0]} pertandingan")

    print("2. Membentuk fitur (feature engineering)...")
    X, y = build_training_features(matches_df, ranking_df)
    print(f"   - Total sampel fitur: {X.shape[0]}")
    print(f"   - Jumlah fitur      : {X.shape[1]}")

    print("3. Membagi data training (80%) & testing (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   - Data latih : {X_train.shape[0]} sampel")
    print(f"   - Data uji   : {X_test.shape[0]} sampel")

    print("4. Melatih model (Random Forest Classifier)...")
    model = get_model()
    model.fit(X_train, y_train)

    print("5. Evaluasi model...")
    y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred)
    print_metrics(metrics)

    print("6. Menyimpan model ke", MODEL_OUTPUT_PATH)
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
    }, MODEL_OUTPUT_PATH)

    print("\nTraining selesai. Model berhasil disimpan.")
    return model, metrics


def evaluate_model(y_test, y_pred) -> dict:
    """
    Menghitung metrik evaluasi model klasifikasi multi-kelas
    (Home menang / Seri / Away menang).

    Parameters
    ----------
    y_test : array-like
        Label sebenarnya.
    y_pred : array-like
        Label hasil prediksi model.

    Returns
    -------
    dict
        Kumpulan metrik evaluasi (accuracy, precision, recall, f1,
        confusion matrix, classification report).
    """
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[1, 0, -1]),
        "classification_report": classification_report(
            y_test, y_pred, labels=[1, 0, -1],
            target_names=["Home Menang", "Seri", "Away Menang"],
            zero_division=0,
        ),
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    """Menampilkan hasil evaluasi model ke console dengan rapi."""
    print(f"   - Accuracy  : {metrics['accuracy']:.4f}")
    print(f"   - Precision : {metrics['precision']:.4f}")
    print(f"   - Recall    : {metrics['recall']:.4f}")
    print(f"   - F1 Score  : {metrics['f1_score']:.4f}")
    print("   - Confusion Matrix (baris=aktual, kolom=prediksi) [Home, Seri, Away]:")
    print(metrics["confusion_matrix"])
    print("   - Classification Report:")
    print(metrics["classification_report"])


if __name__ == "__main__":
    train()
