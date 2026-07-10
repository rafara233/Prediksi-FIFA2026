"""
predict.py
-----------
Modul untuk melakukan prediksi hasil pertandingan menggunakan model
yang sudah dilatih (model/model.pkl), tanpa perlu melatih ulang.
"""

import joblib
import pandas as pd
from pathlib import Path

from feature_engineering import build_features_from_stats

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

# Label numerik model -> label yang mudah dibaca manusia
LABEL_MAP = {
    1: "Home Menang",
    0: "Seri",
    -1: "Away Menang",
}


def load_model(path: str = MODEL_PATH):
    """
    Memuat model Machine Learning yang sudah disimpan (joblib).

    Parameters
    ----------
    path : str
        Path menuju file model.pkl

    Returns
    -------
    dict
        Dictionary berisi 'model', 'feature_columns', dan 'metrics'.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan di: {path}\n"
            "Jalankan `python train_model.py` terlebih dahulu, dan pastikan "
            "folder 'model/' ikut ter-commit/ter-upload bersama app.py."
        )
    return joblib.load(path)


def get_team_stats(team: str, ranking_df: pd.DataFrame) -> dict:
    """
    Mengambil statistik satu tim dari dataset ranking.

    Parameters
    ----------
    team : str
        Nama tim.
    ranking_df : pd.DataFrame
        Dataframe ranking yang sudah bersih.

    Returns
    -------
    dict
        Statistik tim (Rating, Rating Rank, Total Matches, Wins, Losses,
        Draws, Goals For, Goals Against, dst).
    """
    row = ranking_df[ranking_df["Team"] == team]
    if row.empty:
        raise ValueError(f"Tim '{team}' tidak ditemukan pada dataset ranking.")
    return row.iloc[0].to_dict()


def predict_match(home_team: str, away_team: str, ranking_df: pd.DataFrame,
                   model_bundle: dict = None) -> dict:
    """
    Memprediksi hasil pertandingan antara dua tim.

    Parameters
    ----------
    home_team : str
        Nama tim Home.
    away_team : str
        Nama tim Away.
    ranking_df : pd.DataFrame
        Dataframe statistik tim (dataset ranking).
    model_bundle : dict, optional
        Hasil load_model(). Jika None, akan otomatis di-load dari MODEL_PATH.

    Returns
    -------
    dict
        Berisi:
        - 'prediction' : label hasil prediksi (string)
        - 'prediction_code' : label numerik (1 / 0 / -1)
        - 'probabilities' : dict probabilitas tiap kelas
        - 'confidence' : confidence score (probabilitas kelas tertinggi)
        - 'home_stats' / 'away_stats' : statistik kedua tim
        - 'features' : dataframe fitur yang dipakai model
    """
    if model_bundle is None:
        model_bundle = load_model()

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    home_stats = get_team_stats(home_team, ranking_df)
    away_stats = get_team_stats(away_team, ranking_df)

    features = build_features_from_stats(home_stats, away_stats)
    features = features[feature_columns]  # pastikan urutan kolom konsisten

    pred_code = model.predict(features)[0]
    pred_proba = model.predict_proba(features)[0]

    # Petakan probabilitas ke masing-masing kelas sesuai urutan model.classes_
    proba_dict = {
        LABEL_MAP[cls]: float(prob)
        for cls, prob in zip(model.classes_, pred_proba)
    }

    confidence = float(max(pred_proba))

    return {
        "prediction": LABEL_MAP[pred_code],
        "prediction_code": int(pred_code),
        "probabilities": proba_dict,
        "confidence": confidence,
        "home_stats": home_stats,
        "away_stats": away_stats,
        "features": features,
    }


def compute_win_percentages(result: dict, home_team: str, away_team: str) -> dict:
    """
    Mengonversi probabilitas kelas (Home/Seri/Away) menjadi persentase
    peluang menang murni untuk kedua tim (menormalkan agar Home% + Away% = 100%,
    dengan peluang seri dibagi rata ke kedua tim).

    Parameters
    ----------
    result : dict
        Hasil dari predict_match().
    home_team : str
        Nama tim Home.
    away_team : str
        Nama tim Away.

    Returns
    -------
    dict
        {'home_win_pct': float, 'away_win_pct': float}
    """
    probs = result["probabilities"]
    home_p = probs.get("Home Menang", 0.0)
    away_p = probs.get("Away Menang", 0.0)
    draw_p = probs.get("Seri", 0.0)

    # Bagi rata peluang seri ke kedua tim (karena babak gugur harus ada pemenang)
    home_total = home_p + draw_p / 2
    away_total = away_p + draw_p / 2

    total = home_total + away_total
    if total == 0:
        home_total = away_total = 0.5
        total = 1.0

    return {
        "home_win_pct": round((home_total / total) * 100, 2),
        "away_win_pct": round((away_total / total) * 100, 2),
    }
