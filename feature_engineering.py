"""
feature_engineering.py
------------------------
Modul untuk membentuk fitur (features) yang digunakan Machine Learning.
Modul ini dipakai baik saat TRAINING (train_model.py) maupun saat
PREDIKSI (predict.py) agar fitur yang dibentuk selalu konsisten.
"""

import pandas as pd

# Urutan kolom fitur -> HARUS SAMA persis antara training & prediksi
FEATURE_COLUMNS = [
    "Home Rating", "Away Rating", "Rating Difference",
    "Home Rank", "Away Rank", "Rank Difference",
    "Total Matches Home", "Total Matches Away",
    "Home Wins", "Away Wins",
    "Home Losses", "Away Losses",
    "Home Draws", "Away Draws",
    "Home Goals For", "Away Goals For",
    "Home Goals Against", "Away Goals Against",
    "Goal Difference",
    "Win Rate Home", "Win Rate Away",
]


def _win_rate(wins, total_matches):
    """Menghitung win rate dengan aman (menghindari pembagian nol)."""
    total_matches = total_matches if total_matches else 1
    return wins / total_matches


def build_features_from_stats(home_stats: dict, away_stats: dict,
                               home_rating: float = None,
                               away_rating: float = None) -> pd.DataFrame:
    """
    Membentuk satu baris fitur dari statistik dua tim (dipakai saat PREDIKSI,
    dan juga bisa dipanggil ulang secara internal saat training).

    Parameters
    ----------
    home_stats : dict
        Statistik tim Home, harus memiliki key:
        'Rating', 'Rating Rank', 'Total Matches', 'Wins', 'Losses',
        'Draws', 'Goals For', 'Goals Against'
    away_stats : dict
        Statistik tim Away, format sama seperti home_stats.
    home_rating : float, optional
        Override rating Home (misalnya rating sebelum pertandingan saat
        training). Jika None, memakai home_stats['Rating'].
    away_rating : float, optional
        Override rating Away. Jika None, memakai away_stats['Rating'].

    Returns
    -------
    pd.DataFrame
        Dataframe satu baris berisi fitur sesuai FEATURE_COLUMNS.
    """
    h_rating = home_rating if home_rating is not None else home_stats["Rating"]
    a_rating = away_rating if away_rating is not None else away_stats["Rating"]

    row = {
        "Home Rating": h_rating,
        "Away Rating": a_rating,
        "Rating Difference": h_rating - a_rating,

        "Home Rank": home_stats["Rating Rank"],
        "Away Rank": away_stats["Rating Rank"],
        "Rank Difference": home_stats["Rating Rank"] - away_stats["Rating Rank"],

        "Total Matches Home": home_stats["Total Matches"],
        "Total Matches Away": away_stats["Total Matches"],

        "Home Wins": home_stats["Wins"],
        "Away Wins": away_stats["Wins"],
        "Home Losses": home_stats["Losses"],
        "Away Losses": away_stats["Losses"],
        "Home Draws": home_stats["Draws"],
        "Away Draws": away_stats["Draws"],

        "Home Goals For": home_stats["Goals For"],
        "Away Goals For": away_stats["Goals For"],
        "Home Goals Against": home_stats["Goals Against"],
        "Away Goals Against": away_stats["Goals Against"],
        "Goal Difference": (home_stats["Goals For"] - home_stats["Goals Against"]) -
                            (away_stats["Goals For"] - away_stats["Goals Against"]),

        "Win Rate Home": _win_rate(home_stats["Wins"], home_stats["Total Matches"]),
        "Win Rate Away": _win_rate(away_stats["Wins"], away_stats["Total Matches"]),
    }

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def build_training_features(matches_df: pd.DataFrame, ranking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Membentuk seluruh fitur untuk data training dengan menggabungkan
    (merge) riwayat pertandingan dengan statistik tim dari dataset ranking.
    Rating yang dipakai adalah rating SEBELUM pertandingan (time-appropriate),
    sedangkan statistik lain (menang/kalah/gol/rank) memakai data ranking.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Dataframe riwayat pertandingan yang sudah bersih & berlabel (Result).
    ranking_df : pd.DataFrame
        Dataframe statistik tim yang sudah bersih.

    Returns
    -------
    pd.DataFrame
        Dataframe fitur (X) dan label (y = 'Result') siap dipakai training.
    """
    ranking_indexed = ranking_df.set_index("Team")

    feature_rows = []
    labels = []

    for _, match in matches_df.iterrows():
        home_team = match["Home Team"]
        away_team = match["Away Team"]

        # Lewati pertandingan jika salah satu tim tidak ada di dataset ranking
        if home_team not in ranking_indexed.index or away_team not in ranking_indexed.index:
            continue

        home_stats = ranking_indexed.loc[home_team].to_dict()
        away_stats = ranking_indexed.loc[away_team].to_dict()

        # Gunakan rating sebelum pertandingan jika tersedia di dataset matches
        home_rating = match.get("Home_Rating_Before", home_stats["Rating"])
        away_rating = match.get("Away_Rating_Before", away_stats["Rating"])

        feature_row = build_features_from_stats(
            home_stats, away_stats,
            home_rating=home_rating, away_rating=away_rating
        )
        feature_rows.append(feature_row)
        labels.append(match["Result"])

    X = pd.concat(feature_rows, ignore_index=True)
    y = pd.Series(labels, name="Result")

    return X, y
