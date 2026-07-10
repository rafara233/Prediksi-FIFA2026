"""
generate_sample_data.py
------------------------
Script BANTUAN (opsional) untuk menghasilkan contoh dataset:
    - data/ranking.csv
    - data/matches.csv

Dataset dibuat menggunakan simulasi Elo Rating agar hubungan antar variabel
(rating, hasil pertandingan, statistik tim) realistis dan konsisten.

CATATAN PENTING:
Jika Anda memiliki dataset asli (misalnya dari FIFA/Kaggle), silakan GANTI
file data/ranking.csv dan data/matches.csv dengan data asli Anda selama
format kolomnya sama. Script ini hanya untuk membuat data contoh agar
aplikasi dapat langsung dicoba (demo).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# 32 tim yang biasa berlaga di Piala Dunia beserta kekuatan dasar (base strength)
# Semakin tinggi base_strength, semakin besar peluang rating awal tinggi
TEAMS = {
    "Brazil": 2000, "Argentina": 1980, "France": 1970, "England": 1930,
    "Spain": 1950, "Germany": 1920, "Portugal": 1910, "Netherlands": 1900,
    "Belgium": 1890, "Italy": 1880, "Croatia": 1830, "Uruguay": 1820,
    "Colombia": 1790, "Morocco": 1780, "Switzerland": 1760, "USA": 1740,
    "Mexico": 1730, "Japan": 1720, "Senegal": 1710, "Denmark": 1700,
    "Germany2": 1690, "Wales": 1650, "Poland": 1640, "Serbia": 1630,
    "South Korea": 1620, "Ecuador": 1610, "Iran": 1590, "Canada": 1580,
    "Australia": 1570, "Tunisia": 1560, "Ghana": 1550, "Costa Rica": 1540,
}
# fix duplicate key typo above
TEAMS.pop("Germany2", None)
TEAMS["Sweden"] = 1690

TEAM_NAMES = list(TEAMS.keys())

K_FACTOR = 30  # faktor pembelajaran Elo standar untuk sepak bola


def expected_score(rating_a, rating_b):
    """Menghitung probabilitas menang tim A terhadap tim B berdasarkan Elo."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def simulate_match(rating_home, rating_away, home_advantage=60):
    """
    Simulasikan skor pertandingan menggunakan distribusi Poisson yang
    dipengaruhi oleh selisih rating (termasuk keuntungan tuan rumah).
    Mengembalikan (home_score, away_score).
    """
    rating_diff = (rating_home + home_advantage) - rating_away
    # rata-rata gol dipengaruhi rating diff, dibatasi agar tetap realistis
    lambda_home = max(0.4, 1.3 + rating_diff / 500)
    lambda_away = max(0.4, 1.3 - rating_diff / 500)
    home_score = np.random.poisson(lambda_home)
    away_score = np.random.poisson(lambda_away)
    return home_score, away_score


def generate_matches(n_matches=1200, start_year=2010, end_year=2024):
    """Membuat riwayat pertandingan sekaligus mengupdate rating Elo tiap tim."""
    ratings = dict(TEAMS)
    stats = {t: {"Total Matches": 0, "Home Matches": 0, "Away Matches": 0,
                 "Neutral Matches": 0, "Wins": 0, "Losses": 0, "Draws": 0,
                 "Goals For": 0, "Goals Against": 0} for t in TEAM_NAMES}

    tournaments = ["World Cup Qualifier", "Friendly", "World Cup Group Stage",
                   "World Cup Quarter Final", "World Cup Semi Final", "World Cup Final",
                   "Continental Cup"]

    records = []
    total_days = (end_year - start_year) * 365

    for i in range(n_matches):
        home, away = np.random.choice(TEAM_NAMES, size=2, replace=False)
        is_neutral = np.random.rand() < 0.25  # 25% pertandingan di tempat netral

        rating_home_before = ratings[home]
        rating_away_before = ratings[away]

        home_score, away_score = simulate_match(
            rating_home_before, rating_away_before,
            home_advantage=0 if is_neutral else 60
        )

        # Tentukan skor aktual Elo (1 = menang, 0.5 = seri, 0 = kalah)
        if home_score > away_score:
            actual_home, actual_away = 1.0, 0.0
        elif home_score < away_score:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        exp_home = expected_score(rating_home_before, rating_away_before)
        exp_away = 1 - exp_home

        change_home = round(K_FACTOR * (actual_home - exp_home), 2)
        change_away = round(K_FACTOR * (actual_away - exp_away), 2)

        ratings[home] = round(rating_home_before + change_home, 2)
        ratings[away] = round(rating_away_before + change_away, 2)

        # update statistik kumulatif
        stats[home]["Total Matches"] += 1
        stats[away]["Total Matches"] += 1
        if is_neutral:
            stats[home]["Neutral Matches"] += 1
            stats[away]["Neutral Matches"] += 1
        else:
            stats[home]["Home Matches"] += 1
            stats[away]["Away Matches"] += 1

        stats[home]["Goals For"] += home_score
        stats[home]["Goals Against"] += away_score
        stats[away]["Goals For"] += away_score
        stats[away]["Goals Against"] += home_score

        if home_score > away_score:
            stats[home]["Wins"] += 1
            stats[away]["Losses"] += 1
        elif home_score < away_score:
            stats[away]["Wins"] += 1
            stats[home]["Losses"] += 1
        else:
            stats[home]["Draws"] += 1
            stats[away]["Draws"] += 1

        match_date = datetime(start_year, 1, 1) + timedelta(days=int(i / n_matches * total_days))

        records.append({
            "Date": match_date.strftime("%Y-%m-%d"),
            "Home Team": home,
            "Away Team": away,
            "Home Score": home_score,
            "Away Score": away_score,
            "Tournament": np.random.choice(tournaments, p=[0.35, 0.25, 0.2, 0.1, 0.06, 0.02, 0.02]),
            "Home_Rating_Before": rating_home_before,
            "Away_Rating_Before": rating_away_before,
            "Home_Rating_Change": change_home,
            "Away_Rating_Change": change_away,
        })

    matches_df = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
    return matches_df, ratings, stats


def build_ranking_df(ratings, stats):
    """Menggabungkan rating akhir dan statistik kumulatif menjadi dataset ranking."""
    rows = []
    for team in TEAM_NAMES:
        rows.append({
            "Team": team,
            "Rating": round(ratings[team], 2),
            **stats[team],
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("Rating", ascending=False).reset_index(drop=True)
    df["Rating Rank"] = df.index + 1
    # reorder kolom sesuai spesifikasi
    df = df[["Team", "Rating", "Rating Rank", "Total Matches", "Home Matches",
              "Away Matches", "Neutral Matches", "Wins", "Losses", "Draws",
              "Goals For", "Goals Against"]]
    return df


if __name__ == "__main__":
    matches_df, final_ratings, final_stats = generate_matches(n_matches=1500)
    ranking_df = build_ranking_df(final_ratings, final_stats)

    matches_df.to_csv("data/matches.csv", index=False)
    ranking_df.to_csv("data/ranking.csv", index=False)

    print("Dataset berhasil dibuat:")
    print(f" - data/ranking.csv ({len(ranking_df)} tim)")
    print(f" - data/matches.csv ({len(matches_df)} pertandingan)")
