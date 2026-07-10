"""
preprocessing.py
-----------------
Modul untuk memuat dan membersihkan dataset ranking tim & riwayat pertandingan.
"""

from pathlib import Path

import pandas as pd

# Path absolut berbasis lokasi file ini, BUKAN current working directory.
# Ini penting agar aplikasi tetap berjalan benar saat di-deploy (mis. di
# Streamlit Cloud), karena working directory saat deploy bisa berbeda
# dengan lokasi file app.py.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RANKING_PATH = BASE_DIR / "data" / "ranking.csv"
DEFAULT_MATCHES_PATH = BASE_DIR / "data" / "matches.csv"


def load_ranking_data(path: str = None) -> pd.DataFrame:
    """
    Memuat dataset ranking/statistik tim dari file CSV.

    Parameters
    ----------
    path : str, optional
        Path menuju file ranking.csv. Jika None, memakai data/ranking.csv
        relatif terhadap lokasi project (bukan current working directory).

    Returns
    -------
    pd.DataFrame
        Dataframe statistik tim yang sudah dibersihkan.
    """
    path = path or DEFAULT_RANKING_PATH
    if not Path(path).exists():
        raise FileNotFoundError(
            f"File dataset ranking tidak ditemukan di: {path}\n"
            "Pastikan folder 'data/' (berisi ranking.csv & matches.csv) "
            "sudah ikut ter-commit/ter-upload bersama app.py."
        )
    df = pd.read_csv(path)
    df = clean_ranking_data(df)
    return df


def load_match_data(path: str = None) -> pd.DataFrame:
    """
    Memuat dataset riwayat pertandingan dari file CSV.

    Parameters
    ----------
    path : str, optional
        Path menuju file matches.csv. Jika None, memakai data/matches.csv
        relatif terhadap lokasi project (bukan current working directory).

    Returns
    -------
    pd.DataFrame
        Dataframe riwayat pertandingan yang sudah dibersihkan.
    """
    path = path or DEFAULT_MATCHES_PATH
    if not Path(path).exists():
        raise FileNotFoundError(
            f"File dataset pertandingan tidak ditemukan di: {path}\n"
            "Pastikan folder 'data/' (berisi ranking.csv & matches.csv) "
            "sudah ikut ter-commit/ter-upload bersama app.py."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    df = clean_match_data(df)
    return df


def clean_ranking_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan dataset ranking:
    - Menghapus baris duplikat
    - Menghapus baris dengan nama tim kosong
    - Mengisi nilai numerik kosong dengan 0
    - Menstandarkan nama tim (strip spasi)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe ranking mentah.

    Returns
    -------
    pd.DataFrame
        Dataframe ranking yang sudah bersih.
    """
    df = df.copy()
    df["Team"] = df["Team"].astype(str).str.strip()
    df = df.dropna(subset=["Team"])
    df = df.drop_duplicates(subset=["Team"], keep="last")

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Hindari pembagian dengan nol saat menghitung win rate nanti
    df["Total Matches"] = df["Total Matches"].replace(0, 1)

    return df.reset_index(drop=True)


def clean_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan dataset riwayat pertandingan:
    - Menghapus baris duplikat
    - Menghapus baris dengan data tim/skor kosong
    - Mengonversi tipe data numerik
    - Membuat label hasil pertandingan (Result):
        1  -> Home menang
        0  -> Seri
        -1 -> Away menang

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe pertandingan mentah.

    Returns
    -------
    pd.DataFrame
        Dataframe pertandingan yang sudah bersih dan berlabel.
    """
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=["Home Team", "Away Team", "Home Score", "Away Score"])

    df["Home Team"] = df["Home Team"].astype(str).str.strip()
    df["Away Team"] = df["Away Team"].astype(str).str.strip()

    df["Home Score"] = pd.to_numeric(df["Home Score"], errors="coerce")
    df["Away Score"] = pd.to_numeric(df["Away Score"], errors="coerce")
    df = df.dropna(subset=["Home Score", "Away Score"])

    # Buat label hasil pertandingan
    df["Result"] = df.apply(_label_result, axis=1)

    return df.reset_index(drop=True)


def _label_result(row) -> int:
    """Fungsi bantu untuk membuat label hasil pertandingan dari satu baris data."""
    if row["Home Score"] > row["Away Score"]:
        return 1
    elif row["Home Score"] < row["Away Score"]:
        return -1
    else:
        return 0
