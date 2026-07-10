"""
fetch_live_data.py
--------------------
Modul untuk MENGAMBIL DATA LANGSUNG (crawl) dari eloratings.net, sebagai
alternatif dari dataset sintetis (generate_sample_data.py).

Sumber data:
- https://eloratings.net/World.tsv        -> ranking Elo semua tim saat ini
- https://eloratings.net/<Nama_Tim>.tsv   -> riwayat pertandingan per tim

Kita TIDAK melakukan HTTP request manual ke file .tsv tersebut secara
langsung, melainkan memakai library pihak ketiga `datafc` (tersedia di PyPI,
MIT license) yang sudah membungkus endpoint tsv tersebut menjadi DataFrame
pandas yang bersih, dengan nama kolom yang sudah diverifikasi terhadap
source JavaScript situs eloratings.net.

Instalasi:
    pip install datafc

Cara pakai:
    python fetch_live_data.py

Script ini akan menimpa (overwrite) data/ranking.csv dan data/matches.csv
dengan data ASLI/TERKINI dari eloratings.net, lalu Anda tinggal jalankan
ulang `python train_model.py` untuk melatih model dengan data nyata.

CATATAN PENTING:
1. Situs eloratings.net TIDAK mempublikasikan API resmi, sehingga struktur
   datanya bisa berubah sewaktu-waktu tanpa pemberitahuan. Jika suatu saat
   script ini error / kolom terasa salah, cek rilis terbaru `datafc`.
2. Penentuan tim "Home" vs "Away" pada riwayat pertandingan memakai
   heuristik dari kolom `host` (kode negara tuan rumah). Untuk pertandingan
   di tempat netral (mis. Piala Dunia), heuristik ini mungkin tidak 100%
   akurat -- silakan periksa beberapa baris data/matches.csv setelah
   fetching untuk memastikan hasilnya masuk akal.
3. Proses ini melakukan request ke internet untuk SETIAP tim (rate-limited
   ~2 request/detik oleh `datafc`), sehingga bisa memakan waktu beberapa
   menit tergantung jumlah tim yang diambil.
"""

import time
from pathlib import Path

import pandas as pd

try:
    from datafc import eloratings, DiskCache
except ImportError as e:
    raise SystemExit(
        "Library 'datafc' belum terinstall. Jalankan: pip install datafc"
    ) from e

# ============================================================
# KONFIGURASI
# ============================================================
# Path absolut berbasis lokasi file ini, BUKAN current working directory,
# supaya konsisten dengan modul lain (preprocessing.py, train_model.py, dst)
# dan tetap benar saat dijalankan dari lokasi mana pun (mis. dipanggil dari
# app.py di Streamlit Cloud).
BASE_DIR = Path(__file__).resolve().parent
RANKING_OUTPUT_PATH = BASE_DIR / "data" / "ranking.csv"
MATCHES_OUTPUT_PATH = BASE_DIR / "data" / "matches.csv"

# Jumlah tim teratas yang riwayat pertandingannya akan diambil.
# Semakin besar, semakin lama waktu fetching (dan semakin besar dataset).
# Set None untuk mengambil SEMUA tim di ranking dunia (bisa >200 negara).
TOP_N_TEAMS = 60

RATE_LIMIT = 2.0  # request per detik, mengikuti etika crawling datafc
CACHE = DiskCache(cache_dir=str(BASE_DIR / ".eloratings_cache"), ttl_hours=24)


_CODE_TO_NAME_CACHE = {}


def _get_code_to_name_map() -> dict:
    """
    Mengambil (dan meng-cache di memori) pemetaan kode negara -> nama
    lengkap, dipakai untuk mengubah kode 2-3 huruf pada data mentah
    (mis. 'ES', 'AR') menjadi nama tim yang mudah dibaca ('Spain', 'Argentina').
    """
    if not _CODE_TO_NAME_CACHE:
        codes_df = eloratings.country_codes_data(rate_limit=RATE_LIMIT, cache=CACHE)
        _CODE_TO_NAME_CACHE.update(dict(zip(codes_df["country_code"], codes_df["country_name"])))
    return _CODE_TO_NAME_CACHE


def fetch_world_ranking() -> pd.DataFrame:
    """
    Mengambil ranking Elo dunia saat ini dan menggabungkannya dengan nama
    negara lengkap (bukan kode 2-3 huruf), lalu mengubahnya ke format
    yang sama seperti data/ranking.csv pada project ini.

    Returns
    -------
    pd.DataFrame
        Kolom: Team, Rating, Rating Rank, Total Matches, Home Matches,
        Away Matches, Neutral Matches, Wins, Losses, Draws, Goals For,
        Goals Against.
    """
    print("Mengambil ranking Elo dunia dari eloratings.net ...")
    world_df = eloratings.world_ranking_data(rate_limit=RATE_LIMIT, cache=CACHE)
    code_to_name = _get_code_to_name_map()

    world_df["Team"] = world_df["country"].map(code_to_name).fillna(world_df["country"])

    ranking_df = pd.DataFrame({
        "Team": world_df["Team"],
        "Rating": world_df["elo"],
        "Rating Rank": world_df["rank"],
        "Total Matches": world_df["matches_total"],
        "Home Matches": world_df["matches_home"],
        "Away Matches": world_df["matches_away"],
        "Neutral Matches": world_df["matches_neutral"],
        "Wins": world_df["wins"],
        "Losses": world_df["losses"],
        "Draws": world_df["draws"],
        "Goals For": world_df["goals_for"],
        "Goals Against": world_df["goals_against"],
    })

    ranking_df = ranking_df.sort_values("Rating Rank").reset_index(drop=True)
    print(f"   -> {len(ranking_df)} tim berhasil diambil.")
    return ranking_df


def _team_slug(team_name: str) -> str:
    """Mengubah nama tim menjadi slug URL eloratings.net (spasi -> underscore)."""
    return team_name.strip().replace(" ", "_")


def fetch_team_matches(team_name: str) -> pd.DataFrame:
    """
    Mengambil riwayat pertandingan satu tim dan mengonversinya ke format
    matches.csv pada project ini (Home/Away ditentukan dari kolom `host`).

    Parameters
    ----------
    team_name : str
        Nama lengkap tim (mis. "Spain", "Brazil").

    Returns
    -------
    pd.DataFrame
        Kolom sesuai data/matches.csv, atau DataFrame kosong jika gagal.
    """
    slug = _team_slug(team_name)
    try:
        df = eloratings.country_matches_data(slug, rate_limit=RATE_LIMIT, cache=CACHE)
    except Exception as e:
        print(f"   [!] Gagal mengambil data '{team_name}' ({slug}): {e}")
        return pd.DataFrame()

    code_to_name = _get_code_to_name_map()

    rows = []
    for _, m in df.iterrows():
        # team_a/team_b/host pada data mentah berupa kode negara (mis. 'ES'),
        # ubah menjadi nama lengkap agar konsisten dengan ranking.csv
        team_a = code_to_name.get(m["team_a"], m["team_a"])
        team_b = code_to_name.get(m["team_b"], m["team_b"])
        host_raw = m.get("host")
        host = code_to_name.get(host_raw, host_raw)

        # Tentukan tim Home berdasarkan kode negara tuan rumah (host).
        # Jika host tidak cocok dengan salah satu tim -> anggap netral,
        # dan team_a diperlakukan sebagai "Home" secara konvensi saja.
        if pd.notna(host) and host == team_a:
            home, away = team_a, team_b
            home_score, away_score = m["team_a_score"], m["team_b_score"]
            home_rating, away_rating = m["team_a_rating"], m["team_b_rating"]
            home_rank_chg, away_rank_chg = m["team_a_rank_change"], m["team_b_rank_change"]
        elif pd.notna(host) and host == team_b:
            home, away = team_b, team_a
            home_score, away_score = m["team_b_score"], m["team_a_score"]
            home_rating, away_rating = m["team_b_rating"], m["team_a_rating"]
            home_rank_chg, away_rank_chg = m["team_b_rank_change"], m["team_a_rank_change"]
        else:
            home, away = team_a, team_b
            home_score, away_score = m["team_a_score"], m["team_b_score"]
            home_rating, away_rating = m["team_a_rating"], m["team_b_rating"]
            home_rank_chg, away_rank_chg = m["team_a_rank_change"], m["team_b_rank_change"]

        rows.append({
            "Date": m["date"],
            "Home Team": home,
            "Away Team": away,
            "Home Score": home_score,
            "Away Score": away_score,
            "Tournament": m["tournament"],
            "Home_Rating_Before": home_rating,
            "Away_Rating_Before": away_rating,
            # Situs tidak memberi "rating change" langsung per pertandingan
            # secara eksplisit di kolom terpisah untuk versi ini; kita
            # pakai proxy dari perubahan RANK sebagai indikator arah,
            # nilai rating sesudah pertandingan bisa dihitung ulang saat
            # training (lihat catatan di feature_engineering.py).
            "Home_Rating_Change": home_rank_chg,
            "Away_Rating_Change": away_rank_chg,
        })

    return pd.DataFrame(rows)


def fetch_all_matches(team_names) -> pd.DataFrame:
    """
    Mengambil riwayat pertandingan untuk sekumpulan tim, lalu menghapus
    duplikat (karena satu pertandingan bisa muncul di riwayat kedua tim).

    Parameters
    ----------
    team_names : list[str]
        Daftar nama tim yang akan diambil riwayat pertandingannya.

    Returns
    -------
    pd.DataFrame
        Dataframe gabungan seluruh pertandingan, tanpa duplikat.
    """
    all_matches = []
    for i, team in enumerate(team_names, 1):
        print(f"[{i}/{len(team_names)}] Mengambil riwayat pertandingan: {team} ...")
        team_df = fetch_team_matches(team)
        if not team_df.empty:
            all_matches.append(team_df)
        time.sleep(1.0 / RATE_LIMIT)  # jaga-jaga tambahan delay antar tim

    if not all_matches:
        raise RuntimeError("Tidak ada data pertandingan yang berhasil diambil.")

    matches_df = pd.concat(all_matches, ignore_index=True)

    # Hapus duplikat: pertandingan yang sama bisa muncul 2x (dari sudut
    # pandang tim Home & tim Away). Kita anggap duplikat jika tanggal +
    # pasangan tim (tanpa memandang urutan) + skor sama.
    matches_df["_team_pair"] = matches_df.apply(
        lambda r: tuple(sorted([r["Home Team"], r["Away Team"]])), axis=1
    )
    matches_df = matches_df.drop_duplicates(subset=["Date", "_team_pair", "Home Score", "Away Score"])
    matches_df = matches_df.drop(columns=["_team_pair"]).sort_values("Date").reset_index(drop=True)

    print(f"Total pertandingan unik: {len(matches_df)}")
    return matches_df


def main():
    """Fungsi utama: fetch ranking + matches, lalu simpan ke data/*.csv."""
    ranking_df = fetch_world_ranking()

    if TOP_N_TEAMS is not None:
        team_names = ranking_df.sort_values("Rating Rank")["Team"].head(TOP_N_TEAMS).tolist()
    else:
        team_names = ranking_df["Team"].tolist()

    matches_df = fetch_all_matches(team_names)

    # Hanya simpan ranking untuk tim yang riwayat pertandingannya diambil,
    # supaya konsisten dengan matches.csv (menghindari tim tanpa histori).
    ranking_df = ranking_df[ranking_df["Team"].isin(team_names)].reset_index(drop=True)

    ranking_df.to_csv(RANKING_OUTPUT_PATH, index=False)
    matches_df.to_csv(MATCHES_OUTPUT_PATH, index=False)

    print("\nSelesai! Data asli dari eloratings.net disimpan ke:")
    print(f" - {RANKING_OUTPUT_PATH} ({len(ranking_df)} tim)")
    print(f" - {MATCHES_OUTPUT_PATH} ({len(matches_df)} pertandingan)")
    print("\nLangkah selanjutnya: jalankan `python train_model.py` untuk melatih ulang model.")


if __name__ == "__main__":
    main()
