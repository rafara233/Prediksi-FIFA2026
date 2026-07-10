"""
app.py
-------
Aplikasi Streamlit untuk memprediksi hasil pertandingan babak gugur
(Perempat Final, Semifinal, Final) Piala Dunia FIFA menggunakan
model Machine Learning yang sudah dilatih sebelumnya.

Jalankan dengan:
    streamlit run app.py
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from preprocessing import load_ranking_data
from predict import load_model, predict_match, compute_win_percentages

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prediksi Babak Gugur Piala Dunia FIFA",
    page_icon="⚽",
    layout="wide",
)

# Path absolut berbasis lokasi file app.py ini, BUKAN current working
# directory. Penting agar aplikasi tetap menemukan file data/model saat
# di-deploy (mis. Streamlit Cloud), karena working directory saat deploy
# bisa berbeda dari lokasi app.py.
BASE_DIR = Path(__file__).resolve().parent
RANKING_PATH = BASE_DIR / "data" / "ranking.csv"
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

ROUND_LABELS = {
    "Perempat Final": "Quarter Final",
    "Semifinal": "Semi Final",
    "Final": "Final",
}


# ============================================================
# CACHING - Load data & model sekali saja
# ============================================================
@st.cache_data
def get_ranking_data():
    """Memuat dataset ranking tim (di-cache agar tidak dibaca berulang kali)."""
    return load_ranking_data(RANKING_PATH)


@st.cache_resource
def get_model_bundle():
    """Memuat model Machine Learning yang sudah dilatih (di-cache)."""
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH)


def format_number(value) -> str:
    """Memformat angka statistik agar mudah dibaca (tanpa desimal berlebihan)."""
    try:
        if float(value) == int(value):
            return str(int(value))
        return f"{float(value):.2f}"
    except (ValueError, TypeError):
        return str(value)


def render_stat_bar_chart(home_team, away_team, home_stats, away_stats):
    """Menampilkan bar chart perbandingan statistik dua tim menggunakan Plotly."""
    categories = ["Elo Rating", "Total Match", "Wins", "Goals For", "Goals Against"]
    home_values = [
        home_stats["Rating"], home_stats["Total Matches"],
        home_stats["Wins"], home_stats["Goals For"], home_stats["Goals Against"],
    ]
    away_values = [
        away_stats["Rating"], away_stats["Total Matches"],
        away_stats["Wins"], away_stats["Goals For"], away_stats["Goals Against"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(name=home_team, x=categories, y=home_values, marker_color="#1f77b4"))
    fig.add_trace(go.Bar(name=away_team, x=categories, y=away_values, marker_color="#d62728"))
    fig.update_layout(
        barmode="group",
        title="Perbandingan Statistik Tim",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_radar_chart(home_team, away_team, home_stats, away_stats):
    """Menampilkan radar chart perbandingan performa dua tim (dinormalisasi 0-100)."""
    metrics = ["Rating", "Win Rate", "Goals For", "Goals Against (Inverse)", "Total Matches"]

    def normalize(value, min_v, max_v, invert=False):
        if max_v == min_v:
            return 50
        score = (value - min_v) / (max_v - min_v) * 100
        return 100 - score if invert else score

    home_win_rate = home_stats["Wins"] / max(home_stats["Total Matches"], 1)
    away_win_rate = away_stats["Wins"] / max(away_stats["Total Matches"], 1)

    home_values = [
        normalize(home_stats["Rating"], 1500, 2100),
        home_win_rate * 100,
        normalize(home_stats["Goals For"], 0, 250),
        normalize(home_stats["Goals Against"], 0, 200, invert=True),
        normalize(home_stats["Total Matches"], 0, 120),
    ]
    away_values = [
        normalize(away_stats["Rating"], 1500, 2100),
        away_win_rate * 100,
        normalize(away_stats["Goals For"], 0, 250),
        normalize(away_stats["Goals Against"], 0, 200, invert=True),
        normalize(away_stats["Total Matches"], 0, 120),
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=home_values + [home_values[0]],
                                   theta=metrics + [metrics[0]],
                                   fill="toself", name=home_team, line_color="#1f77b4"))
    fig.add_trace(go.Scatterpolar(r=away_values + [away_values[0]],
                                   theta=metrics + [metrics[0]],
                                   fill="toself", name=away_team, line_color="#d62728"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Radar Perbandingan Performa",
        height=450,
    )
    return fig


def render_probability_pie(home_team, away_team, home_pct, away_pct):
    """Menampilkan pie chart probabilitas kemenangan kedua tim."""
    fig = go.Figure(data=[go.Pie(
        labels=[home_team, away_team],
        values=[home_pct, away_pct],
        hole=0.45,
        marker_colors=["#1f77b4", "#d62728"],
    )])
    fig.update_traces(textinfo="label+percent", textfont_size=14)
    fig.update_layout(title="Probabilitas Kemenangan", height=420)
    return fig


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar(ranking_df: pd.DataFrame):
    """Membuat sidebar untuk memilih babak dan kedua tim yang akan bertanding."""
    st.sidebar.title("⚽ Pengaturan Pertandingan")

    babak = st.sidebar.selectbox(
        "Pilih Babak",
        options=["Perempat Final", "Semifinal", "Final"],
    )

    team_list = sorted(ranking_df["Team"].unique().tolist())

    home_team = st.sidebar.selectbox("Pilih Tim Home", options=team_list, index=0)
    away_options = [t for t in team_list if t != home_team]
    away_team = st.sidebar.selectbox("Pilih Tim Away", options=away_options, index=0)

    st.sidebar.markdown("---")
    predict_clicked = st.sidebar.button("🔮 Prediksi", use_container_width=True, type="primary")

    st.sidebar.markdown("---")
    with st.sidebar.expander("🌐 Ambil Data Terbaru (eloratings.net)"):
        st.caption(
            "Mengambil ranking Elo & riwayat pertandingan TERKINI langsung dari "
            "eloratings.net. Proses ini butuh koneksi internet dan bisa memakan "
            "waktu beberapa menit, lalu model perlu dilatih ulang."
        )
        refresh_clicked = st.button("🔄 Refresh Data & Latih Ulang", use_container_width=True)
        if refresh_clicked:
            st.session_state["refresh_data_requested"] = True

    st.sidebar.caption(
        "Model: Random Forest Classifier\n\n"
        "Dataset contoh dibuat secara sintetis untuk demonstrasi. "
        "Ganti data/ranking.csv & data/matches.csv dengan data asli, atau gunakan "
        "tombol Refresh di atas, untuk hasil yang lebih akurat."
    )

    return babak, home_team, away_team, predict_clicked


# ============================================================
# MAIN APP
# ============================================================
def main():
    st.title("🏆 Prediksi Hasil Pertandingan Babak Gugur Piala Dunia FIFA")
    st.markdown(
        "Aplikasi ini memprediksi peluang kemenangan dua tim yang akan "
        "bertanding pada babak **Perempat Final**, **Semifinal**, atau **Final** "
        "berdasarkan data historis pertandingan dan statistik tim, menggunakan model "
        "**Machine Learning (Random Forest Classifier)**."
    )

    # ----------------------------------------------------------
    # Tangani permintaan refresh data dari eloratings.net (jika diminta)
    # ----------------------------------------------------------
    if st.session_state.get("refresh_data_requested"):
        st.session_state["refresh_data_requested"] = False
        with st.spinner("Mengambil data terbaru dari eloratings.net & melatih ulang model... "
                         "Ini bisa memakan waktu beberapa menit."):
            try:
                import fetch_live_data
                import train_model as tm

                fetch_live_data.main()
                tm.train()

                get_ranking_data.clear()
                get_model_bundle.clear()
                st.success("Data & model berhasil diperbarui dengan data terkini!")
            except Exception as e:
                st.error(
                    f"Gagal mengambil data terbaru: {e}\n\n"
                    "Pastikan library `datafc` terinstall (`pip install datafc`) "
                    "dan komputer Anda terhubung ke internet."
                )

    ranking_df = get_ranking_data()
    model_bundle = get_model_bundle()

    if model_bundle is None:
        st.error(
            "Model belum ditemukan (model/model.pkl). "
            "Jalankan `python train_model.py` terlebih dahulu untuk melatih model."
        )
        st.stop()

    babak, home_team, away_team, predict_clicked = render_sidebar(ranking_df)

    if not predict_clicked:
        st.info("Pilih babak dan kedua tim pada sidebar, lalu tekan tombol **Prediksi**.")
        st.subheader("📊 Dataset Ranking Tim (Contoh)")
        st.dataframe(ranking_df, use_container_width=True)
        return

    if home_team == away_team:
        st.warning("Tim Home dan Away tidak boleh sama. Silakan pilih tim yang berbeda.")
        return

    # ----------------------------------------------------------
    # Jalankan prediksi
    # ----------------------------------------------------------
    result = predict_match(home_team, away_team, ranking_df, model_bundle)
    win_pct = compute_win_percentages(result, home_team, away_team)
    home_stats = result["home_stats"]
    away_stats = result["away_stats"]

    # ----------------------------------------------------------
    # 1 & 2. Nama tim & babak pertandingan
    # ----------------------------------------------------------
    st.header(f"{home_team}  🆚  {away_team}")
    st.subheader(f"Babak: {babak} ({ROUND_LABELS[babak]})")

    # ----------------------------------------------------------
    # 3. Statistik kedua tim
    # ----------------------------------------------------------
    st.markdown("### 📈 Statistik Kedua Tim")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{home_team}**")
        st.metric("Elo Rating", format_number(home_stats["Rating"]))
        st.metric("Ranking", f"#{int(home_stats['Rating Rank'])}")
        st.metric("Total Match", format_number(home_stats["Total Matches"]))
        st.metric("Win Rate", f"{(home_stats['Wins'] / max(home_stats['Total Matches'],1)) * 100:.1f}%")
        st.metric("Goals For / Against", f"{format_number(home_stats['Goals For'])} / {format_number(home_stats['Goals Against'])}")
    with col2:
        st.markdown(f"**{away_team}**")
        st.metric("Elo Rating", format_number(away_stats["Rating"]))
        st.metric("Ranking", f"#{int(away_stats['Rating Rank'])}")
        st.metric("Total Match", format_number(away_stats["Total Matches"]))
        st.metric("Win Rate", f"{(away_stats['Wins'] / max(away_stats['Total Matches'],1)) * 100:.1f}%")
        st.metric("Goals For / Against", f"{format_number(away_stats['Goals For'])} / {format_number(away_stats['Goals Against'])}")

    # ----------------------------------------------------------
    # 4. Grafik perbandingan statistik
    # ----------------------------------------------------------
    st.markdown("### 📊 Grafik Perbandingan Statistik")
    tab1, tab2 = st.tabs(["Bar Chart", "Radar Chart"])
    with tab1:
        st.plotly_chart(render_stat_bar_chart(home_team, away_team, home_stats, away_stats),
                         use_container_width=True)
    with tab2:
        st.plotly_chart(render_radar_chart(home_team, away_team, home_stats, away_stats),
                         use_container_width=True)

    # ----------------------------------------------------------
    # 5, 6, 7. Prediksi hasil, probabilitas, confidence score
    # ----------------------------------------------------------
    st.markdown("### 🔮 Hasil Prediksi")

    pred_col1, pred_col2, pred_col3 = st.columns(3)
    pred_col1.metric("Prediksi Hasil", result["prediction"])
    pred_col2.metric("Confidence Score", f"{result['confidence'] * 100:.1f}%")
    winner = home_team if win_pct["home_win_pct"] >= win_pct["away_win_pct"] else away_team
    pred_col3.metric("Tim Diprediksi Menang", winner)

    prob_col1, prob_col2 = st.columns([1, 1])
    with prob_col1:
        st.markdown("**Probabilitas Kelas Model:**")
        prob_df = pd.DataFrame({
            "Hasil": list(result["probabilities"].keys()),
            "Probabilitas": [f"{v * 100:.1f}%" for v in result["probabilities"].values()],
        })
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
    with prob_col2:
        st.plotly_chart(
            render_probability_pie(home_team, away_team, win_pct["home_win_pct"], win_pct["away_win_pct"]),
            use_container_width=True,
        )

    # ----------------------------------------------------------
    # 8 & 9. Persentase peluang menang & kesimpulan otomatis
    # ----------------------------------------------------------
    st.markdown("### 📝 Kesimpulan")
    fav_team = home_team if win_pct["home_win_pct"] >= win_pct["away_win_pct"] else away_team
    fav_pct = max(win_pct["home_win_pct"], win_pct["away_win_pct"])
    underdog_team = away_team if fav_team == home_team else home_team
    underdog_pct = min(win_pct["home_win_pct"], win_pct["away_win_pct"])

    st.success(
        f"Berdasarkan model Machine Learning, **{fav_team}** memiliki peluang kemenangan "
        f"sebesar **{fav_pct:.0f}%**, sedangkan **{underdog_team}** memiliki peluang sebesar "
        f"**{underdog_pct:.0f}%**. Oleh karena itu, **{fav_team}** diprediksi memenangkan "
        f"pertandingan pada babak {babak} ini."
    )

    st.caption(
        "⚠️ Catatan: Prediksi ini bersifat statistik berdasarkan data historis dan "
        "tidak menjamin hasil pertandingan yang sesungguhnya."
    )


if __name__ == "__main__":
    main()
