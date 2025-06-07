# Diasumsikan impor dan kode awal lainnya sama
# ... (kode impor dan setup awal)

# st.caption(
#  "Aplikasi ini membandingkan performa model Bi-Encoder (msmarco-distilbert-base-v3) dan Cross-Encoder (ms-marco-MiniLM-L-6-v2) dalam Information Retrieval."
# )

import streamlit as st
import torch
import pandas as pd
import altair as alt
from sentence_transformers import SentenceTransformer, CrossEncoder
from PyPDF2 import PdfReader
import nltk
import math

# 1. Call st.set_page_config() IMMEDIATELY after imports
st.set_page_config(
    page_title="Perbandingan Bi-Encoder vs Cross-Encoder", layout="centered"
)

# 2. Then, proceed with other imports or setup that might not be Streamlit commands,
#    or function definitions.
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
from nltk.tokenize import sent_tokenize


# 3. Now define your cached functions and call them
@st.cache_resource
def load_models():
    bi_encoder = SentenceTransformer(
        "sentence-transformers/msmarco-distilbert-base-v3", device="cpu"
    )
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    return bi_encoder, cross_encoder


bi_encoder, cross_encoder = load_models()

# 4. The rest of your Streamlit app
st.title("📄 Perbandingan Bi-Encoder dan Cross-Encoder dari Sentence Transformers")
st.caption(
    "Aplikasi ini membandingkan performa model Bi-Encoder (msmarco-distilbert-base-v3) dan Cross-Encoder (ms-marco-MiniLM-L-6-v2) dalam Information Retrieval."
)

st.markdown("Unggah file `.txt` atau `.pdf`, lalu masukkan query yang ingin dicari.")

# Upload File
uploaded_file = st.file_uploader("Unggah dokumen (PDF/TXT)", type=["pdf", "txt"])
passages = []


def extract_passages(text):
    if not isinstance(text, str):
        text = str(text)
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


if uploaded_file:
    if uploaded_file.type == "application/pdf":
        pdf = PdfReader(uploaded_file)
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                passages.extend(extract_passages(text))
    elif uploaded_file.type == "text/plain":
        stringio = uploaded_file.getvalue().decode("utf-8")
        passages = extract_passages(stringio)

st.write(f"📘 Jumlah passages terdeteksi: {len(passages)}")

query = st.text_input("Masukkan Query:", "Apa manfaat AI dalam kehidupan sehari-hari?")

if st.button("🔍 Jalankan"):
    if not query or not passages:
        st.warning("Harap unggah file dan isi query.")
    else:
        with st.spinner(
            "⏳ Menjalankan model... Ini mungkin memakan waktu beberapa saat."
        ):
            query_emb = bi_encoder.encode(query)
            passage_embs = bi_encoder.encode(passages)
            scores_bi = [
                torch.cosine_similarity(
                    torch.tensor(query_emb), torch.tensor(p), dim=0
                ).item()
                for p in passage_embs
            ]

            pairs = [[query, passage] for passage in passages]
            scores_cross = cross_encoder.predict(
                pairs
            ).tolist()  # Pastikan tolist() jika output numpy array

            st.session_state.results_generated = True
            st.session_state.passages_processed = passages[:]
            st.session_state.scores_bi = scores_bi[:]
            st.session_state.scores_cross = scores_cross[:]

            # --- Perubahan Dimulai Di Sini ---
            passage_to_score_bi = {
                p: s
                for p, s in zip(
                    st.session_state.passages_processed, st.session_state.scores_bi
                )
            }
            passage_to_score_cross = {
                p: s
                for p, s in zip(
                    st.session_state.passages_processed, st.session_state.scores_cross
                )
            }

            # Buat daftar passage yang diurutkan berdasarkan skor untuk setiap model
            # (passage, score)
            sorted_by_bi = sorted(
                passage_to_score_bi.items(), key=lambda item: item[1], reverse=True
            )
            sorted_by_cross = sorted(
                passage_to_score_cross.items(), key=lambda item: item[1], reverse=True
            )

            # Buat pemetaan passage ke rank untuk setiap model
            # {passage_text: rank_number}
            passage_to_rank_bi = {
                passage: rank + 1 for rank, (passage, score) in enumerate(sorted_by_bi)
            }
            passage_to_rank_cross = {
                passage: rank + 1
                for rank, (passage, score) in enumerate(sorted_by_cross)
            }

            # Simpan hasil urutan Bi-Encoder (digunakan juga untuk perbandingan Top-1)
            st.session_state.sorted_bi_results = sorted_by_bi

            table_data = []
            # Iterasi berdasarkan urutan Bi-Encoder sebagai default tampilan tabel
            for i, (passage_text, bi_score_val) in enumerate(
                st.session_state.sorted_bi_results
            ):
                bi_actual_rank = i + 1  # Rank Bi-Encoder sesuai urutan iterasi
                cross_score_val = passage_to_score_cross.get(passage_text, 0.0)
                cross_actual_rank = passage_to_rank_cross.get(
                    passage_text
                )  # Dapatkan rank Cross-Encoder untuk passage ini

                table_data.append(
                    {
                        "Passage": passage_text.replace("\n", " ").replace("\r", " "),
                        "Bi-Encoder Rank": bi_actual_rank,
                        "Bi-Encoder Score": bi_score_val,  # Biarkan sebagai float
                        "Cross-Encoder Rank": cross_actual_rank,
                        "Cross-Encoder Score": cross_score_val,  # Biarkan sebagai float
                    }
                )
            st.session_state.table_data = table_data
            # --- Perubahan Selesai Di Sini ---

            # Untuk perbandingan top-1, sorted_cross_passages_only masih relevan
            st.session_state.sorted_cross_passages_only = [
                item[0] for item in sorted_by_cross
            ]

            st.session_state.page = 1

if "page" not in st.session_state:
    st.session_state.page = 1

if st.session_state.get("results_generated", False):
    st.info("Hasil Perbandingan Model:")
    table_data_to_display = st.session_state.get("table_data", [])

    if not table_data_to_display:
        st.warning("Tidak ada data untuk ditampilkan. Coba jalankan ulang.")
    else:
        page_size = 10
        total_pages = math.ceil(len(table_data_to_display) / page_size)

        if total_pages > 0:
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                if st.session_state.page > 1:
                    if st.button("⬅️ Prev"):
                        st.session_state.page -= 1
                        st.rerun()
            with col3:
                if st.session_state.page < total_pages:
                    if st.button("Next ➡️"):
                        st.session_state.page += 1
                        st.rerun()
            with col2:
                max_page_val = max(1, total_pages)
                selected_page = st.number_input(
                    f"Halaman (1-{total_pages})",
                    min_value=1,
                    max_value=max_page_val,
                    value=st.session_state.page,
                    step=1,
                    key="page_selector",
                )
                if selected_page != st.session_state.page:
                    st.session_state.page = selected_page
                    st.rerun()
        else:
            st.write("Tidak ada data untuk dipaginasi.")

        start_idx = (st.session_state.page - 1) * page_size
        end_idx = start_idx + page_size
        current_page_data = table_data_to_display[start_idx:end_idx]

        df_table = pd.DataFrame(current_page_data)

        # --- Perubahan untuk Tampilan DataFrame ---
        # Tentukan urutan kolom yang diinginkan
        column_order = [
            "Passage",
            "Bi-Encoder Rank",
            "Bi-Encoder Score",
            "Cross-Encoder Rank",
            "Cross-Encoder Score",
        ]
        # Filter kolom yang ada di DataFrame untuk menghindari error jika ada yang hilang
        columns_to_display = [col for col in column_order if col in df_table.columns]

        # Konfigurasi kolom untuk format angka
        column_config_display = {
            "Bi-Encoder Score": st.column_config.NumberColumn(format="%.4f"),
            "Cross-Encoder Score": st.column_config.NumberColumn(format="%.4f"),
            "Bi-Encoder Rank": st.column_config.NumberColumn(format="%d"),
            "Cross-Encoder Rank": st.column_config.NumberColumn(format="%d"),
        }
        st.dataframe(
            df_table[columns_to_display],
            use_container_width=True,
            hide_index=True,
            column_config=column_config_display,
        )
        # --- Perubahan Selesai ---

        st.markdown("---")
        st.subheader("📊 Evaluasi dan Statistik")

        current_scores_bi = st.session_state.get("scores_bi", [])
        current_scores_cross = st.session_state.get("scores_cross", [])

        def compute_mrr(scores):
            scores_list = list(scores)
            if not scores_list:
                return 0.0
            if not any(isinstance(s, (int, float)) for s in scores_list):
                return 0.0
            max_score = max(scores_list)
            try:
                return (
                    1 / (scores_list.index(max_score) + 1)
                    if max_score in scores_list
                    else 0.0
                )
            except ValueError:
                return 0.0

        mrr_bi = compute_mrr(current_scores_bi)
        mrr_cross = compute_mrr(current_scores_cross)

        st.markdown(f"**Mean Reciprocal Rank (MRR) - *berdasarkan skor tertinggi***:")
        st.write(f"🔹 Bi-Encoder: `{mrr_bi:.3f}`")
        st.write(f"🔹 Cross-Encoder: `{mrr_cross:.3f}`")
        st.caption(
            "_Catatan: MRR di sini dihitung sebagai 1 dibagi dengan rank dari passage dengan skor tertinggi yang ditemukan oleh masing-masing model._"
        )

        st.markdown("**Apakah Top-1 Sama?**")
        top_bi_passage = (
            st.session_state.sorted_bi_results[0][0]
            if st.session_state.sorted_bi_results
            else None
        )
        top_cross_passage = (
            st.session_state.sorted_cross_passages_only[0]
            if st.session_state.sorted_cross_passages_only
            else None
        )

        if top_bi_passage and top_cross_passage:
            if top_bi_passage == top_cross_passage:
                st.success("✅ Kedua model memilih passage paling relevan yang sama.")
                st.markdown(
                    f"<small>Passage: `{top_bi_passage[:100].replace('`','')}...`</small>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning("❌ Model berbeda dalam menentukan passage paling relevan.")
                st.markdown(
                    f"<small>Top Bi-Encoder: `{top_bi_passage[:100].replace('`','')}...`</small>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<small>Top Cross-Encoder: `{top_cross_passage[:100].replace('`','')}...`</small>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Tidak dapat menentukan top-1 passage (mungkin tidak ada hasil).")

        df_chart_data = pd.DataFrame(
            {
                "Passage": st.session_state.passages_processed,
                "Bi-Encoder Score": st.session_state.scores_bi,
                "Cross-Encoder Score": st.session_state.scores_cross,
            }
        )

        MAX_PASSAGES_IN_CHART = 20
        if len(df_chart_data) > MAX_PASSAGES_IN_CHART:
            # Menggunakan sorted_bi_results untuk mendapatkan top N passages
            top_n_passages_for_chart = [
                item[0]
                for item in st.session_state.sorted_bi_results[:MAX_PASSAGES_IN_CHART]
            ]
            df_chart_data = df_chart_data[
                df_chart_data["Passage"].isin(top_n_passages_for_chart)
            ]
            st.caption(
                f"Grafik menampilkan skor untuk {min(MAX_PASSAGES_IN_CHART, len(st.session_state.passages_processed))} passage teratas (berdasarkan Bi-Encoder)."
            )

        if not df_chart_data.empty:
            chart = (
                alt.Chart(df_chart_data.reset_index())
                .transform_fold(
                    ["Bi-Encoder Score", "Cross-Encoder Score"], as_=["Model", "Score"]
                )
                .mark_bar()
                .encode(
                    x=alt.X("Score:Q", title="Skor Relevansi"),
                    y=alt.Y(
                        "Passage:N",
                        sort=alt.EncodingSortField(
                            field="Score", op="max", order="descending"
                        ),
                        title="Passage",
                    ),
                    color="Model:N",
                    tooltip=["Passage:N", "Model:N", "Score:Q"],
                )
                .properties(
                    height=max(300, len(df_chart_data["Passage"].unique()) * 25)
                )  # Sedikit menambah tinggi per item
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("Tidak cukup data untuk menampilkan grafik.")

elif not uploaded_file or not query:
    st.info("Silakan unggah dokumen dan masukkan query, lalu klik 'Jalankan'.")
