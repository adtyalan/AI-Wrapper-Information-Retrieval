import streamlit as st
import torch
import pandas as pd
import altair as alt
from sentence_transformers import SentenceTransformer, CrossEncoder
from PyPDF2 import PdfReader
import math
import re
from scipy.stats import kendalltau

st.set_page_config(page_title="Perbandingan Bi-Encoder vs Cross-Encoder", layout="wide")


# --- Fungsi-fungsi di-cache untuk performa ---
@st.cache_resource
def load_models():
    bi_encoder = SentenceTransformer(
        "sentence-transformers/msmarco-distilbert-base-v3", device="cpu"
    )
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    return bi_encoder, cross_encoder


@st.cache_data
def process_document(file_content, file_type, chunking_method):
    full_document_text = ""
    if file_type == "application/pdf":
        try:
            pdf = PdfReader(file_content)
            full_document_text = "\n".join(
                page.extract_text() for page in pdf.pages if page.extract_text()
            )
        except Exception as e:
            st.error(f"Gagal membaca PDF: {e}")
            return []
    else:  # TXT
        try:
            full_document_text = file_content.getvalue().decode("utf-8")
        except Exception as e:
            st.error(f"Gagal membaca TXT: {e}")
            return []

    cleaned_text = clean_text(full_document_text)
    if chunking_method == "Per Pasal":
        return chunk_by_pasal(cleaned_text)
    else:  # Per BAB
        return chunk_by_bab(cleaned_text)


def clean_text(full_text):
    cleaned_text = re.sub(r"([a-zA-Z,])\n([a-zA-Z])", r"\1 \2", full_text)
    cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)
    return cleaned_text


def chunk_by_pasal(cleaned_text):
    parts = re.split(r"(\n\s*Pasal\s+\d+[A-Z]?)", cleaned_text, flags=re.IGNORECASE)
    passages = [parts[0].strip()] if parts[0].strip() else []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            chunk = (parts[i] + parts[i + 1]).strip()
            if len(chunk) > 70 and chunk.lower().count("pasal") < 4:
                passages.append(chunk)
    return passages


def chunk_by_bab(cleaned_text):
    parts = re.split(
        r"(\n\s*BAB\s+[IVXLCDM]+.*|ATURAN\s+PERALIHAN|ATURAN\s+TAMBAHAN)",
        cleaned_text,
        flags=re.IGNORECASE,
    )
    passages = [parts[0].strip()] if parts[0].strip() else []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            chunk = (parts[i] + parts[i + 1]).strip()
            if len(chunk) > 100:
                passages.append(chunk)
    return passages


def display_placeholder():
    st.markdown(
        """
        <div style="background-color:#f8f9fa; border:2px dashed #e9ecef; border-radius:0.5rem; padding:3rem 1rem; text-align:center; margin-top:1rem;">
            <h3 style="color:#6c757d;">Menunggu Analisis</h3>
            <p style="color:#6c757d; font-size:1.1em;">Silakan klik tombol <b>'Jalankan Analisis'</b> di sidebar untuk menampilkan hasilnya di sini.</p>
        </div>
    """,
        unsafe_allow_html=True,
    )


# --- UI Aplikasi ---
bi_encoder, cross_encoder = load_models()

st.title("🔬 Aplikasi Komparasi Model Information Retrieval")
st.caption("Membandingkan Kinerja Bi-Encoder vs. Cross-Encoder dengan Berbagai Metrik")

# --- PERUBAHAN DIMULAI DI SINI ---
with st.sidebar:
    st.header("⚙️ Pengaturan Analisis")

    st.subheader("1. Pilih Sumber Dokumen")
    use_example_doc = st.checkbox(
        "Gunakan Contoh Dokumen (UUD 1945)",
        value=True,
        help="Jika dicentang, akan menggunakan file UUD 1945 yang sudah tersedia. Anda perlu menempatkan file 'UUD1945.pdf' di direktori yang sama dengan aplikasi ini.",
    )
    uploaded_file = st.file_uploader(
        "Atau Unggah Dokumen Anda (PDF/TXT)",
        type=["pdf", "txt"],
        disabled=use_example_doc,
    )

    st.subheader("2. Atur Parameter")
    chunking_method = st.radio(
        "Metode Pemisahan Passages:",
        ("Per Pasal", "Per BAB"),
        help="**Per Pasal**: Granularitas tinggi, baik untuk query spesifik. **Per BAB**: Konteks luas, baik untuk query konseptual.",
    )
    query = st.text_area("Masukkan Query:", "hak dan kewajiban warga negara")

    # Tombol Jalankan Analisis
    # Tombol akan dinonaktifkan jika tidak ada dokumen yang dipilih (baik contoh maupun unggahan)
    is_doc_ready = use_example_doc or uploaded_file is not None
    run_button = st.button(
        "🔍 Jalankan Analisis",
        use_container_width=True,
        type="primary",
        disabled=not is_doc_ready,
    )
# --- PERUBAHAN SELESAI ---

# --- Logika Pemrosesan ---
document_source = None
file_type = None

if use_example_doc:
    try:
        document_source = open("UUD1945.pdf", "rb")
        file_type = "application/pdf"
    except FileNotFoundError:
        st.error(
            "File contoh 'UUD1945.pdf' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama dengan `app.py`."
        )
        st.stop()
elif uploaded_file:
    document_source = uploaded_file
    file_type = uploaded_file.type

if document_source:
    # Menggunakan session state untuk menyimpan passages agar tidak diproses ulang terus-menerus
    # Key unik berdasarkan nama file atau status checkbox
    doc_id = "UUD1945_example" if use_example_doc else uploaded_file.name
    if (
        "current_doc_id" not in st.session_state
        or st.session_state.current_doc_id != doc_id
        or st.session_state.chunk_method != chunking_method
    ):
        st.session_state.current_doc_id = doc_id
        st.session_state.chunk_method = chunking_method
        st.session_state.passages = process_document(
            document_source, file_type, chunking_method
        )
        st.session_state.results_generated = (
            False  # Reset hasil jika file/metode berubah
        )
        st.sidebar.success(
            f"✅ Dokumen diproses.\n\n**{len(st.session_state.passages)}** passages terdeteksi."
        )

    passages = st.session_state.passages

    if run_button:
        if not query or not passages:
            st.warning("Harap isi query dan pastikan dokumen terproses dengan benar.")
        else:
            with st.spinner("⏳ Menganalisis relevansi..."):
                # Sisa logika analisis sama persis
                scores_bi = bi_encoder.encode(query, convert_to_tensor=True)
                passage_embs = bi_encoder.encode(passages, convert_to_tensor=True)
                from sentence_transformers.util import cos_sim

                scores_bi = cos_sim(scores_bi, passage_embs)[0].cpu().tolist()
                pairs = [[query, passage] for passage in passages]
                scores_cross = cross_encoder.predict(pairs).tolist()
                results = [
                    {"passage": p, "score_bi": sb, "score_cross": sc}
                    for p, sb, sc in zip(passages, scores_bi, scores_cross)
                ]
                st.session_state.results = results
                st.session_state.results_generated = True

# --- Tampilan Hasil dengan Tab ---
if document_source:
    tab1, tab2, tab3 = st.tabs(
        ["📊 Perbandingan Peringkat", "📈 Metrik & Statistik", "📉 Grafik Skor"]
    )
    is_results_ready = st.session_state.get("results_generated", False)

    with tab1:
        if is_results_ready:
            # Kode untuk menampilkan tabel (sama seperti sebelumnya)
            results = st.session_state.results
            sorted_bi = sorted(results, key=lambda x: x["score_bi"], reverse=True)
            sorted_cross = sorted(results, key=lambda x: x["score_cross"], reverse=True)
            df_bi = pd.DataFrame(
                [
                    {"Rank": i + 1, "Passage": res["passage"], "Score": res["score_bi"]}
                    for i, res in enumerate(sorted_bi)
                ]
            )
            df_cross = pd.DataFrame(
                [
                    {
                        "Rank": i + 1,
                        "Passage": res["passage"],
                        "Score": res["score_cross"],
                    }
                    for i, res in enumerate(sorted_cross)
                ]
            )
            st.header("Tabel Peringkat Hasil")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Peringkat Bi-Encoder")
                st.dataframe(
                    df_bi,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            format="%.4f", min_value=0, max_value=1
                        )
                    },
                )
            with col2:
                st.subheader("Peringkat Cross-Encoder")
                st.dataframe(
                    df_cross,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn(format="%.4f")
                    },
                )
        else:
            display_placeholder()

    # Kode untuk tab2 dan tab3 sama persis dengan versi sebelumnya,
    # masing-masing dengan blok `if is_results_ready: ... else: display_placeholder()`
    with tab2:
        if is_results_ready:
            # (Salin kode dari tab2 versi sebelumnya ke sini)
            results = st.session_state.results
            sorted_bi = sorted(results, key=lambda x: x["score_bi"], reverse=True)
            sorted_cross = sorted(results, key=lambda x: x["score_cross"], reverse=True)
            st.header("Metrik Evaluasi & Statistik")
            st.subheader("Perbandingan Hasil Peringkat Teratas (Top-1)")
            top_bi_passage = sorted_bi[0]["passage"]
            top_cross_passage = sorted_cross[0]["passage"]
            if top_bi_passage == top_cross_passage:
                st.success("✅ Kedua model setuju pada passage paling relevan.")
                with st.expander("Lihat Passage Teratas"):
                    st.info(top_bi_passage)
            else:
                st.warning("❌ Kedua model berbeda dalam menentukan passage teratas.")
                with st.expander("Lihat Perbandingan Passage Teratas"):
                    c1, c2 = st.columns(2)
                    c1.info(f"**Bi-Encoder:**\n\n{top_bi_passage}")
                    c2.info(f"**Cross-Encoder:**\n\n{top_cross_passage}")
            st.markdown("---")
            st.subheader("Metrik Kuantitatif")
            mrr_bi = 1 / (pd.Series([r["score_bi"] for r in sorted_bi]).idxmax() + 1)
            mrr_cross = 1 / (
                pd.Series([r["score_cross"] for r in sorted_cross]).idxmax() + 1
            )
            bi_rank_map = {res["passage"]: i for i, res in enumerate(sorted_bi)}
            cross_ranks_for_bi_order = [0] * len(sorted_bi)
            for i, res in enumerate(sorted_cross):
                if res["passage"] in bi_rank_map:
                    cross_ranks_for_bi_order[bi_rank_map[res["passage"]]] = i
            tau, p_value = kendalltau(range(len(results)), cross_ranks_for_bi_order)
            c1, c2 = st.columns(2)
            with c1:
                st.metric(label="MRR Bi-Encoder", value=f"{mrr_bi:.4f}")
                st.metric(label="MRR Cross-Encoder", value=f"{mrr_cross:.4f}")
            with c2:
                st.metric(
                    label="Korelasi Peringkat (Kendall's Tau)", value=f"{tau:.4f}"
                )
                st.caption(f"P-value: {p_value:.4f}")
            with st.expander("Penjelasan Metrik"):
                st.markdown(
                    """1. **MRR (Mean Reciprocal Rank)**: Memberi Anda gambaran seberapa cepat masing-masing model menemukan passage yang menurutnya sendiri paling relevan. Ini adalah metrik performa individu yang baik dalam konteks ini. Semakin tinggi **MRR**, berarti dokumen relevan muncul lebih cepat dalam hasil pencarian, menunjukkan efektivitas sistem dalam menampilkan informasi yang diharapkan."""
                )
                st.markdown(
                    """2. **Kendall's Tau (Korelasi Peringkat)**: Memberi Anda gambaran seberapa "sepakat" atau "tidak sepakat" kedua model tersebut dalam menyusun keseluruhan daftar peringkat. Ini adalah metrik perbandingan yang sangat kuat. Untuk **Kendall Tau**, skor yang tinggi berarti urutan hasil pencarian sangat mirip dengan peringkat ideal, menandakan kesesuaian sistem dalam menyusun relevansi dokumen."""
                )
        else:
            display_placeholder()

    with tab3:
        if is_results_ready:
            # (Salin kode dari tab3 versi sebelumnya ke sini)
            results = st.session_state.results
            sorted_bi = sorted(results, key=lambda x: x["score_bi"], reverse=True)
            top_n = 20
            chart_data = sorted_bi[:top_n]
            df_chart = pd.DataFrame(chart_data)
            st.header("Grafik Perbandingan Skor")
            st.caption(
                f"Grafik menampilkan perbandingan skor untuk {min(top_n, len(df_chart))} passages teratas menurut Bi-Encoder."
            )
            df_melted = df_chart.melt(
                id_vars="passage",
                value_vars=["score_bi", "score_cross"],
                var_name="model",
                value_name="score",
            )
            df_melted["model"] = df_melted["model"].map(
                {"score_bi": "Bi-Encoder", "score_cross": "Cross-Encoder"}
            )
            chart = (
                alt.Chart(df_melted)
                .mark_bar()
                .encode(
                    x=alt.X("score:Q", title="Skor Relevansi"),
                    y=alt.Y(
                        "passage:N",
                        sort=alt.EncodingSortField(
                            field="score", op="max", order="descending"
                        ),
                        title="Passage",
                        axis=alt.Axis(labelLimit=300),
                    ),
                    color=alt.Color("model:N", title="Model"),
                    tooltip=[
                        alt.Tooltip("passage:N", title="Passage"),
                        alt.Tooltip("model:N", title="Model"),
                        alt.Tooltip("score:Q", title="Score", format=".4f"),
                    ],
                )
                .properties(
                    title=f"Perbandingan Skor untuk Top {min(top_n, len(df_chart))} Passages",
                    height=alt.Step(25),
                )
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            display_placeholder()
else:
    st.info(
        "👋 Selamat Datang! Untuk memulai, pilih sumber dokumen Anda di sidebar kiri."
    )
