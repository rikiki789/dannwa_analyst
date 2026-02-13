import io
import os
import sys
import tempfile
import textwrap
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is on sys.path even if run from another working dir.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    HF_TOKEN,
    MAX_FILE_SIZE_MB,
    OPENAI_API_KEY,
    SILENCE_DB_OPTIONS,
    SILENCE_DB_THRESHOLD,
    SUPPORTED_FORMATS,
)
from services import AudioProcessor, MemoGenerationService, SpeakerDiarizationService, TranscriptionService


def _validate_upload(uploaded_file):
    if uploaded_file is None:
        return "音声ファイルをアップロードしてください。"
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_FORMATS:
        return f"対応形式は {', '.join(SUPPORTED_FORMATS)} です。"
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return f"ファイルサイズが上限({MAX_FILE_SIZE_MB}MB)を超えています。"
    return None


def main():
    st.set_page_config(page_title="音声転換ツール", page_icon="🎙️")
    st.markdown(
        """
        <style>
        :root {
            --bg: #ffffff;
            --card: #ffffff;
            --accent: #3f5bdc;
            --accent-soft: #e8edff;
            --text: #111827;
            --muted: #6b7280;
            --border: #e5e7eb;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #cfe8ff 0%, #ffffff 45%);
            color: var(--text);
        }
        header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
            display: none;
        }
        .block-container {
            padding-top: 2.5rem;
            max-width: 880px;
        }
        h1, h2, h3 {
            letter-spacing: 0.2px;
            font-weight: 600;
        }
        .app-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .app-subtitle {
            color: var(--muted);
            margin-bottom: 2.0rem;
        }
        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        }
        .card + .card {
            margin-top: 1.2rem;
        }
        .stButton>button {
            background: #ffffff;
            color: var(--accent);
            border: 1px solid var(--accent);
            border-radius: 10px;
            padding: 0.55rem 1.1rem;
            font-weight: 600;
        }
        .stButton>button:hover,
        .stButton>button:active {
            background: var(--accent);
            color: #fff;
        }
        .stDownloadButton>button {
            border-radius: 10px;
            border: 1px solid var(--accent);
            color: var(--accent);
            background: #fff;
            font-weight: 600;
        }
        [data-testid="stToggle"] div[role="switch"][aria-checked="true"] {
            background: var(--accent);
        }
        [data-testid="stToggle"] div[role="switch"] {
            background: #e5e7eb;
        }
        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="app-title">🎙️ 音声転換ツール</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">会話音声分析ツール - 全文文字起こし＋沈黙統計＋要点分析</div>',
        unsafe_allow_html=True,
    )

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY が見つかりません。.env を確認してください。")
        st.stop()

    st.markdown(
        '<div style="text-align:center;font-weight:700;font-size:1.1rem;margin:0.2rem 0 0.8rem 0;">'
        "音声ファイルをアップロード"
        "</div>",
        unsafe_allow_html=True,
    )
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        db_threshold = st.selectbox(
            "沈黙判定しきい値 (dB)",
            options=SILENCE_DB_OPTIONS,
            index=SILENCE_DB_OPTIONS.index(SILENCE_DB_THRESHOLD),
            help="数値が小さいほど沈黙判定が厳しくなります。",
        )
        uploaded_file = st.file_uploader(
            "upload",
            type=SUPPORTED_FORMATS,
            label_visibility="collapsed",
        )
        enable_diarization = st.toggle("話者分離を有効化", value=False)
        st.markdown("</div>", unsafe_allow_html=True)

    error = _validate_upload(uploaded_file)
    if error:
        st.info(error)
        return

    if st.button("分析開始"):
        with st.spinner("処理中..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                y, sr = AudioProcessor.load_audio(tmp_path)
                silence_events = AudioProcessor.detect_silence(y, sr, db_threshold=db_threshold)
                silence_stats = AudioProcessor.calculate_silence_stats(silence_events)
                total_duration = AudioProcessor.get_duration(y, sr)
                rms_times, rms_db = AudioProcessor.rms_db(y, sr)

                if enable_diarization:
                    transcript, segments = TranscriptionService().transcribe(tmp_path, return_segments=True)
                else:
                    transcript = TranscriptionService().transcribe(tmp_path, return_segments=False)
                    segments = []
                memo = MemoGenerationService().generate_memo(
                    transcript=transcript,
                    silence_stats=silence_stats,
                    total_duration=total_duration,
                )

                st.session_state["silence_stats"] = silence_stats
                st.session_state["transcript"] = transcript
                st.session_state["memo"] = memo
                st.session_state["duration"] = total_duration
                st.session_state["rms_times"] = rms_times
                st.session_state["rms_db"] = rms_db
                st.session_state["db_threshold"] = db_threshold
                st.session_state["speaker_lines"] = []

                if enable_diarization:
                    if not HF_TOKEN:
                        st.warning("HF_TOKEN が未設定のため話者分離をスキップしました。")
                    else:
                        try:
                            diarizer = SpeakerDiarizationService()
                            diar_segments = diarizer.diarize(tmp_path)
                            speaker_lines = []
                            for dseg in diar_segments:
                                start = dseg["start"]
                                end = dseg["end"]
                                speaker = dseg["speaker"]
                                texts = []
                                for tseg in segments:
                                    mid = (tseg["start"] + tseg["end"]) / 2
                                    if start <= mid <= end:
                                        texts.append(tseg["text"])
                                line = f"{speaker}: {''.join(texts).strip()}"
                                if line.strip() and line.strip() != f"{speaker}:":
                                    speaker_lines.append(line)
                            st.session_state["speaker_lines"] = speaker_lines
                        except Exception as exc:
                            st.warning(f"話者分離に失敗しました: {exc}")
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    if "silence_stats" in st.session_state:
        tabs = st.tabs(["沈黙統計", "分析メモ", "文字プレビュー", "沈黙プレビュー", "声量波形"])

        with tabs[0]:
            stats = st.session_state["silence_stats"]
            st.metric("全体の沈黙時間 (秒)", stats["total_silence_time"])
            st.metric("2秒以上 沈黙回数", stats["2s+"]["count"])
            st.subheader("Top10 長い沈黙")
            st.dataframe(stats["longest_silences"], use_container_width=True)
            summary_df = pd.DataFrame(
                [{
                    "total_silence_time_s": stats["total_silence_time"],
                    "long_count": stats["2s+"]["count"],
                    "long_total_time_s": stats["2s+"]["total_time"],
                }]
            )
            top10_df = pd.DataFrame(stats["longest_silences"])
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                summary_df.to_excel(writer, index=False, sheet_name="summary")
                top10_df.to_excel(writer, index=False, sheet_name="top10")
            st.download_button(
                label="沈黙統計をExcelでダウンロード",
                data=output.getvalue(),
                file_name="silence_stats.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with tabs[1]:
            st.text_area("分析メモ", st.session_state["memo"], height=300)
            st.download_button(
                label="分析メモをTXTでダウンロード",
                data=st.session_state["memo"],
                file_name="analysis_memo.txt",
                mime="text/plain",
            )

        with tabs[2]:
            transcript_text = st.session_state["transcript"] or ""
            if st.session_state.get("speaker_lines"):
                preview_lines = "\n".join(st.session_state["speaker_lines"][:10])
            else:
                wrapped = textwrap.wrap(transcript_text, width=80, replace_whitespace=False)
                preview_lines = "\n".join(wrapped[:10])
            st.text_area("文字プレビュー（10行）", preview_lines, height=260)
            st.download_button(
                label="全文字起こしをTXTでダウンロード",
                data=st.session_state["transcript"],
                file_name="transcript.txt",
                mime="text/plain",
            )

        with tabs[3]:
            all_silences_df = pd.DataFrame(st.session_state["silence_stats"]["all_silences"])
            st.dataframe(all_silences_df.head(10), use_container_width=True)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                all_silences_df.to_excel(writer, index=False, sheet_name="all_silences")
            st.download_button(
                label="全沈黙一覧をExcelでダウンロード",
                data=output.getvalue(),
                file_name="all_silences.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with tabs[4]:
            chart_data = {"time_s": st.session_state["rms_times"], "dB": st.session_state["rms_db"]}
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(st.session_state["rms_times"], st.session_state["rms_db"], linewidth=0.8)
            for e in st.session_state["silence_stats"]["all_silences"]:
                ax.axvspan(e["start"], e["end"], color="#93a7ff", alpha=0.12, linewidth=0)
                mid = (e["start"] + e["end"]) / 2
                ax.axvline(mid, color="#1f2a7a", linewidth=1.0, alpha=0.8)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("RMS dB")
            ax.axhline(st.session_state.get("db_threshold", SILENCE_DB_THRESHOLD), color="red", linestyle="--", linewidth=0.8)
            ax.set_title("Volume Waveform (RMS dB)")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            st.caption(
                f"沈黙判定しきい値: {st.session_state.get('db_threshold', SILENCE_DB_THRESHOLD)} dB（最大音量=0 dB）"
            )
            png = io.BytesIO()
            fig.savefig(png, format="png", dpi=150)
            plt.close(fig)
            st.download_button(
                label="声量波形をPNGでダウンロード",
                data=png.getvalue(),
                file_name="volume_waveform.png",
                mime="image/png",
            )
            waveform_df = pd.DataFrame(chart_data)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                waveform_df.to_excel(writer, index=False, sheet_name="waveform")
            st.download_button(
                label="声量波形をExcelでダウンロード",
                data=output.getvalue(),
                file_name="volume_waveform.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


if __name__ == "__main__":
    main()
