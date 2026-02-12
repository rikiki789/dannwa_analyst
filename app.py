import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path even if run from another working dir.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import MAX_FILE_SIZE_MB, OPENAI_API_KEY, SILENCE_DB_THRESHOLD, SUPPORTED_FORMATS
from services import AudioProcessor, MemoGenerationService, TranscriptionService


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
    st.set_page_config(page_title="dannwa_analyst", page_icon="🎙️")
    st.title("🎙️ dannwa_analyst")
    st.caption("会話音声分析ツール - 全文文字起こし＋沈黙統計＋要点分析")

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY が見つかりません。.env を確認してください。")
        st.stop()

    uploaded_file = st.file_uploader(
        "音声ファイルをアップロード（MP3/WAV/M4A）",
        type=SUPPORTED_FORMATS,
    )

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
                silence_events = AudioProcessor.detect_silence(y, sr)
                silence_stats = AudioProcessor.calculate_silence_stats(silence_events)
                total_duration = AudioProcessor.get_duration(y, sr)
                rms_times, rms_db = AudioProcessor.rms_db(y, sr)

                transcript = TranscriptionService().transcribe(tmp_path)
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
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    if "silence_stats" in st.session_state:
        tabs = st.tabs(["沈黙統計", "分析メモ", "全文字起こし", "全沈黙一覧", "声量波形"])

        with tabs[0]:
            stats = st.session_state["silence_stats"]
            st.metric("全体の沈黙時間 (秒)", stats["total_silence_time"])
            st.metric("1.5-2秒 沈黙回数", stats["1.5-2s"]["count"])
            st.metric("2秒以上 沈黙回数", stats["2s+"]["count"])
            st.subheader("Top10 長い沈黙")
            st.dataframe(stats["longest_silences"], use_container_width=True)

        with tabs[1]:
            st.text_area("分析メモ", st.session_state["memo"], height=300)

        with tabs[2]:
            st.text_area("全文字起こし", st.session_state["transcript"], height=400)

        with tabs[3]:
            st.dataframe(st.session_state["silence_stats"]["all_silences"], use_container_width=True)

        with tabs[4]:
            st.line_chart({"dB": st.session_state["rms_db"]}, x=st.session_state["rms_times"], use_container_width=True)
            st.caption(f"沈黙判定しきい値: {SILENCE_DB_THRESHOLD} dB（最大音量=0 dB）")


if __name__ == "__main__":
    main()
