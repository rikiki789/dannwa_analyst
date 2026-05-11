from faster_whisper import WhisperModel
from config import WHISPER_MODEL_SIZE


class TranscriptionService:
    """faster-whisper を使用したローカル文字起こしサービス"""

    def __init__(self):
        self.model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",       # use "cuda" if GPU is available
            compute_type="int8", # int8 for CPU, float16 for GPU
        )

    def transcribe(self, audio_file_path, return_segments=False):
        """音声ファイルを文字起こし

        Args:
            audio_file_path: 音声ファイルパス
            return_segments: True の場合 (text, segments) のタプルを返す

        Returns:
            str or tuple: 文字起こしテキスト、または (テキスト, セグメントリスト)
        """
        segments, info = self.model.transcribe(
            audio_file_path,
            language="ja",
            beam_size=5,
            vad_filter=True,           # filter out silence for better accuracy
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )

        segments = list(segments)  # materialize generator

        full_text = "".join(seg.text for seg in segments)

        if not return_segments:
            return full_text

        segment_list = [
            {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text or "",
            }
            for seg in segments
        ]
        return full_text, segment_list
