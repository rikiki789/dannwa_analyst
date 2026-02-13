from typing import List, Dict

from config import HF_TOKEN


class SpeakerDiarizationService:
    """pyannote.audio を使った話者分離"""

    def __init__(self):
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN が設定されていません。")
        try:
            from pyannote.audio import Pipeline
            from huggingface_hub import login
        except Exception as exc:
            raise RuntimeError("pyannote.audio がインストールされていません。") from exc
        # Authenticate once, then load without passing token arg (API compatibility).
        login(token=HF_TOKEN)
        self._pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        if self._pipeline is None:
            raise RuntimeError(
                "話者分離モデルの読み込みに失敗しました。Hugging Faceでモデル利用規約に同意済みか確認してください。"
            )

    def diarize(self, audio_path: str) -> List[Dict]:
        if self._pipeline is None:
            raise RuntimeError("話者分離パイプラインが初期化されていません。")
        diarization = self._pipeline(audio_path)
        segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "speaker": str(speaker),
                }
            )
        return segments
