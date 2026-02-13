from openai import OpenAI
from config import OPENAI_API_KEY, WHISPER_MODEL


class TranscriptionService:
    """Whisper APIを使用した文字起こしサービス"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = WHISPER_MODEL
    
    def transcribe(self, audio_file_path, return_segments=False):
        """音声ファイルを文字起こし
        
        Args:
            audio_file_path: 音声ファイルパス
            
        Returns:
            str: 文字起こしテキスト
        """
        with open(audio_file_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language="ja",  # 日本語指定
                response_format="verbose_json",
            )

        text = transcript.text
        if not return_segments:
            return text

        segments = []
        for seg in getattr(transcript, "segments", []) or []:
            if isinstance(seg, dict):
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)
                text = seg.get("text", "")
            else:
                start = getattr(seg, "start", 0.0)
                end = getattr(seg, "end", 0.0)
                text = getattr(seg, "text", "")
            segments.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "text": text or "",
                }
            )
        return text, segments
