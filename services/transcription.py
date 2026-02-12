from openai import OpenAI
from config import OPENAI_API_KEY, WHISPER_MODEL


class TranscriptionService:
    """Whisper APIを使用した文字起こしサービス"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = WHISPER_MODEL
    
    def transcribe(self, audio_file_path):
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
                language="ja"  # 日本語指定
            )
        
        return transcript.text
