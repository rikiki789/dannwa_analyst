import os
from pathlib import Path

from dotenv import load_dotenv

# Ensure .env is loaded even when the app is launched from another directory.
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
WHISPER_MODEL = "whisper-1"
GPT_MODEL = "gpt-4o-mini"

# Silence Detection Configuration
SILENCE_CONFIG = {
    "threshold_short": {"min": 1.5, "max": 2.0},  # 1.5-2秒
    "threshold_long": {"min": 2.0},               # 2秒以上
}
SILENCE_DB_OPTIONS = [-35.0, -40.0]  # dB choices (relative to max RMS)
SILENCE_DB_THRESHOLD = SILENCE_DB_OPTIONS[0]
MIN_SILENCE_DURATION = 0.5  # 秒

# Audio Processing
SUPPORTED_FORMATS = ["mp3", "wav", "m4a"]
MAX_FILE_SIZE_MB = 100
