import librosa
import numpy as np
from config import MIN_SILENCE_DURATION, SILENCE_CONFIG, SILENCE_DB_THRESHOLD


class AudioProcessor:
    """沈黙検出とオーディオ処理"""
    
    @staticmethod
    def load_audio(file_path):
        """音声ファイルを読み込む
        
        Args:
            file_path: 音声ファイルパス
            
        Returns:
            tuple: (y: 音声配列, sr: サンプリングレート)
        """
        y, sr = librosa.load(file_path, sr=None)
        return y, sr
    
    @staticmethod
    def detect_silence(y, sr, frame_length=2048, hop_length=512, db_threshold=SILENCE_DB_THRESHOLD):
        """沈黙を検出
        
        Args:
            y: 音声配列
            sr: サンプリングレート
            frame_length: フレーム長
            hop_length: ホップ長
            
        Returns:
            list: 沈黙イベントのリスト
        """
        # RMS（音量）をdBに変換（最大値=0dB）
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        silent_frames = rms_db < db_threshold
        
        # フレームから時間に変換
        times = librosa.frames_to_time(np.arange(len(silent_frames)), sr=sr, hop_length=hop_length)
        
        # 連続した沈黙区間を検出
        silence_events = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silent_frames):
            if is_silent and not in_silence:
                silence_start = times[i]
                in_silence = True
            elif not is_silent and in_silence:
                silence_end = times[i]
                duration = silence_end - silence_start
                
                # 規定秒以上の沈黙のみカウント
                if duration >= MIN_SILENCE_DURATION:
                    silence_events.append({
                        "start": round(silence_start, 2),
                        "end": round(silence_end, 2),
                        "duration": round(duration, 2),
                        "category": AudioProcessor._categorize_silence(duration)
                    })
                in_silence = False
        
        # 最後が沈黙で終わった場合
        if in_silence:
            silence_end = times[-1]
            duration = silence_end - silence_start
            if duration >= MIN_SILENCE_DURATION:
                silence_events.append({
                    "start": round(silence_start, 2),
                    "end": round(silence_end, 2),
                    "duration": round(duration, 2),
                    "category": AudioProcessor._categorize_silence(duration)
                })
        
        return silence_events

    @staticmethod
    def rms_db(y, sr, frame_length=2048, hop_length=512):
        """RMS音量(dB)と時間軸を取得"""
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        times = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=hop_length)
        return times, rms_db
    
    @staticmethod
    def _categorize_silence(duration):
        """沈黙の長さをカテゴリ分け
        
        Args:
            duration: 沈黙の継続時間（秒）
            
        Returns:
            str: カテゴリ名
        """
        if SILENCE_CONFIG["threshold_short"]["min"] <= duration < SILENCE_CONFIG["threshold_short"]["max"]:
            return "1.5-2s"
        elif duration >= SILENCE_CONFIG["threshold_long"]["min"]:
            return "2s+"
        else:
            return "other"
    
    @staticmethod
    def calculate_silence_stats(silence_events):
        """沈黙統計を計算
        
        Args:
            silence_events: 沈黙イベントのリスト
            
        Returns:
            dict: 統計情報
        """
        # 統計は2秒以上のみ対象
        short_silences = []
        long_silences = [e for e in silence_events if e["category"] == "2s+"]
        
        total_duration = sum(e["duration"] for e in long_silences)
        
        # Top10の長い沈黙を取得（2秒以上のみ）
        sorted_silences = sorted(long_silences, key=lambda x: x["duration"], reverse=True)[:10]
        
        stats = {
            "total_silence_time": round(total_duration, 2),
            "1.5-2s": {
                "count": len(short_silences),
                "total_time": round(sum(e["duration"] for e in short_silences), 2)
            },
            "2s+": {
                "count": len(long_silences),
                "total_time": round(sum(e["duration"] for e in long_silences), 2)
            },
            "longest_silences": sorted_silences,
            "all_silences": silence_events
        }
        
        return stats
    
    @staticmethod
    def get_duration(y, sr):
        """音声全体の長さを取得
        
        Args:
            y: 音声配列
            sr: サンプリングレート
            
        Returns:
            float: 音声の長さ（秒）
        """
        return round(librosa.get_duration(y=y, sr=sr), 2)
