from openai import OpenAI
from config import OPENAI_API_KEY, GPT_MODEL


class MemoGenerationService:
    """LLMを使用した分析メモ生成サービス"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = GPT_MODEL
    
    def generate_memo(self, transcript, silence_stats, total_duration):
        """文字起こしと沈黙統計から分析メモを生成
        
        Args:
            transcript: 文字起こしテキスト
            silence_stats: 沈黙統計情報
            total_duration: 音声全体の長さ（秒）
            
        Returns:
            str: 分析メモ
        """
        # 統計情報をプリントアウト用に整形
        total_silence = silence_stats["total_silence_time"]
        silence_percentage = round((total_silence / total_duration * 100), 1) if total_duration > 0 else 0
        
        short_count = silence_stats["1.5-2s"]["count"]
        short_time = silence_stats["1.5-2s"]["total_time"]
        
        long_count = silence_stats["2s+"]["count"]
        long_time = silence_stats["2s+"]["total_time"]
        
        longest_silence = max([e["duration"] for e in silence_stats["all_silences"]], default=0)
        
        # Prompt を作成
        prompt = f"""以下の会話音声の分析データを基に、簡潔な分析メモを生成してください。
        
【音声データ】
- 全体の長さ: {self._format_time(total_duration)}
- 文字起こし:
{transcript[:1000]}... (以下省略)

【沈黙統計】
- 全体の沈黙時間: {self._format_time(total_silence)} ({silence_percentage}%)
- 1.5～2秒の沈黙: {short_count}回（計{self._format_time(short_time)}）
- 2秒以上の沈黙: {long_count}回（計{self._format_time(long_time)}）
- 最長沈黙: {self._format_time(longest_silence)}

【出力形式】
以下の3項目を簡潔に出力してください：
1. 【特徴】 - 沈黙パターンや会話の流れの特徴（2-3行）
2. 【注目区間】 - 長い沈黙や重要そうな箇所の解釈（2-3行、具体的な時間を含む）
3. 【注意点】 - 音声品質や特記事項があれば（1-2行、なければ「なし」）

簡潔に、箇条書きで出力してください。"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "あなたは会話分析の専門家です。与えられたデータから簡潔で実用的な分析メモを生成します。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    @staticmethod
    def _format_time(seconds):
        """秒を MM:SS形式に変換
        
        Args:
            seconds: 秒数
            
        Returns:
            str: MM:SS形式の時間文字列
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m{secs}s"
