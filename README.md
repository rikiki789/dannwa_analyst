# 🎙️ dannwa_analyst

会話音声をアップロードすると、**全文文字起こし＋沈黙統計＋要点分析**をまとめて返すアプリです。

## 🎯 主な機能

✅ **全文文字起こし** - Whisper APIで日本語音声を正確に変換  
✅ **沈黙検出＆統計** - 1.5～2秒、2秒以上の沈黙を自動検出  
✅ **分析メモ生成** - LLMが沈黙パターンから要点を自動生成  
✅ **全沈黙一覧** - 検出した全沈黙区間の時間を表示  

## 📋 セットアップ

### 1. 環境構築

```bash
# リポジトリクローン
cd dannwa_analyst

# 仮想環境作成（推奨）
python -m venv venv
source venv/bin/activate  # Mac/Linux
# または
venv\Scripts\activate  # Windows

# 依存ライブラリをインストール
pip install -r requirements.txt
```

### 2. API Key設定

```bash
# .env ファイルを作成
cp .env.example .env

# .env を編集してOpenAI API Keyを設定
# OPENAI_API_KEY=sk-your-api-key-here
```

## 🚀 使い方

```bash
streamlit run app.py
```

ブラウザで自動的に開きます。（デフォルト: http://localhost:8501）

### 操作フロー

1. **📁 ファイルアップロード**  
   - MP3, WAV, M4Aをアップロード

2. **🚀 分析開始**  
   - ボタンをクリックすると自動処理開始

3. **📊 結果確認**
   - **沈黙統計タブ**: 沈黙の統計情報とTop10
   - **分析メモタブ**: LLMが生成した要点分析
   - **全文字起こしタブ**: 音声の完全な文字起こし
   - **全沈黙一覧タブ**: 検出された全沈黙区間の時間

## 📁 ファイル構成

```
dannwa_analyst/
├─ app.py                      # Streamlit メインUI
├─ config.py                   # 設定（API key、パラメータ）
├─ requirements.txt            # 依存ライブラリ
├─ .env.example                # API Key設定用テンプレート
├─ .env                        # API Key（.gitignoreで除外）
└─ services/
   ├─ __init__.py
   ├─ audio_processor.py       # 沈黙検出
   ├─ transcription.py         # Whisper API統合
   └─ memo_generator.py        # LLM分析メモ生成
```

## ⚙️ パラメータ調整

### 沈黙検出の閾値（config.py）

```python
SILENCE_CONFIG = {
    "threshold_short": {"min": 1.5, "max": 2.0},  # 1.5-2秒
    "threshold_long": {"min": 2.0},               # 2秒以上
}
```

### 使用モデル（config.py）

```python
WHISPER_MODEL = "whisper-1"   # Whisper API
GPT_MODEL = "gpt-4o-mini"     # GPT モデル
```

他の LLM に切り替える場合は、`services/memo_generator.py` の `OpenAI` 部分を修正。

## 💡 技術詳細

### 沈黙検出アルゴリズム
- **Mel-spectrogramエネルギー**を計算
- エネルギーが低い（閾値以下）部分を沈黙と判定
- 連続した沈黙を1つのイベントにまとめる

### 文字起こし
- OpenAI Whisper API を使用
- 日本語に最適化（language="ja"）

### 分析メモ生成
- GPT-4o-mini で沈黙パターンを解析
- 特徴、注目区間、注意点を自動生成

## 🔧 トラブルシューティング

### "OPENAI_API_KEY not found"
→ `.env` ファイルが存在し、API Keyが正しく設定されているか確認

### 沈黙検出がうまくいかない
→ `audio_processor.py` の `threshold = 0.02` を調整

### 処理が遅い場合
→ Whisper API と GPT API の処理時間は音声の長さに依存します

## 📝 注意事項

- このアプリはローカル実行を想定
- OpenAI API の利用料金が発生します（$0.4程度/1時間音声）
- 医療情報などの個人情報は含めないように注意

## 🔄 今後の拡張

- [ ] 複数ファイル連続処理＆比較
- [ ] ダウンロード機能（JSON/CSV/TXT）
- [ ] 話者分離＆話者別統計
- [ ] タイムライン可視化
- [ ] 他の LLM への対応（Claude, Cohere等）

---

**開発:**  2026年2月  
**ステータス:** MVP完成
