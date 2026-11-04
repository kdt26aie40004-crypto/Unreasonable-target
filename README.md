# 第二の自分 — 声で作るPowerShellツール

声で指示すると、ローカルLLMがPowerShellスクリプトやWebサイトを生成し、保存してそのまま実行します。
外部APIには一切データを送信せず、すべてローカルで完結します。構想は [docs/system_architecture_concept.md](docs/system_architecture_concept.md) を参照。

```
[音声] → 1. Whisper（文字起こし） → 2. Ollama（履歴検索 + コード生成）
       → 3. Python（抽出・BOM付きUTF-8で保存） → 4. PowerShell（新しいコンソールで実行）
```

## セットアップ

1. [Ollama](https://ollama.com/) をインストールし、モデルを取得
   ```
   ollama pull qwen2.5-coder:7b
   ollama pull nomic-embed-text
   ```
2. Python の依存パッケージをインストール
   ```
   pip install -r requirements.txt
   ```

## 使い方

```
python main.py
```

Enterで録音開始 → 話す → Enterで録音終了。生成されたコードは `my_powershell_tools/` に保存されます。

`main.py` 冒頭の設定:
- `MODEL_NAME` — 使うローカルLLM
- `WHISPER_MODEL` — 文字起こしの精度と速度（`small` が標準）
- `CONFIRM_BEFORE_RUN` — 実行前に確認するか（`False` で即時実行）
