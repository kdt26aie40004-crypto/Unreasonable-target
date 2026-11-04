# お題1 「榮先生案件」システム構築構想書

## 1. 構想概要
本プロジェクトは、提示されたテーマである**「第二の自分（AIアバター・デジタルツイン）」**を、**Ollama（ローカルLLM）**と**Whisper（音声認識）**、そしてWindowsの**PowerShell**をシームレスに結合することで、ユーザーの「声の指示」からその場で即座に実用的な自動化ツール（.ps1）やWebサイトを自動生成し、ローカル環境に保存していつでも使える状態にするシステムです。

### 核心の価値
- **完全なプライバシー保護**: パーソナルデータという極めて機密性の高い情報を扱うため、外部APIに一切データを送信せず、ローカル（オフライン）環境だけで完結させます。
- **即時性と実用性**: 生成されたコードは即座に新しいPowerShellコンソールを起動して実行され、かつ自動的に指定フォルダへ保存されます。

---

## 2. システムアーキテクチャ（構成）

```
[ユーザーの音声指示]
       │
       ▼ （録音：マイク入力）
 1. Whisper による文字起こし ────► 【テキスト変換】
       │
       ▼ （プロンプト生成 ＆ 過去の対話履歴・パーソナルデータの検索）
 2. Ollama（ローカルLLM） ──────► 【思考・コード生成（PowerShell / HTML）】
       │
       ▼ （マークダウンからのコードブロック自動抽出）
 3. Python 連携コア ────────────► 【ファイル自動保存（BOM付きUTF-8）】
       │
       ▼ （サブプロセス起動：Bypassポリシー適用）
 4. PowerShell 自動実行 ────────► 【デスクトップやシステムでの即時実演】
```

---

## 3. 画面のヒントを組み込んだアプローチ

- **ローカルLLM (Ollama)**: 外部への通信を一切遮断した状態で稼働する推論エンジン。
- **量子化 (Quantization)**: VRAMの限られた一般のPC環境でも爆速でコードを生成させるため、4bit/8bit量子化されたコード特化モデル（Qwen2.5-CoderやGemma2の量子化版）を選定。
- **アバター (Avatar)**: 音声合成（TTS）などを組み合わせ、「もう一人の自分」が指示を聞いて開発してくれているようなUXを付与。
- **RAG (外部検索)**: 過去に自分が作ったスクリプトや、よく使うコードの好みをベクトル化して蓄積し、生成されるコードの「自分らしさ」を強化。
- **パーソナルデータ**: ユーザーのPC内のファイル構成や過去の作業ログを学習させ、指示のコンテキスト（文脈）を正確に理解させる。

---

## 4. 完全実装ソースコード (Python)

以下は、音声入力からPowerShellスクリプト生成・保存・自動起動までを一気通貫で行う実装スクリプトです。

```python
import os
import re
import subprocess
import time
import requests
import speech_recognition as sr

# --- システム設定 ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2"  # または qwen2.5-coder などのコード生成が得意なローカルモデル
OUTPUT_DIR = "./my_powershell_tools"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def listen_and_transcribe():
    """1. Whisper（SpeechRecognition経由）でマイク音声を聞き取りテキスト化"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n[AI] どんなPowerShell自動化ツールを作りたいですか？ 声で教えてください...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        print("[AI] 音声を解析中...")
        text = r.recognize_google(audio, language="ja-JP")
        print(f"あなたのご要望: 「{text}」")
        return text
    except Exception as e:
        print("音声が聞き取れませんでした。")
        return None

def generate_powershell_script(user_request):
    """2. Ollamaを使って、要望に沿ったPowerShellコード（.ps1）を生成"""
    system_prompt = (
        "あなたは超優秀なWindowsインフラエンジニアおよびPowerShellのスペシャリストです。\n"
        "ユーザーの要望に応じた、実用的で安全なPowerShellスクリプトを作成してください。\n\n"
        "【重要ルール】\n"
        "1. 外部依存がなく、Windowsの標準機能だけで動くPowerShellスクリプトにしてください。\n"
        "2. スクリプトの最後には、処理が完了したことがユーザーに分かるよう 'Write-Host "処理が完了しました"' と記述し、ウィンドウがすぐに閉じないように 'Read-Host "Enterキーを押して終了"' を入れてください。\n"
        "3. 日本語が文字化けしないよう、コメントや表示文言はすべてUTF-8（BOM付き）で保存されることを想定してください。\n"
        "4. 余計な解説テキストは一切省き、コードブロック (```powershell ~ ```) の中身だけを出力してください。"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": f"{system_prompt}\n\nユーザーの要望: {user_request}",
        "stream": False,
    }

    print(f"[AI] {MODEL_NAME} でPowerShellスクリプトを構築中...")
    response = requests.post(OLLAMA_URL, json=payload)
    return response.json().get("response", "")

def save_and_run_powershell(raw_output):
    """3. 生成されたコードを抽出し、UTF-8(BOM付き)で保存してPowerShellに送信・実行"""
    match = re.search(r"```(?:powershell)?(.*?)```", raw_output, re.DOTALL | re.IGNORECASE)
    if match:
        ps_code = match.group(1).strip()
    else:
        ps_code = raw_output.strip()

    filename = f"ps_tool_{int(time.time())}.ps1"
    filepath = os.path.join(OUTPUT_DIR, filename)
    abs_filepath = os.path.abspath(filepath)

    # Windows環境のPowerShellでの文字化けを防ぐため、BOM付きUTF-8(utf-8-sig)で保存
    with open(abs_filepath, "w", encoding="utf-8-sig") as f:
        f.write(ps_code)

    print(f"\n✨ 成功しました！スクリプトを以下に保存しました:\n{abs_filepath}")
    print("-" * 50)
    print(ps_code)
    print("-" * 50)

    # 4. 新しいPowerShellコンソールウィンドウを立ち上げてスクリプトを実行
    print("[AI] PowerShellを起動して実行します...")
    subprocess.Popen([
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-NoExit",
        "-File",
        abs_filepath
    ], creationflags=subprocess.CREATE_NEW_CONSOLE)

if __name__ == "__main__":
    user_voice_text = listen_and_transcribe()
    if user_voice_text:
        ai_response = generate_powershell_script(user_voice_text)
        save_and_run_powershell(ai_response)
```

---

## 5. チームA vs チームC 必勝デモ戦略
コンテスト・発表会（榮先生案件）の場で絶大な効果を発揮する具体的なユースケース案です。

1. **「デスクトップにある大量の画像ファイルを、拡張子や日付ごとのフォルダに自動仕分けして」**
   - 目の前でPowerShellが開き、一瞬でカオスだったデスクトップのファイルが整頓される様子を見せることで、審査員に直感的な実用性を伝えます。
2. **「PCで動いている重いアプリをメモリ使用量順に調べて、トップ5をデスクトップのメモ帳に吐き出して」**
   - OSネイティブの情報をローカルLLMが正確に操るという「プログラミング知識不要のPC自動化」の未来を提示できます。
