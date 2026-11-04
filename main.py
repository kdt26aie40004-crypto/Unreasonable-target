"""第二の自分 — 声の指示からPowerShellツール / Webサイトを自動生成するシステム

構成（system_architecture_concept.md に準拠）:
  [ユーザーの音声指示]
   1. Whisper による文字起こし         -> transcribe()
   2. Ollama（ローカルLLM）            -> search_history() + generate_code()
   3. Python 連携コア                  -> extract_code() + save_code()
   4. PowerShell 自動実行              -> run_with_powershell()

すべてローカルで完結し、外部APIには一切データを送信しない。
"""

import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import requests
import sounddevice as sd
from faster_whisper import WhisperModel

# --- システム設定 ---
OLLAMA_BASE = "http://localhost:11434"
MODEL_NAME = "qwen2.5-coder:7b"      # 4bit量子化済みのコード特化モデル（gemma2 なども可）
EMBED_MODEL = "nomic-embed-text"     # RAG用の埋め込みモデル
WHISPER_MODEL = "small"              # tiny / base / small / medium / large-v3
SAMPLE_RATE = 16000                  # Whisper が想定するサンプリングレート
OUTPUT_DIR = "./my_powershell_tools"
HISTORY_FILE = os.path.join(OUTPUT_DIR, "history.jsonl")
CONFIRM_BEFORE_RUN = True            # 生成コードを実行前に確認する（False で即時実行）

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. Whisper による文字起こし
# ============================================================
def record_voice():
    """マイクから録音する（Enterで開始、もう一度Enterで終了）"""
    input("\n[AI] どんなツールを作りたいですか？ Enterで録音開始...")
    frames = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            callback=lambda data, *_: frames.append(data.copy()))
    with stream:
        input("[AI] 録音中... 話し終えたらEnter")
    if not frames:
        return None
    return np.concatenate(frames).flatten()


def transcribe(whisper, audio):
    """ローカルの Whisper で音声をテキスト化する"""
    print("[AI] 音声を解析中...")
    segments, _ = whisper.transcribe(audio, language="ja")
    text = "".join(s.text for s in segments).strip()
    if text:
        print(f"あなたのご要望: 「{text}」")
    else:
        print("音声が聞き取れませんでした。")
    return text or None


# ============================================================
# 2. Ollama（ローカルLLM）: 履歴検索 + コード生成
# ============================================================
def embed(text):
    """Ollama で文章をベクトル化する（埋め込みモデルが無ければ None）"""
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/embed",
                          json={"model": EMBED_MODEL, "input": text}, timeout=60)
        r.raise_for_status()
        return np.array(r.json()["embeddings"][0])
    except (requests.RequestException, KeyError, IndexError):
        return None


def search_history(user_request, top_k=2):
    """過去の対話履歴から、今回の要望に近いものを検索する（RAG）"""
    if not os.path.exists(HISTORY_FILE):
        return []
    query = embed(user_request)
    if query is None:
        return []
    scored = []
    with open(HISTORY_FILE, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            vec = np.array(item["embedding"])
            score = float(query @ vec / (np.linalg.norm(query) * np.linalg.norm(vec)))
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored[:top_k] if score > 0.5]


SYSTEM_PROMPT = """あなたはユーザー本人の「第二の自分」であり、超優秀なWindowsエンジニアです。
ユーザーの要望に応じて、次のどちらかを作成してください。
- PC操作の自動化 → PowerShellスクリプト（```powershell で囲む）
- Webサイト・画面 → 単一ファイルのHTML（```html で囲む。CSS/JSも同じファイル内に書く）

【重要ルール】
1. 外部依存がなく、Windowsの標準機能だけで動くこと。
2. PowerShellの場合、最後に Write-Host "処理が完了しました" を出力すること。
3. ファイルを削除・上書きする処理は避け、必要なら移動やコピーで代替すること。
4. 余計な解説は一切省き、コードブロック1つだけを出力すること。"""


def generate_code(user_request, related):
    """Ollama を使って、要望に沿ったコード（PowerShell / HTML）を生成する"""
    prompt = f"ユーザーの要望: {user_request}"
    if related:
        examples = "\n\n".join(f"要望: {h['request']}\n```{h['lang']}\n{h['code']}\n```"
                               for h in related)
        prompt = f"参考: 過去にあなたが作ったコード（書き方の好みを合わせること）\n{examples}\n\n{prompt}"

    print(f"[AI] {MODEL_NAME} でコードを構築中...")
    r = requests.post(f"{OLLAMA_BASE}/api/generate",
                      json={"model": MODEL_NAME, "system": SYSTEM_PROMPT,
                            "prompt": prompt, "stream": False},
                      timeout=600)
    r.raise_for_status()
    return r.json().get("response", "")


# ============================================================
# 3. Python 連携コア: コード抽出 + 保存
# ============================================================
LANG_ALIASES = {"powershell": "powershell", "ps1": "powershell", "pwsh": "powershell",
                "ps": "powershell", "html": "html"}


def extract_code(raw_output):
    """マークダウンからコードブロックを抽出し、(言語, コード) を返す"""
    match = re.search(r"```([\w-]*)\s*\n(.*?)```", raw_output, re.DOTALL)
    if not match:
        return "powershell", raw_output.strip()
    lang = LANG_ALIASES.get(match.group(1).lower(), "powershell")
    return lang, match.group(2).strip()


def save_code(lang, code):
    """BOM付きUTF-8で保存する（Windows PowerShell での文字化け防止）"""
    ext = "html" if lang == "html" else "ps1"
    path = os.path.abspath(os.path.join(OUTPUT_DIR, f"tool_{int(time.time())}.{ext}"))
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(code)
    print(f"\n✨ 成功しました！以下に保存しました:\n{path}")
    print("-" * 50)
    print(code)
    print("-" * 50)
    return path


def save_history(user_request, lang, code):
    """今回の対話を履歴に追加し、次回以降の検索対象にする"""
    vec = embed(user_request)
    if vec is None:
        return
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"request": user_request, "lang": lang, "code": code,
                            "embedding": vec.tolist()}, ensure_ascii=False) + "\n")


# ============================================================
# 4. PowerShell 自動実行
# ============================================================
DANGEROUS = re.compile(r"Remove-Item|\brm\b|\bdel\b|Format-Volume|Clear-Disk|Stop-Computer|"
                       r"Restart-Computer|Set-ExecutionPolicy|Invoke-Expression|\biex\b", re.IGNORECASE)


def run_with_powershell(lang, path, code):
    """新しいPowerShellコンソールを起動して実行する（HTMLはブラウザで開く）"""
    if CONFIRM_BEFORE_RUN:
        if lang == "powershell" and DANGEROUS.search(code):
            print("⚠ 削除・システム変更などの危険なコマンドが含まれています。内容をよく確認してください。")
        if input("[AI] 実行しますか？ (y/n): ").strip().lower() != "y":
            print("[AI] 実行をスキップしました（ファイルは保存済みです）。")
            return

    print("[AI] PowerShellを起動して実行します...")
    if lang == "html":
        command = ["powershell.exe", "-NoProfile", "-Command", f"Start-Process -FilePath '{path}'"]
    else:
        command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                   "-NoExit", "-File", path]
    subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)


# ============================================================
def main():
    # --text: マイクの代わりにキーボードで要望を入力する（Whisper を使わない）
    text_mode = "--text" in sys.argv
    if not text_mode:
        print("[AI] Whisper モデルを読み込み中...")
        whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    while True:
        if text_mode:
            user_request = input("\n[AI] どんなツールを作りたいですか？ 文字で入力してください: ").strip()
        else:
            audio = record_voice()
            if audio is None:
                continue
            user_request = transcribe(whisper, audio)
        if not user_request:
            continue

        try:
            related = search_history(user_request)
            raw = generate_code(user_request, related)
        except requests.RequestException as e:
            print(f"Ollama に接続できません（`ollama serve` が起動しているか確認してください）: {e}")
            continue

        lang, code = extract_code(raw)
        path = save_code(lang, code)
        save_history(user_request, lang, code)
        run_with_powershell(lang, path, code)

        if input("\n[AI] 続けて作りますか？ (y/n): ").strip().lower() != "y":
            break


if __name__ == "__main__":
    main()
