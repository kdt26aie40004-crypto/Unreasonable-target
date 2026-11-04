"""第二の自分 — 声の指示から業務を代わりにこなすシステム

構成（system_architecture_concept.md に準拠）:
  [ユーザーの音声指示]
   1. Whisper による文字起こし         -> transcribe()
   2. Ollama（ローカルLLM）            -> agent.py のエージェントが手順を考え、道具を順に使う
                                          （コード生成・返信・調整案などの生成はこのファイル）
   3. Python 連携コア                  -> extract_code() / save_code() / save_text()
   4. PowerShell 自動実行              -> run_ps() / delegate() / open_draft() / notify() / speak()

このファイルはエージェントが使う「道具」の中身。会話・判断は agent.py、画面は gui.py。
すべてローカルで完結し、外部APIには一切データを送信しない。
"""

import base64
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import requests
import sounddevice as sd
from faster_whisper import WhisperModel

# --- システム設定 ---
OLLAMA_BASE = "http://localhost:11434"
MODEL_NAME = "qwen2.5-coder:7b"      # 4bit量子化済みのコード特化モデル（gemma2 なども可）
TEXT_MODEL = "qwen2.5:7b"            # 返信・調整案・報告など文章を書くモデル（未取得なら MODEL_NAME を使う）
EMBED_MODEL = "nomic-embed-text"     # RAG用の埋め込みモデル
WHISPER_MODEL = "small"              # tiny / base / small / medium / large-v3
SAMPLE_RATE = 16000                  # Whisper が想定するサンプリングレート
OUTPUT_DIR = "./my_powershell_tools"
HISTORY_FILE = os.path.join(OUTPUT_DIR, "history.jsonl")
CONFIRM_BEFORE_RUN = True            # 生成コードを実行前に確認する（False で即時実行）
BROWSER = "chrome"                   # 作ったWebサイトを開くブラウザ（chrome / msedge / firefox）
NUM_CTX = 8192                       # LLMが一度に読める文章の量（標準の4096では足りないため）
VERBOSE = True                       # 生成したコードや保存先などの詳細を表示する（画面版では False）

# --- パーソナルデータ（GitHubには上げない） ---
DATA_DIR = "./data"
CALENDAR_DIR = os.path.join(DATA_DIR, "calendar")       # Google/Outlook から書き出した .ics を置く
MESSAGES_DIR = os.path.join(DATA_DIR, "my_messages")    # 自分が過去に書いた文章（.txt）を置く
PROFILE_FILE = os.path.join(DATA_DIR, "profile.md")     # 自分の情報・予定の優先順位
SCHEDULE_DAYS = 14                   # 何日先までダブルブッキングを確認するか
WORK_HOURS = (9, 21)                 # 予定を入れてよい時間帯（空き時間の計算に使う）
TASK_PREFIX = "SecondSelf_"          # タスクスケジューラに登録するタスク名の接頭辞

for d in (OUTPUT_DIR, CALENDAR_DIR, MESSAGES_DIR):
    os.makedirs(d, exist_ok=True)


# ============================================================
# 1. Whisper による文字起こし
# ============================================================
WHISPER = None  # main() で読み込む。None のときは文字入力モード


def record_voice(wait=True):
    """マイクから録音する（Enterで開始、もう一度Enterで終了）"""
    if wait:
        input("\n[AI] ご用件をどうぞ。Enterで録音開始...")
    frames = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            callback=lambda data, *_: frames.append(data.copy()))
    with stream:
        input("[AI] 録音中... 話し終えたらEnter")
    if not frames:
        return None
    return np.concatenate(frames).flatten()


def transcribe(whisper, audio, label="あなたのご要望"):
    """ローカルの Whisper で音声をテキスト化する"""
    print("[AI] 音声を解析中...")
    segments, _ = whisper.transcribe(audio, language="ja")
    text = "".join(s.text for s in segments).strip()
    if text:
        print(f"{label}: 「{text}」")
    else:
        print("音声が聞き取れませんでした。")
    return text or None


def ask_user(question):
    """宛先などを本人に聞く。文字で入力するか、音声モードなら Enter だけで声で答えられる"""
    if WHISPER is None:
        return input(f"[AI] {question}（Enterでスキップ）: ").strip("﻿ ")
    answer = input(f"[AI] {question}（文字で入力 / Enterで音声入力）: ").strip("﻿ ")
    if answer:
        return answer
    audio = record_voice(wait=False)
    text = (transcribe(WHISPER, audio, label="聞き取った内容") if audio is not None else None) or ""
    return text.rstrip("。、 ")


# ============================================================
# 2. Ollama（ローカルLLM）
# ============================================================
def progress(text):
    """処理の途中経過（詳細表示のときだけ出す）"""
    if VERBOSE:
        print(f"[AI] {text}")


def ask_ollama(system, prompt, json_mode=False, model=MODEL_NAME):
    """Ollama に問い合わせて応答テキストを返す（モデルが未取得なら MODEL_NAME で代用）"""
    payload = {"model": model, "system": system, "prompt": prompt, "stream": False,
               "options": {"num_ctx": NUM_CTX}}
    if json_mode:
        payload["format"] = "json"
    r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=600)
    if r.status_code == 404 and model != MODEL_NAME:
        return ask_ollama(system, prompt, json_mode, MODEL_NAME)
    r.raise_for_status()
    return r.json().get("response", "")


def embed(text):
    """Ollama で文章をベクトル化する（埋め込みモデルが無ければ None）"""
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/embed",
                          json={"model": EMBED_MODEL, "input": text}, timeout=60)
        r.raise_for_status()
        return np.array(r.json()["embeddings"][0])
    except (requests.RequestException, KeyError, IndexError):
        return None


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_profile():
    """自分の情報・優先順位（profile.md）を読み込む"""
    if not os.path.exists(PROFILE_FILE):
        return ""
    with open(PROFILE_FILE, encoding="utf-8") as f:
        return f.read().strip()


# --- tool: コード生成 ---
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
            scored.append((cosine(query, np.array(item["embedding"])), item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored[:top_k] if score > 0.5]


CODE_PROMPT = """あなたはユーザー本人の「第二の自分」であり、超優秀なWindowsエンジニアです。
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
    progress(f"{MODEL_NAME} でコードを構築中...")
    return ask_ollama(CODE_PROMPT, prompt)


EXPLAIN_PROMPT = """PowerShellスクリプトが何をするかを、プログラミングを知らない人にも分かるように説明してください。
箇条書きで3行以内、です・ます調で。ファイルの移動・削除・PCの設定変更があれば必ず書くこと。説明だけを出力すること。"""


def explain_code(code):
    """実行前の確認用に、コードがすることを普通の言葉で説明する"""
    try:
        return ask_ollama(EXPLAIN_PROMPT, code, model=TEXT_MODEL).strip()
    except requests.RequestException:
        return ""


# --- reply: 代理返信 ---
def find_my_messages(text, top_k=3):
    """自分が過去に書いた文章から、文体の参考になるものを検索する（RAG）"""
    paths = sorted(glob.glob(os.path.join(MESSAGES_DIR, "*.txt")))
    query = embed(text)
    if not paths or query is None:
        return []
    scored = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            body = f.read().strip()
        vec = embed(body)
        if body and vec is not None:
            scored.append((cosine(query, vec), body))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [body for _, body in scored[:top_k]]


REPLY_PROMPT = """あなたはユーザー本人の「第二の自分」です。
ユーザーに届いたメッセージへの返信を、ユーザー本人として書いてください。

【重要ルール】
1. 「自分が過去に書いた文章」があれば、その口調・敬語の度合いを真似ること。
2. 「ユーザーからの指示」に書かれた返事（参加する・断るなど）は本人の決定なので、はっきり伝えること。
3. 指示に無く本人に確認しないと分からないこと（具体的な日時や金額など）だけは、断定せず「〇〇（要確認）」と書くこと。
4. 宛名（〇〇様）と署名は後から付けるので書かないこと。本文だけを出力し、解説も書かないこと。"""


def signature():
    """profile.md の「署名:」（無ければ「名前:」）を返す"""
    match = re.search(r"(?:返信の)?署名[:：]\s*(.+)", load_profile()) or \
        re.search(r"名前[:：]\s*(.+)", load_profile())
    return match.group(1).strip() if match else ""


HONORIFIC = re.compile(r"^\S{1,15}(様|さん|先生|殿|各位|くん|ちゃん)[、,]?$")


def strip_salutation(body):
    """AIが書いてしまった宛名・署名の行を取り除く"""
    lines = body.strip().splitlines()
    names = {signature(), *(re.findall(r"名前[:：]\s*(.+)", load_profile()))} - {""}
    while lines and (HONORIFIC.match(lines[0].strip()) or not lines[0].strip()):
        lines.pop(0)
    # 「山田」「山田です。」「山田より」のような署名の行
    while lines and (re.sub(r"(です|より)?[。.]?$", "", lines[-1].strip()) in names or not lines[-1].strip()):
        lines.pop()
    return "\n".join(lines)


def compose_letter(to, body):
    """本人が指定した宛先と、プロフィールの署名で本文を挟む（AIに宛先を推測させない）"""
    body = strip_salutation(body)
    lines = []
    if to:
        lines += [to if to.endswith(("様", "さん", "先生", "殿", "各位")) else f"{to}様", ""]
    lines.append(body.strip())
    if signature():
        lines += ["", signature()]
    return "\n".join(lines)


def generate_reply(user_request, message):
    """届いたメッセージへの返信の本文を作る"""
    parts = []
    profile = load_profile()
    if profile:
        parts.append(f"■ ユーザーの情報\n{profile}")
    examples = find_my_messages(message)
    if examples:
        parts.append("■ 自分が過去に書いた文章\n" + "\n---\n".join(examples))
    parts.append(f"■ 届いたメッセージ\n{message}")
    parts.append(f"■ ユーザーからの指示\n{user_request}")
    progress(f"{TEXT_MODEL} で返信を考え中...")
    return ask_ollama(REPLY_PROMPT, "\n\n".join(parts), model=TEXT_MODEL).strip()


# --- schedule: ダブルブッキング検出 ---
def load_events(days=SCHEDULE_DAYS):
    """カレンダー（.ics）から、今から days 日先までの予定を読み込む"""
    import icalendar
    import recurring_ical_events

    now = datetime.now().astimezone()
    until = now + timedelta(days=days)
    events = []
    for path in sorted(glob.glob(os.path.join(CALENDAR_DIR, "*.ics"))):
        with open(path, "rb") as f:
            cal = icalendar.Calendar.from_ical(f.read())
        for ev in recurring_ical_events.of(cal).between(now, until):
            start = ev["DTSTART"].dt
            if not isinstance(start, datetime):
                continue  # 終日の予定は重複判定の対象外
            if "DTEND" in ev:
                end = ev["DTEND"].dt
            elif "DURATION" in ev:
                end = start + ev["DURATION"].dt
            else:
                end = start + timedelta(hours=1)
            events.append({"title": str(ev.get("SUMMARY", "(無題)")),
                           "start": start.astimezone(), "end": end.astimezone(),
                           "calendar": os.path.basename(path)})
    events.sort(key=lambda e: e["start"])
    return events


def find_conflicts(events):
    """時間が重なっている予定の組を返す"""
    conflicts = []
    for i, a in enumerate(events):
        for b in events[i + 1:]:
            if b["start"] >= a["end"]:
                break
            conflicts.append((a, b))
    return conflicts


def format_event(e):
    return f"{e['start']:%m/%d(%a) %H:%M}〜{e['end']:%H:%M} {e['title']}（{e['calendar']}）"


def free_slots(events, days=7, min_minutes=60, limit=15):
    """活動時間内で予定が入っていない時間帯を返す（LLMに空き時間を推測させないため）"""
    now = datetime.now().astimezone()
    slots = []
    for offset in range(days + 1):
        day = (now + timedelta(days=offset)).replace(minute=0, second=0, microsecond=0)
        cursor = max(day.replace(hour=WORK_HOURS[0]), now)
        day_end = day.replace(hour=WORK_HOURS[1])
        for e in events:
            if e["end"] <= cursor or e["start"] >= day_end:
                continue
            if (e["start"] - cursor) >= timedelta(minutes=min_minutes):
                slots.append((cursor, e["start"]))
            cursor = max(cursor, e["end"])
        if (day_end - cursor) >= timedelta(minutes=min_minutes):
            slots.append((cursor, day_end))
    return slots[:limit]


SCHEDULE_PROMPT = """あなたはユーザー本人の「第二の自分」であり、有能な秘書です。
重なっている予定A・Bのどちらを優先するか決め、もう一方の調整案をJSONだけで出力してください。

出力形式:
{"keep": "A または B",
 "reason": "優先する理由（ユーザーの情報の優先順位のどの項目に当たるか）",
 "action": "リスケ / 代理出席 / 途中参加 のどれか",
 "slots": [移動先として選んだ「空いている時間」の番号を2つ]}

優先順位は「ユーザーの情報」に書かれた順位に従うこと。各予定を最も近い項目に当てはめて比べること。"""

RESCHEDULE_PROMPT = """あなたはユーザー本人の「第二の自分」です。
ユーザー本人として、予定の相手に日程変更をお願いする丁寧で短いメッセージの本文を書いてください。

【重要ルール】
1. 変更をお願いする予定と、移動先の候補日時を必ず本文に書くこと。
2. 別の予定と重なったことを理由に、丁寧にお詫びすること。
3. 宛名（〇〇様）と署名は後から付けるので書かないこと。本文だけを出力すること。"""


def generate_schedule_decision(events, a, b, slots):
    """重なった予定A・Bの、どちらを優先してどう調整するかを決める"""
    parts = []
    profile = load_profile()
    if profile:
        parts.append(f"■ ユーザーの情報\n{profile}")
    parts.append(f"■ 重なっている予定\nA: {format_event(a)}\nB: {format_event(b)}")
    parts.append(f"■ 今後{SCHEDULE_DAYS}日間の予定一覧\n" + "\n".join(
        f"- {format_event(e)}" for e in events))
    parts.append("■ 空いている時間\n" + "\n".join(
        f"{i}: {s:%m/%d(%a) %H:%M}〜{e:%H:%M}" for i, (s, e) in enumerate(slots)))
    progress(f"{TEXT_MODEL} で「{a['title']}」と「{b['title']}」の調整案を考え中...")
    try:
        decision = json.loads(ask_ollama(SCHEDULE_PROMPT, "\n\n".join(parts),
                                         json_mode=True, model=TEXT_MODEL))
    except json.JSONDecodeError:
        decision = {}
    # 移動先は計算済みの空き時間からだけ選ぶ（AIが予定のある時間を提案しないように）
    picked = [slots[i] for i in decision.get("slots", []) if isinstance(i, int) and 0 <= i < len(slots)]
    decision["slots"] = picked[:2] or slots[:2]
    decision["keep"] = "B" if decision.get("keep") == "B" else "A"
    return decision


def generate_reschedule_body(move, slots):
    """優先しない方の予定の相手へ送る、日程変更のお願いの本文を作る"""
    prompt = (f"変更をお願いする予定: {format_event(move)}\n"
              "移動先の候補: " + "、".join(f"{s:%m/%d(%a) %H:%M}〜{e:%H:%M}" for s, e in slots))
    return ask_ollama(RESCHEDULE_PROMPT, prompt, model=TEXT_MODEL).strip()


# --- report: お任せした作業の結果報告 ---
NEVER_RUN = 267011  # タスクスケジューラの「まだ一度も実行されていない」


def task_status(entry, has_log):
    """お任せ作業が実際に実行されたかを、ログとタスクスケジューラの記録から判定する"""
    if not entry["task_name"].startswith(TASK_PREFIX):
        return "実行済み" if has_log else "実行中、またはまだ実行されていません"
    result = run_ps(f"$i = Get-ScheduledTaskInfo -TaskName {ps_quote(entry['task_name'])} -ErrorAction Stop; "
                    "'{0:yyyy-MM-dd HH:mm}|{1}|{2:yyyy-MM-dd HH:mm}' -f $i.LastRunTime, $i.LastTaskResult, $i.NextRunTime")
    if result.returncode != 0:
        return "タスクが見つかりません（削除された可能性があります）"
    last_run, code, next_run = result.stdout.strip().split("|")
    next_info = f"、次回は {next_run}" if next_run else ""
    if int(code) == NEVER_RUN:
        return f"まだ実行されていません{next_info}"
    if int(code) == 0:
        return f"成功（最終実行 {last_run}{next_info}）"
    return f"エラー（終了コード {code}、最終実行 {last_run}{next_info}）"


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


def save_text(prefix, ext, text):
    """BOM付きUTF-8で保存する（Windows PowerShell・メモ帳での文字化け防止）"""
    path = os.path.abspath(os.path.join(OUTPUT_DIR, f"{prefix}_{int(time.time())}.{ext}"))
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(text)
    if VERBOSE:
        print(f"\n✨ 成功しました！以下に保存しました:\n{path}")
        print("-" * 50)
        print(text)
        print("-" * 50)
    return path


def save_code(lang, code):
    return save_text("tool", "html" if lang == "html" else "ps1", code)


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


def ps_quote(text):
    """PowerShell の単一引用符文字列にする"""
    return "'" + text.replace("'", "''") + "'"


def run_ps(script, timeout=None):
    """PowerShell スクリプトをこのコンソールで実行し、結果を返す"""
    encoded = base64.b64encode(script.encode("utf-16-le")).decode()
    return subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW)


def confirm(question, code=None):
    """実行前に確認する（CONFIRM_BEFORE_RUN が False なら常に実行）"""
    if not CONFIRM_BEFORE_RUN:
        return True
    if code and DANGEROUS.search(code):
        print("⚠ 削除・システム変更などの危険なコマンドが含まれています。内容をよく確認してください。")
    return input(f"[AI] {question} (y/n): ").strip().lower() == "y"


def run_with_powershell(lang, path, code):
    """新しいPowerShellコンソールを起動して実行する（HTMLはブラウザで開く）"""
    if not confirm("実行しますか？", code if lang == "powershell" else None):
        print("[AI] 実行をスキップしました（ファイルは保存済みです）。")
        return
    progress("PowerShellを起動して実行します...")
    if lang == "html":
        # 指定のブラウザで開く（見つからなければ既定のブラウザ）
        script = (f"try {{ Start-Process -FilePath {ps_quote(BROWSER)} -ArgumentList {ps_quote(f'"{path}"')} -ErrorAction Stop }} "
                  f"catch {{ Start-Process -FilePath {ps_quote(path)} }}")
        command = ["powershell.exe", "-NoProfile", "-Command", script]
    else:
        command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                   "-NoExit", "-File", path]
    subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)


def read_clipboard():
    """クリップボードの文字列を PowerShell で取得する"""
    result = run_ps("[Console]::OutputEncoding = [Text.Encoding]::UTF8; Get-Clipboard -Raw")
    return result.stdout.strip()


def open_draft(path, copy=False):
    """下書きをメモ帳で開く（copy=True ならクリップボードにもコピー）"""
    script = f"Start-Process notepad.exe -ArgumentList {ps_quote(path)}"
    if copy:
        script = f"Get-Content -Raw -Encoding UTF8 {ps_quote(path)} | Set-Clipboard; " + script
    run_ps(script)


APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"


def toast_script(title, body):
    """Windows の通知を出す PowerShell スクリプト"""
    return ("[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null; "
            "$x = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
            "[Windows.UI.Notifications.ToastTemplateType]::ToastText02); "
            "$t = $x.GetElementsByTagName('text'); "
            f"$t.Item(0).AppendChild($x.CreateTextNode({ps_quote(title)})) | Out-Null; "
            f"$t.Item(1).AppendChild($x.CreateTextNode({ps_quote(body)})) | Out-Null; "
            "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
            f"{ps_quote(APP_ID)}).Show([Windows.UI.Notifications.ToastNotification]::new($x))")


def notify(title, body):
    """Windows の通知（画面右下）でお知らせする"""
    run_ps(toast_script(title, body))


def speak_script(text):
    """日本語の音声で読み上げる PowerShell スクリプト（Windows標準の音声合成）"""
    return ("Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "try { $s.SelectVoiceByHints('NotSet', 'NotSet', 0, [Globalization.CultureInfo]'ja-JP') } catch {}; "
            f"$s.Speak({ps_quote(text)})")


_speaker = None


def speak(text):
    """AIの返事を声で読み上げる（前の読み上げが続いていれば止めてから）"""
    global _speaker
    if _speaker and _speaker.poll() is None:
        _speaker.terminate()
    encoded = base64.b64encode(speak_script(text).encode("utf-16-le")).decode()
    _speaker = subprocess.Popen(["powershell.exe", "-NoProfile", "-EncodedCommand", encoded],
                                creationflags=subprocess.CREATE_NO_WINDOW)


WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
DELEGATED_FILE = os.path.join(OUTPUT_DIR, "delegated.jsonl")


def schedule_trigger(info):
    """お任せの実行タイミングを (タスクスケジューラのトリガー, 説明) にする。今すぐなら (None, ...)"""
    freq, at = info.get("frequency"), info.get("time", "")
    if freq in ("daily", "weekly"):
        if not re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", at):
            return "invalid", at
        if freq == "weekly" and info.get("day") in WEEKDAYS:
            return f"New-ScheduledTaskTrigger -Weekly -DaysOfWeek {info['day']} -At {at}", f"毎週{info['day']} {at}"
        return f"New-ScheduledTaskTrigger -Daily -At {at}", f"毎日 {at}"
    if freq == "once" and at:  # 時刻の指定が無い「〜しといて」は今すぐ実行する
        try:
            run_at = datetime.strptime(at, "%Y-%m-%d %H:%M")
        except ValueError:
            if not re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", at):
                return "invalid", at
            # 時刻だけなら、次に来るその時刻（今日がもう過ぎていれば明日）
            run_at = datetime.combine(datetime.now().date(), datetime.strptime(at, "%H:%M").time())
            if run_at <= datetime.now():
                run_at += timedelta(days=1)
        if run_at > datetime.now():
            return f"New-ScheduledTaskTrigger -Once -At {ps_quote(f'{run_at:%Y-%m-%d %H:%M}')}", f"{run_at:%m/%d %H:%M} に1回"
    return None, "今すぐ（バックグラウンド）"


def delegate(path, info):
    """生成したスクリプトを、本人がいなくても動くように実行・登録する"""
    trigger, when = schedule_trigger(info)
    if trigger == "invalid":
        return f"実行する日時（{when or '未指定'}）が分かりませんでした。"
    log = os.path.splitext(path)[0] + ".log"
    # 実行結果はログファイルに追記し、休んでいる間に何が起きたか後から報告できるようにする
    command = f"& {ps_quote(path)} *>> {ps_quote(log)}"

    if trigger is None:
        subprocess.Popen(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                          "-WindowStyle", "Hidden", "-Command", command],
                         creationflags=subprocess.CREATE_NO_WINDOW)
        name = "(バックグラウンド実行)"
    else:
        name = TASK_PREFIX + os.path.splitext(os.path.basename(path))[0]
        argument = f'-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "{command}"'
        script = (f"$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument {ps_quote(argument)}; "
                  f"$trigger = {trigger}; "
                  f"Register-ScheduledTask -TaskName {ps_quote(name)} -Action $action -Trigger $trigger "
                  f"-Description {ps_quote(info.get('task', ''))} | Out-Null")
        result = run_ps(script)
        if result.returncode != 0:
            return f"登録に失敗しました: {result.stderr.strip()}"

    with open(DELEGATED_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"task": info.get("task", ""), "when": when, "task_name": name,
                            "path": path, "log": log,
                            "registered_at": f"{datetime.now():%Y-%m-%d %H:%M}"},
                           ensure_ascii=False) + "\n")
    if VERBOSE:
        print(f"[AI] タスク名: {name}  ログ: {log}")
    return f"{when} に実行するよう登録しました（タスク名: {name}）"


# ============================================================
# ダブルブッキングの調整（エージェントの道具 plan_double_booking から使う）
# ============================================================
def plan_double_booking():
    """重なった予定ごとに優先を決め、連絡文の下書きを作ってメモ帳で開く。結果の要約を返す"""
    if not glob.glob(os.path.join(CALENDAR_DIR, "*.ics")):
        return "カレンダーが読み込まれていません。「予定を読み込む」から .ics ファイルを読み込んでください。"
    events = load_events()
    conflicts = find_conflicts(events)
    if not conflicts:
        return f"今後{SCHEDULE_DAYS}日間にダブルブッキングはありません（予定 {len(events)} 件）。"

    slots = free_slots(events)
    sections, summary = [], []
    for a, b in conflicts:
        d = generate_schedule_decision(events, a, b, slots)
        keep, move = (a, b) if d["keep"] == "A" else (b, a)
        summary.append(f"- 「{keep['title']}」を優先し、「{move['title']}」を{d.get('action', 'リスケ')}する案")
        to = ask_user(f"「{move['title']}」の連絡相手の名前は？")
        letter = compose_letter(to, generate_reschedule_body(move, d["slots"]))
        sections.append("\n".join([
            f"## {format_event(a)} ／ {format_event(b)}",
            f"- **優先する予定**: {keep['title']}（{d.get('reason', '')}）",
            f"- **もう一方の対応**: {move['title']} を{d.get('action', 'リスケ')}",
            "- **移動先の候補**: " + "、".join(f"{s:%m/%d(%a) %H:%M}〜{e:%H:%M}" for s, e in d["slots"]),
            "- **連絡文の下書き**:", "", "```", letter, "```"]))
    path = save_text("schedule", "md", "\n\n".join(sections))
    open_draft(path)
    return "調整案と連絡文の下書きを作り、メモ帳で開きました。\n" + "\n".join(summary)


def load_delegated():
    """お任せした作業の一覧を返す"""
    if not os.path.exists(DELEGATED_FILE):
        return []
    with open(DELEGATED_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cancel_delegated(entry):
    """お任せを取り消す（予約はタスクスケジューラから削除し、一覧からも外す）"""
    if entry["task_name"].startswith(TASK_PREFIX):
        run_ps(f"Unregister-ScheduledTask -TaskName {ps_quote(entry['task_name'])} -Confirm:$false")
    rest = [e for e in load_delegated() if e != entry]
    with open(DELEGATED_FILE, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in rest)


# ============================================================
def main():
    """ターミナル版（--text でキーボード入力）。処理はエージェント（agent.py）が行う"""
    from agent import Agent

    global WHISPER
    text_mode = "--text" in sys.argv
    if not text_mode:
        print("[AI] Whisper モデルを読み込み中...")
        WHISPER = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    agent = Agent()

    while True:
        if text_mode:
            user_request = input("\nあなた: ").strip()
        else:
            audio = record_voice()
            if audio is None:
                continue
            user_request = transcribe(WHISPER, audio)
        if user_request in ("終了", "exit", "quit"):
            break
        if not user_request:
            continue
        try:
            print(f"\n第二の自分: {agent.run(user_request)}")
        except requests.RequestException as e:
            print(f"Ollama に接続できません（`ollama serve` が起動しているか確認してください）: {e}")


if __name__ == "__main__":
    main()
