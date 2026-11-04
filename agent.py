"""第二の自分 — AIエージェント（構成の「2. Ollama（ローカルLLM）」の部分）

頼まれたことを達成するために、エージェントが自分で手順を考え、道具（TOOLS）を順に使う。
道具の中身（コード生成・保存・PowerShell実行）は main.py の 2〜4 段階をそのまま使う。

  - 自分で考えて複数の作業をこなす: Ollama のツール呼び出しで、結果を見ながら次の道具を選ぶ
  - 会話を覚えている: やりとりを conversation.json に保存し、次に起動しても続きから話せる
  - 自分から気づいて知らせる: check_alerts() がダブルブッキングやお任せ作業の結果を見回る
  - 声で返事をする: main.speak()（Windows標準の音声合成）
"""

import difflib
import inspect
import json
import os
from datetime import datetime

import requests

import main

AGENT_MODEL = "qwen2.5:14b"          # 手順を考えて道具を使う頭脳（未取得なら main.TEXT_MODEL を使う）
MAX_STEPS = 8                        # 1つの頼みで道具を使う最大回数
MEMORY_FILE = os.path.join(main.OUTPUT_DIR, "conversation.json")
MEMORY_TURNS = 20                    # 覚えておく発言の数
NOTIFIED_FILE = os.path.join(main.OUTPUT_DIR, "notified.json")
ALERT_DAYS = 7                       # 何日先までのダブルブッキングを見回るか
NUM_CTX = 8192                       # 一度に読める文章の量（道具の説明・会話・結果が収まるように）


# ============================================================
# 道具（エージェントが使える機能）
# ============================================================
def get_schedule(days=7):
    """予定・ダブルブッキング・空き時間を調べる"""
    if not any(name.endswith(".ics") for name in os.listdir(main.CALENDAR_DIR)):
        return "カレンダーが読み込まれていません。「予定を読み込む」から .ics を読み込むよう案内してください。"
    days = max(int(days), 7)  # 「明日」と頼まれても、近くのダブルブッキングを見落とさないよう最低7日分見る
    events = main.load_events(days)
    conflicts = main.find_conflicts(events)
    lines = [f"■ 今後{days}日間の予定（{len(events)}件）"] + [f"- {main.format_event(e)}" for e in events]
    lines.append("■ ダブルブッキング: " + ("なし" if not conflicts else f"{len(conflicts)}件"))
    lines += [f"- {main.format_event(a)} ／ {main.format_event(b)}" for a, b in conflicts]
    if conflicts:
        lines.append("（調整案や相手への連絡文が必要なら、次に plan_double_booking を使うこと）")
    lines.append("■ 空いている時間")
    lines += [f"- {s:%m/%d(%a) %H:%M}〜{e:%H:%M}" for s, e in main.free_slots(events, days=min(int(days), 7))]
    return "\n".join(lines)


def generate_script(description):
    """要望からコードを作って保存し、(言語, コード, 保存先) を返す"""
    lang, code = main.extract_code(main.generate_code(description, main.search_history(description)))
    path = main.save_code(lang, code)
    main.save_history(description, lang, code)
    return lang, code, path


def run_task_now(description):
    """PowerShellスクリプトやWebサイトを作って、今すぐ実行する"""
    lang, code, path = generate_script(description)
    if not main.confirm("この作業を今すぐ実行しますか？", code if lang == "powershell" else None):
        return "ユーザーが実行を取りやめました。"
    if lang == "html":
        main.run_ps(f"try {{ Start-Process -FilePath {main.ps_quote(main.BROWSER)} "
                    f"-ArgumentList {main.ps_quote(chr(34) + path + chr(34))} -ErrorAction Stop }} "
                    f"catch {{ Start-Process -FilePath {main.ps_quote(path)} }}")
        return "Webサイトを作って、ブラウザで開きました。"
    try:
        result = main.run_ps("[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
                             f"& {main.ps_quote(path)} *>&1 | Out-String -Width 160", timeout=300)
    except Exception:
        return "5分たっても終わらなかったので止めました。"
    output = (result.stdout + result.stderr).strip()
    if output:
        print(output[-1500:] + "\n")
    return f"実行しました。出力:\n{output[-3000:] or '（出力なし）'}"


def delegate_task(description, frequency="now", time="", day=""):
    """作業を本人がいなくても動くようにお任せする（今すぐ裏で / 指定時刻に1回 / 毎日 / 毎週）"""
    lang, code, path = generate_script(description)
    if lang != "powershell":
        return "お任せできるのは PowerShell で自動化できる作業だけです。"
    if not main.confirm("この作業をお任せしますか？", code):
        return "ユーザーがお任せを取りやめました。"
    return main.delegate(path, {"task": description, "frequency": frequency, "time": time, "day": day})


def set_reminder(message, time, frequency="once", day=""):
    """指定した時刻に、Windowsの通知と音声でお知らせする"""
    script = main.toast_script("第二の自分", message) + "; " + main.speak_script(message)
    path = main.save_text("reminder", "ps1", script)
    return main.delegate(path, {"task": f"お知らせ: {message}", "frequency": frequency, "time": time, "day": day})


def list_delegated_tasks():
    """お任せした作業の一覧と、実際に実行されたか・結果を調べる"""
    entries = main.load_delegated()[-10:]
    if not entries:
        return "お任せされている作業はありません。"
    lines = []
    for e in entries:
        log = ""
        if os.path.exists(e["log"]):
            with open(e["log"], encoding="utf-8", errors="replace") as f:
                log = f.read()[-300:].strip()
        lines.append(f"- {e['task']}（{e['when']}、タスク名: {e['task_name']}）\n"
                     f"  状態: {main.task_status(e, bool(log))}\n  ログの最後: {log or 'なし'}")
    return "\n".join(lines)


def find_delegated(task):
    """作業の内容・タスク名から、お任せを探す（新しいもの優先。空なら最後のお任せ）"""
    entries = main.load_delegated()
    if not entries:
        return None
    if not task:
        return entries[-1]
    for e in reversed(entries):
        if task == e["task_name"] or task in e["task"]:
            return e
    # 言い方が違っても見つけられるよう、いちばん似ているものを選ぶ（同じくらいなら新しいもの）
    def similarity(e):
        return difflib.SequenceMatcher(None, task, e["task"].removeprefix("お知らせ: ")).ratio()
    best = max(reversed(entries), key=similarity)
    return best if similarity(best) >= 0.3 else None


def cancel_delegated_task(task):
    """お任せを取り消す（作業の内容かタスク名で指定）"""
    e = find_delegated(task)
    if e is None:
        return f"「{task}」に当たるお任せが見つかりませんでした。"
    if not main.confirm(f"「{e['task']}」のお任せを取り消しますか？"):
        return "ユーザーが取り消しをやめました。"
    main.cancel_delegated(e)
    return f"「{e['task']}」のお任せを取り消しました。"


def change_delegated_time(task, time, frequency="once", day=""):
    """お任せ・お知らせの実行日時を変える（前の予約を取り消して、同じ内容で登録し直す）"""
    e = find_delegated(task)
    if e is None:
        return f"「{task}」に当たるお任せが見つかりませんでした。"
    info = {"task": e["task"], "frequency": frequency, "time": time, "day": day}
    trigger, when = main.schedule_trigger(info)
    if trigger is None or trigger == "invalid":  # 新しい日時が正しいと分かるまで、前の予約は消さない
        return f"新しい日時（{time}）が分かりませんでした。前の予約（{e['when']}）はそのまま残しています。"
    main.cancel_delegated(e)
    result = main.delegate(e["path"], info)
    return f"「{e['task']}」の前の予約（{e['when']}）を取り消し、{result}"


def write_message(purpose, reply_to_received=False):
    """メッセージの下書きを作る（届いたメッセージへの返信、または新しく送るもの）。送信はしない"""
    message = ""
    if reply_to_received:
        message = main.read_clipboard()
        if not message:
            return "返信するメッセージが入力されませんでした。"
    to = main.ask_user("連絡相手（宛先）の名前は？")
    body = main.generate_reply(purpose, message or "（届いたメッセージは無く、こちらから新しく送る）")
    draft = main.compose_letter(to, body)
    path = main.save_text("message", "txt", draft)
    main.open_draft(path, copy=True)
    print(draft + "\n")
    return f"下書きを作り、メモ帳で開いてクリップボードにコピーしました（送信はしていません）。\n{draft}"


def plan_double_booking():
    """ダブルブッキングごとに優先を決め、相手への日程変更の連絡文を作る"""
    return main.plan_double_booking()


TOOLS = {f.__name__: f for f in (get_schedule, run_task_now, delegate_task, set_reminder,
                                 list_delegated_tasks, cancel_delegated_task, change_delegated_time, write_message,
                                 plan_double_booking)}

TOOL_LABELS = {
    "get_schedule": "予定を確認しています",
    "run_task_now": "作業を実行しています",
    "delegate_task": "お任せの準備をしています",
    "set_reminder": "お知らせを予約しています",
    "list_delegated_tasks": "お任せした作業の状況を確認しています",
    "cancel_delegated_task": "お任せを取り消しています",
    "change_delegated_time": "お任せの日時を変更しています",
    "write_message": "メッセージの下書きを作っています",
    "plan_double_booking": "ダブルブッキングの調整案を作っています",
}


def param(kind, description, enum=None):
    p = {"type": kind, "description": description}
    if enum:
        p["enum"] = enum
    return p


def tool_spec(name, description, properties=None, required=()):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {}, "required": list(required)}}}


TIME_HELP = "once は YYYY-MM-DD HH:MM、daily/weekly は HH:MM（24時間表記）。now なら空"
TOOL_SPECS = [
    tool_spec("get_schedule", "予定の一覧・ダブルブッキング・空いている時間を調べる。予定について答える前に必ず使う。",
              {"days": param("integer", "何日先まで調べるか（標準は7）")}),
    tool_spec("run_task_now", "PC操作の自動化（PowerShell）やWebサイトを作り、今すぐ目の前で実行して結果を受け取る。",
              {"description": param("string", "やってほしい作業の内容（具体的に）")}, ["description"]),
    tool_spec("delegate_task", "作業をお任せで登録し、本人がいなくても実行する。「〜しといて」「休んでいる間に」「〜時に」「毎朝」など。",
              {"description": param("string", "自動で実行する作業の内容（日時の表現は除く）"),
               "frequency": param("string", "now=今すぐ裏で / once=指定時刻に1回 / daily=毎日 / weekly=毎週",
                                  ["now", "once", "daily", "weekly"]),
               "time": param("string", TIME_HELP),
               "day": param("string", "weekly のときの曜日（Monday〜Sunday）")}, ["description", "frequency"]),
    tool_spec("set_reminder", "指定した時刻に、Windowsの通知と音声でお知らせする（作業はせず知らせるだけ）。",
              {"message": param("string", "お知らせする内容"),
               "time": param("string", TIME_HELP),
               "frequency": param("string", "once=1回 / daily=毎日 / weekly=毎週", ["once", "daily", "weekly"]),
               "day": param("string", "weekly のときの曜日（Monday〜Sunday）")}, ["message", "time"]),
    tool_spec("list_delegated_tasks", "お任せした作業の一覧と、実際に実行されたか・結果・エラーを調べる。"),
    tool_spec("cancel_delegated_task", "お任せを取り消す。",
              {"task": param("string", "取り消す作業の内容かタスク名")}, ["task"]),
    tool_spec("change_delegated_time", "お任せ・お知らせの実行日時を変える（「やっぱり9時にして」など）。",
              {"task": param("string", "変える作業・お知らせの内容（会話から分かる言葉で）"),
               "time": param("string", TIME_HELP),
               "frequency": param("string", "once=1回 / daily=毎日 / weekly=毎週", ["once", "daily", "weekly"]),
               "day": param("string", "weekly のときの曜日（Monday〜Sunday）")}, ["task", "time"]),
    tool_spec("write_message", "メール・チャットの下書きを作る（送信はしない）。宛先は道具が本人に聞く。",
              {"purpose": param("string", "伝えたい内容・返事の方針（例: 参加できると丁寧に伝える）"),
               "reply_to_received": param("boolean", "届いたメッセージへの返信なら true（本人に貼り付けてもらう）")},
              ["purpose"]),
    tool_spec("plan_double_booking", "ダブルブッキングごとに、本人の優先順位で残す予定を決め、もう一方の相手への日程変更の連絡文を作る。"),
]

SYSTEM_PROMPT = """あなたは「第二の自分」。ユーザー本人のデジタルツインとして、本人の代わりに業務をこなすAIエージェントです。
現在日時: {now}

■ ユーザーの情報
{profile}

【行動ルール】
1. 頼まれたことを達成するために必要な道具を、自分で考えて順番に使うこと。1つの頼みに複数の道具を使ってよい。
2. 予定やお任せ作業の状況などの事実は、推測せず必ず道具で確認すること。
3. 「〜しといて」「休んでいる間に」「毎朝」など本人がいなくても進める作業は delegate_task、知らせるだけなら set_reminder、今すぐ目の前でやる作業は run_task_now を使う。
4. メッセージは下書きを作るだけで、送信はしない。宛先は道具が本人に聞くので推測しない。
5. 「1時間後」「明日の朝」などは現在日時から計算し、YYYY-MM-DD HH:MM（24時間表記）で道具に渡す。
6. お任せやお知らせの日時を変えるときは change_delegated_time を使う。
7. 頼みに複数の作業が含まれるときは、全部の作業を道具で終えてから返事をする。
8. 道具を使っていない作業を「やりました」と言ってはいけない。
9. ユーザーが取りやめた作業は、無理に続けない。
10. 道具を使い終えたら、やったことと結果を2〜3文の話し言葉で短く伝える。読み上げに使うので、記号や箇条書きは使わない。
11. 道具が要らない雑談や質問には、そのまま短く答える。"""

PLAN_PROMPT = """あなたはAIエージェントの計画係です。ユーザーの今回の頼みを達成するための作業手順を考え、JSONだけを出力してください。

使える道具:
{tools}

出力形式: {{"steps": ["手順の説明（使う道具名）", ...]}}

【ルール】
1. 頼みに含まれる作業は、1つも漏らさず手順にする。「〜して、〜して」は別々の手順にする。
2. 「重なっていたら〜」のような条件付きの作業も手順に入れる（実行するかは結果を見て決める）。
3. お任せ・お知らせの日時を変えるときは change_delegated_time の1手順にする。
4. 道具が要らない雑談・質問なら {{"steps": []}} にする。"""

STEP_PROMPT = """■ ユーザーの今回の頼み
{text}

■ ここまでの手順の結果
{results}

■ 今の手順
{step}

この手順を道具で実行してください。ここまでの結果から、条件に当てはまらずやる必要がない場合だけ、道具を使わずに理由を一言で答えてください。"""

SUMMARY_PROMPT = """あなたは「第二の自分」。ユーザー本人のデジタルツインです。
ユーザーの頼みと、実際に行った手順の結果をもとに、やったことと結果を2〜3文の話し言葉で伝えてください。

【重要ルール】
1. 「手順の結果」に書かれていないことは、やったと言わない。失敗・取りやめ・見送った手順は正直に伝える。
2. 読み上げに使うので、記号や箇条書きは使わない。返事だけを出力する。"""

ONCE_TOOLS = ("get_schedule", "list_delegated_tasks")  # 1つの頼みの中で結果が変わらない調べもの

# ============================================================
# エージェント本体
# ============================================================
class Agent:
    def __init__(self, on_tool=None, on_plan=None):
        self.on_tool = on_tool or (lambda label: print(f"🔧 {label}..."))
        self.on_plan = on_plan or (lambda steps: print("📝 やること:\n" + "\n".join(
            f"  {i}. {s}" for i, s in enumerate(steps, 1))))
        self.history = self.load_memory()
        self.model = AGENT_MODEL if self.has_model(AGENT_MODEL) else main.TEXT_MODEL

    @staticmethod
    def has_model(name):
        try:
            tags = requests.get(f"{main.OLLAMA_BASE}/api/tags", timeout=5).json().get("models", [])
        except requests.RequestException:
            return False
        return any(m.get("name") in (name, f"{name}:latest") for m in tags)

    # --- 会話の記憶 ---
    def load_memory(self):
        if not os.path.exists(MEMORY_FILE):
            return []
        with open(MEMORY_FILE, encoding="utf-8") as f:
            return json.load(f)

    def remember(self, role, content):
        self.history = (self.history + [{"role": role, "content": content}])[-MEMORY_TURNS:]
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=1)

    def forget(self):
        self.history = []
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)

    # --- 考えて道具を使う ---
    def system_prompt(self):
        return SYSTEM_PROMPT.format(now=f"{datetime.now():%Y-%m-%d %H:%M (%A)}",
                                    profile=main.load_profile() or "（未登録）")

    def chat(self, messages, tools=TOOL_SPECS):
        r = requests.post(f"{main.OLLAMA_BASE}/api/chat", timeout=600, json={
            "model": self.model, "messages": messages, "tools": tools, "stream": False,
            "options": {"temperature": 0.3, "num_ctx": NUM_CTX}})
        r.raise_for_status()
        return r.json()["message"]

    def call_tool(self, name, args):
        fn = TOOLS.get(name)
        if fn is None:
            return f"「{name}」という道具はありません。"
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        params = inspect.signature(fn).parameters
        try:
            return str(fn(**{k: v for k, v in (args or {}).items() if k in params}))
        except requests.RequestException:
            raise
        except Exception as e:
            return f"エラーが起きました: {e}"

    def plan(self, text):
        """作業を始める前に、やることリストを作る（小さなモデルでも手順を漏らさないように）"""
        tools = "\n".join(f"- {s['function']['name']}: {s['function']['description']}" for s in TOOL_SPECS)
        recent = "\n".join(f"{'ユーザー' if m['role'] == 'user' else 'あなた'}: {m['content']}"
                           for m in self.history[-7:-1])
        prompt = (f"現在日時: {datetime.now():%Y-%m-%d %H:%M (%A)}\n"
                  f"■ これまでの会話\n{recent or '（なし）'}\n\n■ 今回の頼み\n{text}")
        try:
            steps = json.loads(main.ask_ollama(PLAN_PROMPT.format(tools=tools), prompt,
                                               json_mode=True, model=self.model)).get("steps", [])
        except (json.JSONDecodeError, AttributeError):
            steps = []
        return [str(s) for s in steps if str(s).strip()][:MAX_STEPS]

    def run(self, text):
        """頼みごとを受け取り、やることリストを1手順ずつ実行して、結果を報告する"""
        self.remember("user", text)
        steps = self.plan(text)
        if not steps:
            return self.answer_directly(text)
        self.on_plan(steps)

        results, cache = [], {}
        for i, step in enumerate(steps, 1):
            # 手順に書かれた道具だけを渡し、その手順に集中させる（小さなモデルが手順を飛ばさないように）
            names = [n for n in TOOLS if n in step]
            specs = [s for s in TOOL_SPECS if s["function"]["name"] in names] or TOOL_SPECS
            if names and names[0] in ONCE_TOOLS and names[0] in cache:
                results.append(f"{i}. {step}\n→ {cache[names[0]]}")
                continue
            done = "\n\n".join(results) or "（まだありません）"
            messages = ([{"role": "system", "content": self.system_prompt()}] + self.history[:-1] +
                        [{"role": "user", "content": STEP_PROMPT.format(text=text, results=done, step=step)}])
            msg = self.chat(messages, specs)
            calls = msg.get("tool_calls") or []
            if not calls:
                reason = (msg.get("content") or "").strip()
                results.append(f"{i}. {step}\n→ 見送り: {reason}")
                continue
            outputs = []
            for call in calls[:1]:
                fn = call.get("function", {})
                name = fn.get("name")
                self.on_tool(TOOL_LABELS.get(name, name))
                output = self.call_tool(name, fn.get("arguments"))[:3000]
                cache[name] = output
                outputs.append(output)
            results.append(f"{i}. {step}\n→ " + "\n".join(outputs))

        report = main.ask_ollama(SUMMARY_PROMPT, f"■ ユーザーの頼み\n{text}\n\n■ 手順の結果\n" +
                                 "\n\n".join(results), model=self.model).strip()
        self.remember("assistant", report)
        return report

    def answer_directly(self, text):
        """道具が要らない雑談・質問（念のため道具も使えるようにしておく）"""
        messages = [{"role": "system", "content": self.system_prompt()}] + self.history
        msg = self.chat(messages)
        for call in (msg.get("tool_calls") or [])[:1]:
            fn = call.get("function", {})
            self.on_tool(TOOL_LABELS.get(fn.get("name"), fn.get("name")))
            output = self.call_tool(fn.get("name"), fn.get("arguments"))[:3000]
            msg = {"content": main.ask_ollama(SUMMARY_PROMPT, f"■ ユーザーの頼み\n{text}\n\n■ 手順の結果\n{output}",
                                              model=self.model)}
        answer = (msg.get("content") or "").strip() or "すみません、うまく答えられませんでした。"
        self.remember("assistant", answer)
        return answer

    # --- 自分から気づいて知らせる ---
    def check_alerts(self):
        """新しいダブルブッキングや、お任せ作業の完了・エラーを見つけて (題名, 内容) の一覧を返す"""
        notified = set()
        if os.path.exists(NOTIFIED_FILE):
            with open(NOTIFIED_FILE, encoding="utf-8") as f:
                notified = set(json.load(f))
        alerts = []

        def alert(key, title, body):
            if key not in notified:
                notified.add(key)
                alerts.append((title, body))

        if any(name.endswith(".ics") for name in os.listdir(main.CALENDAR_DIR)):
            for a, b in main.find_conflicts(main.load_events(ALERT_DAYS)):
                alert(f"conflict:{a['title']}:{a['start']}:{b['title']}:{b['start']}",
                      "ダブルブッキングを見つけました",
                      f"{a['start']:%m月%d日 %H時%M分}からの「{a['title']}」と「{b['title']}」が重なっています。"
                      "調整しましょうか？")
        for e in main.load_delegated():
            has_log = os.path.exists(e["log"])
            status = main.task_status(e, has_log)
            if status.startswith("エラー"):
                alert(f"error:{e['task_name']}:{status}", "お任せの作業でエラーが出ました",
                      f"「{e['task']}」でエラーが出ました。状況を確認しましょうか？")
            elif status.startswith(("成功", "実行済み")) and has_log:
                alert(f"done:{e['log']}:{os.path.getmtime(e['log'])}", "お任せの作業が終わりました",
                      f"「{e['task']}」が終わりました。")

        with open(NOTIFIED_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(notified), f, ensure_ascii=False)
        for _, body in alerts:  # お知らせも会話として覚え、「お願い」と返せば続きができるようにする
            self.remember("assistant", f"（お知らせ）{body}")
        return alerts
