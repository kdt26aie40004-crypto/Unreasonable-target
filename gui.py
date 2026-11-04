"""第二の自分 — 初心者向けの画面（「起動.bat」かデスクトップのショートカットで起動）

処理の中身は main.py の4段階構成（Whisper → Ollama → Python連携コア → PowerShell実行）をそのまま使い、
この画面は入力・確認・設定・結果表示だけを担当する。
"""

import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:  # 高解像度の画面でも文字がぼやけないようにする
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except (AttributeError, OSError):
    pass

try:
    import numpy as np
    import requests
    import sounddevice as sd
    import main
    from agent import Agent
except ImportError as e:
    tk.Tk().withdraw()
    messagebox.showerror("第二の自分", f"必要なパッケージが入っていません。\n\n"
                                      f"「pip install -r requirements.txt」を実行してください。\n\n({e})")
    sys.exit(1)

BG = "#f5f6f8"
BLUE = "#1a5fb4"
RED = "#e01b24"
GRAY = "#e0e0e0"
FONT = ("Yu Gothic UI", 12)
FONT_SMALL = ("Yu Gothic UI", 10)
FONT_BIG = ("Yu Gothic UI", 18, "bold")
FONT_CODE = ("Consolas", 10)

EXAMPLES = [
    "予定を確認して、重なってたら調整案を作って",
    "明日の朝8時に発表会の準備をするよう知らせて",
    "毎朝9時にディスクの空き容量を記録しといて",
    "任せてた作業どうだった？",
]

MONITOR_MINUTES = 15  # 自分から見回る間隔

OLLAMA_APP = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama app.exe")


def has_microphone():
    try:
        sd.query_devices(kind="input")
        return True
    except Exception:
        return False


def ollama_running():
    try:
        requests.get(f"{main.OLLAMA_BASE}/api/version", timeout=2)
        return True
    except requests.RequestException:
        return False


def button(parent, text, command, primary=False, **kw):
    return tk.Button(parent, text=text, command=command, font=FONT, relief="flat", cursor="hand2",
                     bg=BLUE if primary else GRAY, fg="white" if primary else "black",
                     activebackground="#15498a" if primary else "#cfcfcf", padx=12, pady=2, **kw)


class Recorder:
    """マイク録音（start でボタンを押してから stop まで）"""

    def __init__(self):
        self.frames, self.stream = [], None

    def start(self):
        self.frames = []
        self.stream = sd.InputStream(samplerate=main.SAMPLE_RATE, channels=1, dtype="float32",
                                     callback=lambda data, *_: self.frames.append(data.copy()))
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()
        self.stream = None
        return np.concatenate(self.frames).flatten() if self.frames else None


class QueueWriter:
    """main.py の print を画面のログに流す"""

    def __init__(self, q):
        self.q = q

    def write(self, text):
        self.q.put((text.replace("[AI] ", ""), None))

    def flush(self):
        pass


class App:
    def __init__(self, root):
        self.root = root
        self.log_queue = queue.Queue()
        self.recorder = Recorder()
        self.mic = has_microphone()
        self.busy, self.busy_text = False, ""
        self.agent = None
        self.voice = tk.BooleanVar(value=True)
        self.dots = 0

        root.title("第二の自分")
        self.scale = root.winfo_fpixels("1i") / 96  # 画面の拡大率に合わせてウィンドウの大きさを決める
        root.geometry(f"{int(860 * self.scale)}x{int(700 * self.scale)}")
        root.minsize(int(600 * self.scale), int(500 * self.scale))
        root.configure(bg=BG)

        # --- 上: タイトルと設定ボタン ---
        header = tk.Frame(root, bg=BG)
        header.pack(fill="x", padx=16, pady=(12, 0))
        tk.Label(header, text="第二の自分", font=FONT_BIG, bg=BG).pack(side="left")
        tools = tk.Frame(header, bg=BG)
        tools.pack(side="right")
        tk.Label(root, text="あなたの代わりに考えて動くAIエージェント。「〜しといて」と頼むと、休んでいる間にこなします",
                 font=FONT_SMALL, bg=BG, fg="#555", anchor="w").pack(fill="x", padx=16)
        tk.Checkbutton(tools, text="🔊 声で返事", variable=self.voice, font=FONT_SMALL, bg=BG,
                       activebackground=BG, cursor="hand2").pack(side="left", padx=3)
        for text, command in (("🧹 新しい会話", self.new_conversation), ("👤 自分の情報", self.open_profile),
                              ("📅 予定を読み込む", self.open_calendar), ("📋 お任せ一覧", self.open_tasks)):
            tk.Button(tools, text=text, command=command, font=FONT_SMALL, relief="groove", bg="white",
                      cursor="hand2", padx=8).pack(side="left", padx=3)

        # --- 下: 入力欄（先に配置して、ウィンドウが小さくても隠れないようにする） ---
        self.status = tk.Label(root, text="", font=FONT_SMALL, bg=BG, fg="#555", anchor="w")
        self.status.pack(side="bottom", fill="x", padx=16, pady=(0, 10))

        bar = tk.Frame(root, bg=BG)
        bar.pack(side="bottom", fill="x", padx=16, pady=(4, 6))
        self.entry = tk.Entry(bar, font=FONT, relief="solid", bd=1)
        self.entry.pack(side="left", fill="x", expand=True, ipady=6)
        self.entry.bind("<Return>", lambda _: self.send())
        self.send_btn = button(bar, "送信", self.send, primary=True)
        self.send_btn.pack(side="left", padx=(6, 0), fill="y")
        self.mic_btn = button(bar, "🎤 話す", self.toggle_voice, state="disabled")
        self.mic_btn.pack(side="left", padx=(6, 0), fill="y")

        chips = tk.Frame(root, bg=BG)
        chips.pack(side="bottom", fill="x", padx=16)
        tk.Label(chips, text="例:", font=FONT_SMALL, bg=BG).grid(row=0, column=0, rowspan=2, sticky="n")
        for i, text in enumerate(EXAMPLES):
            tk.Button(chips, text=text, font=FONT_SMALL, relief="groove", bg="white", cursor="hand2",
                      command=lambda t=text: self.fill(t)).grid(row=i // 2, column=1 + i % 2,
                                                                 sticky="w", padx=3, pady=2)

        # --- 中央: やりとりの表示 ---
        self.log = scrolledtext.ScrolledText(root, font=FONT, wrap="word", state="disabled",
                                             relief="flat", padx=14, pady=10, spacing1=2, spacing3=2)
        self.log.pack(fill="both", expand=True, padx=16, pady=10)
        self.log.tag_config("you", foreground=BLUE, font=(FONT[0], FONT[1], "bold"), spacing1=10)
        self.log.tag_config("note", foreground="#888")
        self.log.tag_config("warn", foreground=RED)
        self.log.tag_config("ai", foreground="#1c1c1c", font=(FONT[0], FONT[1]), lmargin1=0, spacing1=6)
        self.log.tag_config("plan", foreground="#5e3c99", font=FONT_SMALL)
        self.log.tag_config("tool", foreground="#26a269", font=FONT_SMALL)
        self.log.tag_config("alert", foreground="#c64600", font=(FONT[0], FONT[1], "bold"), spacing1=8)

        main.VERBOSE = False
        sys.stdout = sys.stderr = QueueWriter(self.log_queue)
        main.confirm = self.confirm
        main.ask_user = self.ask_user
        main.read_clipboard = self.ask_message
        self.poll_log()
        self.startup()

    # ============================================================
    # 起動時
    # ============================================================
    def startup(self):
        self.say("こんにちは。下の欄に頼みたいことを書いて「送信」を押してください。\n"
                 "「例」のボタンを押すと、入力欄に例文が入ります。\n", "note")
        self.set_busy(True, "準備中です")
        threading.Thread(target=self.prepare, daemon=True).start()
        if not os.path.exists(main.PROFILE_FILE):
            self.root.after(500, lambda: self.open_profile(first=True))

    def prepare(self):
        """Ollama が止まっていれば起動し、マイクがあれば音声認識を読み込む"""
        if not ollama_running() and os.path.exists(OLLAMA_APP):
            subprocess.Popen([OLLAMA_APP])
            for _ in range(30):
                if ollama_running():
                    break
                time.sleep(1)
        if not ollama_running():
            print("⚠ AI（Ollama）を起動できませんでした。スタートメニューから「Ollama」を起動してください。\n")
        self.agent = Agent(on_tool=lambda label: self.log_queue.put((f"   🔧 {label}...\n", "tool")),
                           on_plan=self.show_plan)
        if self.agent.history:
            self.log_queue.put(("前回の会話を覚えています。続きから話せます（忘れさせるときは「🧹 新しい会話」）。\n", "note"))
        self.root.after(60 * 1000, self.monitor)
        if self.mic:
            main.WHISPER = main.WhisperModel(main.WHISPER_MODEL, device="cpu", compute_type="int8")
        self.root.after(0, lambda: self.set_busy(False))

    # ============================================================
    # 表示
    # ============================================================
    def say(self, text, tag=None):
        self.log.config(state="normal")
        self.log.insert("end", text, tag)
        self.log.see("end")
        self.log.config(state="disabled")

    def poll_log(self):
        while not self.log_queue.empty():
            text, tag = self.log_queue.get()
            self.say(text, tag or ("warn" if text.startswith("⚠") else None))
        if self.busy:  # 考え中は「...」を動かして、止まっていないことを伝える
            self.dots = (self.dots + 1) % 4
            self.status.config(text=self.busy_text + "." * self.dots)
        self.root.after(300 if self.busy else 100, self.poll_log)

    def ready_text(self):
        brain = f"（頭脳: {self.agent.model}）" if self.agent else ""
        if main.WHISPER is not None:
            return f"準備できました。文字で入力するか「🎤 話す」で声で頼めます{brain}"
        return f"準備できました。マイクが見つからないため文字入力で使えます{brain}"

    def fill(self, text):
        self.entry.delete(0, "end")
        self.entry.insert(0, text)
        self.entry.focus_set()

    def set_busy(self, busy, text="考え中です。少しお待ちください"):
        self.busy, self.busy_text = busy, text
        state = "disabled" if busy else "normal"
        self.send_btn.config(state=state)
        self.mic_btn.config(state="normal" if not busy and main.WHISPER is not None else "disabled")
        if not busy:
            self.status.config(text=self.ready_text())

    # ============================================================
    # 入力
    # ============================================================
    def send(self):
        text = self.entry.get().strip()
        if not text or self.busy:
            return
        self.entry.delete(0, "end")
        self.run_request(text)

    def toggle_voice(self):
        if self.recorder.stream is None:
            self.recorder.start()
            self.mic_btn.config(text="■ 止める", bg=RED, fg="white")
            self.send_btn.config(state="disabled")
            self.status.config(text="録音中です。話し終えたら「止める」を押してください")
            return
        audio = self.recorder.stop()
        self.mic_btn.config(text="🎤 話す", bg=GRAY, fg="black")
        if audio is None:
            self.set_busy(False)
            return
        self.set_busy(True, "聞き取り中です")

        def work():
            text = main.transcribe(main.WHISPER, audio)
            self.root.after(0, lambda: self.run_request(text, transcribed=True) if text else self.set_busy(False))
        threading.Thread(target=work, daemon=True).start()

    def run_request(self, text, transcribed=False):
        if not transcribed:
            self.say(f"\nあなた: {text}\n", "you")
        self.set_busy(True)

        def work():
            try:
                answer = self.agent.run(text)
                self.log_queue.put((f"🤖 {answer}\n", "ai"))
                if self.voice.get():
                    main.speak(answer)
            except requests.RequestException:
                print("⚠ AI（Ollama）に接続できません。スタートメニューから「Ollama」を起動してください。\n")
            except Exception as e:
                print(f"⚠ うまくいきませんでした: {e}\n")
            finally:
                self.root.after(0, lambda: self.set_busy(False))
        threading.Thread(target=work, daemon=True).start()

    def show_plan(self, steps):
        lines = "".join(f"   {i}. {step}\n" for i, step in enumerate(steps, 1))
        self.log_queue.put((f"📝 やることを考えました:\n{lines}", "plan"))

    def new_conversation(self):
        if self.agent and messagebox.askyesno("新しい会話", "これまでの会話を忘れて、新しく始めますか？", parent=self.root):
            self.agent.forget()
            self.log.config(state="normal")
            self.log.delete("1.0", "end")
            self.log.config(state="disabled")
            self.say("新しい会話を始めます。ご用件をどうぞ。\n", "note")

    # ============================================================
    # 自分から気づいて知らせる
    # ============================================================
    def monitor(self):
        """予定とお任せ作業を見回り、気づいたことを通知・読み上げる"""
        def work():
            try:
                alerts = self.agent.check_alerts()
            except Exception:
                alerts = []
            for title, body in alerts:
                main.notify(title, body)
                self.log_queue.put((f"\n🔔 {body}\n", "alert"))
            if alerts and self.voice.get():
                main.speak("。".join(body for _, body in alerts))
        if not self.busy:
            threading.Thread(target=work, daemon=True).start()
        self.root.after(MONITOR_MINUTES * 60 * 1000, self.monitor)

    # ============================================================
    # main.py から呼ばれる確認・質問（作業スレッドから画面スレッドに頼む）
    # ============================================================
    def on_ui(self, fn, *args):
        result, done = {}, threading.Event()

        def run():
            try:
                result["value"] = fn(*args)
            finally:
                done.set()
        self.root.after(0, run)
        done.wait()
        return result.get("value")

    def confirm(self, question, code=None):
        if not main.CONFIRM_BEFORE_RUN:
            return True
        explanation = ""
        if code and not code.lstrip().startswith("<"):
            self.busy_text = "内容を確認しています"
            explanation = main.explain_code(code)
        return bool(self.on_ui(self._confirm_dialog, question, code, explanation))

    def _confirm_dialog(self, question, code, explanation):
        win = self.dialog("確認")
        tk.Label(win, text=question, font=FONT_BIG).pack(padx=24, pady=(18, 8))
        if explanation:
            tk.Label(win, text="この作業ですること:", font=FONT, anchor="w").pack(fill="x", padx=24)
            tk.Label(win, text=explanation, font=FONT, justify="left", anchor="w", bg="white",
                     wraplength=int(560 * self.scale), padx=12, pady=8).pack(fill="x", padx=24, pady=4)
        if code and main.DANGEROUS.search(code):
            tk.Label(win, text="⚠ ファイルの削除やPCの設定変更が含まれています。よく確認してください。",
                     font=FONT, fg=RED, wraplength=int(560 * self.scale)).pack(padx=24, pady=4)
        if code:
            box = scrolledtext.ScrolledText(win, font=FONT_CODE, height=12, width=72)
            box.insert("1.0", code)
            box.config(state="disabled")
            toggle = tk.Button(win, text="▶ 中身（コード）を見る", font=FONT_SMALL, relief="flat", bg=BG,
                               fg=BLUE, cursor="hand2")

            def show():
                if box.winfo_ismapped():
                    box.pack_forget()
                    toggle.config(text="▶ 中身（コード）を見る")
                else:
                    box.pack(padx=24, pady=4, after=toggle)
                    toggle.config(text="▼ 中身（コード）を隠す")
            toggle.config(command=show)
            toggle.pack(anchor="w", padx=20)
            if not explanation:
                show()
        answer = {"value": False}
        row = tk.Frame(win)
        row.pack(pady=16)
        button(row, "はい、お願いします", lambda: (answer.update(value=True), win.destroy()), primary=True
               ).pack(side="left", padx=6)
        button(row, "やめる", win.destroy).pack(side="left", padx=6)
        self.root.wait_window(win)
        return answer["value"]

    def ask_user(self, question):
        return self.on_ui(self._ask_dialog, question) or ""

    def _ask_dialog(self, question):
        win = self.dialog("入力")
        tk.Label(win, text=question, font=FONT_BIG, wraplength=int(520 * self.scale)).pack(padx=24, pady=(18, 4))
        tk.Label(win, text="分からなければ「スキップ」を押してください", font=FONT_SMALL, fg="#555").pack()
        row = tk.Frame(win)
        row.pack(padx=24, pady=12)
        entry = tk.Entry(row, font=FONT, width=26, relief="solid", bd=1)
        entry.pack(side="left", ipady=4)
        entry.focus_set()
        if main.WHISPER is not None:
            self.voice_button(row, entry).pack(side="left", padx=(6, 0))
        answer = {"value": ""}

        def ok():
            answer["value"] = entry.get().strip()
            win.destroy()
        entry.bind("<Return>", lambda _: ok())
        buttons = tk.Frame(win)
        buttons.pack(pady=(0, 16))
        button(buttons, "OK", ok, primary=True).pack(side="left", padx=6)
        button(buttons, "スキップ", win.destroy).pack(side="left", padx=6)
        self.root.wait_window(win)
        return answer["value"]

    def voice_button(self, parent, entry):
        """押して話すと、聞き取った文字を entry に入れるボタン"""
        recorder = Recorder()

        def toggle():
            if recorder.stream is None:
                recorder.start()
                mic.config(text="■ 止める", bg=RED, fg="white")
                return
            audio = recorder.stop()
            mic.config(text="聞き取り中...", bg=GRAY, fg="black", state="disabled")

            def work():
                text = (main.transcribe(main.WHISPER, audio, label="聞き取った内容") or "") if audio is not None else ""
                self.root.after(0, lambda: (entry.delete(0, "end"), entry.insert(0, text.rstrip("。、 ")),
                                            mic.config(text="🎤 声で入力", state="normal")))
            threading.Thread(target=work, daemon=True).start()
        mic = button(parent, "🎤 声で入力", toggle)
        return mic

    def ask_message(self):
        """返信したいメッセージを貼り付けてもらう（コピー済みなら最初から入れておく）"""
        return self.on_ui(self._message_dialog) or ""

    def _message_dialog(self):
        win = self.dialog("返信したいメッセージ")
        tk.Label(win, text="返信したいメッセージを貼り付けてください", font=FONT_BIG).pack(padx=24, pady=(18, 4))
        tk.Label(win, text="メールやチャットの文章をコピーして、下の欄で Ctrl+V を押します",
                 font=FONT_SMALL, fg="#555").pack()
        box = scrolledtext.ScrolledText(win, font=FONT, height=10, width=56, wrap="word")
        box.pack(padx=24, pady=10)
        try:
            box.insert("1.0", self.root.clipboard_get())
        except tk.TclError:
            pass
        box.focus_set()
        answer = {"value": ""}
        row = tk.Frame(win)
        row.pack(pady=(0, 16))
        button(row, "このメッセージに返信", lambda: (answer.update(value=box.get("1.0", "end").strip()),
                                                   win.destroy()), primary=True).pack(side="left", padx=6)
        button(row, "やめる", win.destroy).pack(side="left", padx=6)
        self.root.wait_window(win)
        return answer["value"]

    # ============================================================
    # 設定: 自分の情報・予定・お任せ一覧
    # ============================================================
    def open_profile(self, first=False):
        text = main.load_profile()

        def field(label):
            m = re.search(rf"{label}[:：]\s*(.+)", text)
            return m.group(1).strip() if m else ""
        priorities = re.findall(r"^\d+\.\s*(.+)$", text, re.MULTILINE)

        win = self.dialog("自分の情報")
        if first:
            tk.Label(win, text="はじめに、あなたのことを教えてください", font=FONT_BIG).pack(padx=24, pady=(18, 0))
            tk.Label(win, text="返信の署名や、予定が重なったときの判断に使います。あとから変更できます。",
                     font=FONT_SMALL, fg="#555").pack(padx=24)
        form = tk.Frame(win)
        form.pack(padx=24, pady=12)
        entries = {}
        for i, (key, label, hint) in enumerate((("名前", "名前", "例: 山田 太郎"),
                                                 ("所属", "所属", "例: 神戸電子専門学校 AI学科"),
                                                 ("返信の署名", "返信の署名", "例: 山田"))):
            tk.Label(form, text=label, font=FONT).grid(row=i * 2, column=0, sticky="w", pady=(6, 0))
            entry = tk.Entry(form, font=FONT, width=34, relief="solid", bd=1)
            entry.insert(0, field(key))
            entry.grid(row=i * 2, column=1, sticky="w", padx=(10, 0), pady=(6, 0))
            tk.Label(form, text=hint, font=FONT_SMALL, fg="#888").grid(row=i * 2 + 1, column=1, sticky="w", padx=10)
            entries[key] = entry
        tk.Label(form, text="予定が重なったとき\nの優先順位", font=FONT, justify="left").grid(
            row=6, column=0, sticky="nw", pady=(10, 0))
        prio = tk.Text(form, font=FONT, width=34, height=5, relief="solid", bd=1)
        prio.insert("1.0", "\n".join(priorities or ["授業・試験", "発表会・コンテスト", "アルバイト", "友人との約束"]))
        prio.grid(row=6, column=1, sticky="w", padx=(10, 0), pady=(10, 0))
        tk.Label(form, text="大事なものから順に、1行に1つ", font=FONT_SMALL, fg="#888").grid(
            row=7, column=1, sticky="w", padx=10)

        def save():
            items = [line.strip() for line in prio.get("1.0", "end").splitlines() if line.strip()]
            body = ["# 自分の情報", ""] + [f"- {k}: {e.get().strip()}" for k, e in entries.items() if e.get().strip()]
            body += ["", "## 予定が重なったときの優先順位"] + [f"{n}. {item}" for n, item in enumerate(items, 1)]
            with open(main.PROFILE_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(body) + "\n")
            win.destroy()
            self.say("✓ 自分の情報を保存しました。\n", "note")
        row = tk.Frame(win)
        row.pack(pady=(0, 16))
        button(row, "保存", save, primary=True).pack(side="left", padx=6)
        button(row, "あとで" if first else "閉じる", win.destroy).pack(side="left", padx=6)

    def open_calendar(self):
        win = self.dialog("予定を読み込む")
        tk.Label(win, text="予定（カレンダー）を読み込む", font=FONT_BIG).pack(padx=24, pady=(18, 4))
        tk.Label(win, justify="left", font=FONT_SMALL, fg="#555", text=(
            "Googleカレンダーの場合: パソコンで「設定 → インポート/エクスポート → エクスポート」を押すと、\n"
            "予定のファイル（.ics）がダウンロードできます。Outlookでも「予定表の保存」で作れます。")).pack(padx=24)
        listbox = tk.Listbox(win, font=FONT, height=5, width=48)
        listbox.pack(padx=24, pady=10)

        def refresh():
            listbox.delete(0, "end")
            files = sorted(os.listdir(main.CALENDAR_DIR))
            for name in files:
                if name.endswith(".ics"):
                    listbox.insert("end", name)
            if listbox.size() == 0:
                listbox.insert("end", "（まだ読み込まれていません）")

        def add():
            paths = filedialog.askopenfilenames(parent=win, title="予定のファイルを選ぶ",
                                                filetypes=[("カレンダー", "*.ics")])
            for path in paths:
                shutil.copy(path, main.CALENDAR_DIR)
            if paths:
                self.say(f"✓ 予定のファイルを {len(paths)} 個読み込みました。\n", "note")
            refresh()

        def remove():
            for i in listbox.curselection():
                name = listbox.get(i)
                if name.endswith(".ics"):
                    os.remove(os.path.join(main.CALENDAR_DIR, name))
            refresh()
        refresh()
        row = tk.Frame(win)
        row.pack(pady=(0, 16))
        button(row, "ファイルを選ぶ", add, primary=True).pack(side="left", padx=6)
        button(row, "選んだものを外す", remove).pack(side="left", padx=6)
        button(row, "閉じる", win.destroy).pack(side="left", padx=6)

    def open_tasks(self):
        win = self.dialog("お任せ一覧")
        tk.Label(win, text="お任せした作業", font=FONT_BIG).pack(padx=24, pady=(18, 4))
        table = tk.Frame(win)
        table.pack(padx=24, pady=10, fill="x")

        def refresh():
            for child in table.winfo_children():
                child.destroy()
            entries = main.load_delegated()
            if not entries:
                tk.Label(table, text="まだ何もお任せされていません。\n「毎朝9時に〜しといて」のように頼むと、ここに表示されます。",
                         font=FONT, fg="#555").pack(pady=10)
                return
            loading = tk.Label(table, text="状況を確認しています...", font=FONT, fg="#555")
            loading.pack()

            def work():
                rows = [(e, main.task_status(e, os.path.exists(e["log"]))) for e in entries]
                self.root.after(0, lambda: show(rows, loading))
            threading.Thread(target=work, daemon=True).start()

        def show(rows, loading):
            if not win.winfo_exists():
                return
            loading.destroy()
            for i, header in enumerate(("作業", "いつ", "状況", "")):
                tk.Label(table, text=header, font=(FONT[0], 10, "bold")).grid(row=0, column=i, sticky="w", padx=6)
            for r, (entry, status) in enumerate(rows, 1):
                tk.Label(table, text=entry["task"], font=FONT_SMALL, wraplength=int(240 * self.scale),
                         justify="left").grid(row=r, column=0, sticky="w", padx=6, pady=3)
                tk.Label(table, text=entry["when"], font=FONT_SMALL).grid(row=r, column=1, sticky="w", padx=6)
                tk.Label(table, text=status, font=FONT_SMALL, wraplength=int(220 * self.scale), justify="left",
                         fg=RED if status.startswith("エラー") else "black").grid(row=r, column=2, sticky="w", padx=6)
                tk.Button(table, text="取り消す", font=FONT_SMALL, relief="groove", cursor="hand2",
                          command=lambda e=entry: cancel(e)).grid(row=r, column=3, padx=6)

        def cancel(entry):
            if messagebox.askyesno("確認", f"「{entry['task']}」のお任せを取り消しますか？", parent=win):
                main.cancel_delegated(entry)
                self.say(f"✓ 「{entry['task']}」のお任せを取り消しました。\n", "note")
                refresh()
        refresh()
        button(win, "閉じる", win.destroy).pack(pady=(0, 16))

    def dialog(self, title):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.transient(self.root)
        win.grab_set()
        win.resizable(False, False)
        win.configure(padx=4, pady=4)
        win.update_idletasks()
        x = self.root.winfo_rootx() + int(80 * self.scale)
        y = self.root.winfo_rooty() + int(60 * self.scale)
        win.geometry(f"+{x}+{y}")
        return win


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
