"""第二の自分 — AIエージェントの画面（「起動.bat」かデスクトップのショートカットで起動）

処理の中身は main.py の4段階構成（Whisper → Ollama → Python連携コア → PowerShell実行）と
agent.py のエージェントをそのまま使い、この画面は入力・確認・設定・表示・見回りだけを担当する。
"""

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    import customtkinter as ctk
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

# --- 色（ライト, ダーク） ---
ACCENT = ("#6C5CE7", "#8B7CF6")
ACCENT_HOVER = ("#5A4BD1", "#7A6AE8")
BG = ("#F4F5FB", "#14151B")
SURFACE = ("#FFFFFF", "#20212B")
SIDEBAR = ("#ECEBFA", "#1A1B23")
BORDER = ("#E3E3EE", "#2D2E3A")
TEXT = ("#1E1E2A", "#ECECF1")
MUTED = ("#6B6F80", "#9A9DB0")
TOOL_BG, TOOL_FG = ("#E6F6EE", "#1B3A2B"), ("#1F7A4D", "#7EE2A8")
PLAN_BG = ("#F0EDFF", "#272340")
ALERT_BG, ALERT_FG = ("#FFF1E6", "#3A2A1E"), ("#B4530B", "#FFB27A")
DANGER = ("#D92D20", "#FF7A70")
READY, BUSY, OFF = "#22C55E", "#F59E0B", "#EF4444"

EXAMPLES = [
    "予定を確認して、重なってたら調整案を作って",
    "明日の朝8時に発表会の準備をするよう知らせて",
    "毎朝9時にディスクの空き容量を記録しといて",
    "任せてた作業どうだった？",
]

MONITOR_MINUTES = 15  # 自分から見回る間隔
OLLAMA_APP = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama app.exe")
FONT_FAMILY = "Yu Gothic UI"
SETTINGS_FILE = os.path.join("my_powershell_tools", "settings.json")
DEFAULT_DEVICE = "Windowsの既定"


def load_settings():
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_settings(**values):
    settings = {**load_settings(), **values}
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False)

ctk.set_appearance_mode("System")


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


def transcribe(audio):
    """録音した音声をローカルの Whisper で文字にする"""
    segments, _ = main.WHISPER.transcribe(audio, language="ja")
    return "".join(s.text for s in segments).strip().rstrip("。、 ")


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
    """main.py / agent.py の print を、1行ごとにまとめて画面に流す"""

    def __init__(self, q):
        self.q, self.buffer = q, ""

    def write(self, text):
        self.buffer += text
        if self.buffer.endswith("\n"):
            chunk, self.buffer = self.buffer.strip("\n").replace("[AI] ", ""), ""
            if chunk.strip():
                self.q.put(("note", chunk))

    def flush(self):
        pass


class App:
    def __init__(self, root):
        self.root = root
        self.queue = queue.Queue()
        self.recorder = Recorder()
        self.mic = has_microphone()
        self.busy, self.busy_text, self.dots = False, "", 0
        self.agent = None
        self.voice = tk.BooleanVar(value=True)
        self.plan_items, self.typing, self.wrapped = [], None, []

        self.font = lambda size=14, weight="normal": ctk.CTkFont(family=FONT_FAMILY, size=size, weight=weight)
        root.title("第二の自分")
        root.geometry("1120x760")
        root.minsize(860, 600)
        root.configure(fg_color=BG)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        self.build_sidebar()
        self.build_chat()

        main.VERBOSE = False
        sys.stdout = sys.stderr = QueueWriter(self.queue)
        main.confirm = self.confirm
        main.ask_user = self.ask_user
        main.read_clipboard = self.ask_message
        self.poll()
        self.startup()

    # ============================================================
    # 画面の組み立て
    # ============================================================
    def build_sidebar(self):
        side = ctk.CTkFrame(self.root, width=250, corner_radius=0, fg_color=SIDEBAR)
        side.grid(row=0, column=0, sticky="nsw")
        side.grid_propagate(False)
        side.grid_columnconfigure(0, weight=1)

        self.circle(side, 72, 30).grid(row=0, column=0, pady=(32, 10))
        ctk.CTkLabel(side, text="第二の自分", font=self.font(20, "bold"), text_color=TEXT).grid(row=1, column=0)
        ctk.CTkLabel(side, text="あなたの代わりに考えて動く\nAIエージェント", font=self.font(12),
                     text_color=MUTED, justify="center").grid(row=2, column=0, pady=(2, 10))

        status = ctk.CTkFrame(side, fg_color=SURFACE, corner_radius=14)
        status.grid(row=3, column=0, padx=18, pady=(4, 18), sticky="ew")
        self.status_dot = ctk.CTkLabel(status, text="●", font=self.font(14), text_color=BUSY, width=16)
        self.status_dot.pack(side="left", padx=(12, 4), pady=8)
        self.status_label = ctk.CTkLabel(status, text="準備中です", font=self.font(12), text_color=TEXT,
                                         anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True, padx=(0, 10))

        for row, (text, command) in enumerate((("＋  新しい会話", self.new_conversation),
                                               ("👤  自分の情報", self.open_profile),
                                               ("📅  予定を読み込む", self.open_calendar),
                                               ("📋  お任せ一覧", self.open_tasks)), start=4):
            ctk.CTkButton(side, text=text, command=command, anchor="w", height=40, corner_radius=12,
                          font=self.font(14), fg_color="transparent", text_color=TEXT,
                          hover_color=SURFACE).grid(row=row, column=0, padx=14, pady=2, sticky="ew")

        side.grid_rowconfigure(8, weight=1)
        bottom = ctk.CTkFrame(side, fg_color="transparent")
        bottom.grid(row=9, column=0, padx=18, pady=18, sticky="ew")
        ctk.CTkSwitch(bottom, text="声で返事", variable=self.voice, font=self.font(13), text_color=TEXT,
                      progress_color=ACCENT).pack(anchor="w", pady=(0, 8))
        # 声の出力先（聞こえないときに、聞こえるスピーカーを選べるように）
        ctk.CTkLabel(bottom, text="声の出力先", font=self.font(11), text_color=MUTED).pack(anchor="w")
        devices = {name: index for index, name in main.output_devices()}
        saved = load_settings().get("voice_device")
        if saved in devices:
            main.VOICE_DEVICE = devices[saved]
        voice_row = ctk.CTkFrame(bottom, fg_color="transparent")
        voice_row.pack(fill="x", pady=(2, 12))

        def choose(name):
            main.VOICE_DEVICE = devices.get(name)
            save_settings(voice_device=name)
            main.speak("この声が聞こえれば大丈夫です。")
        menu = ctk.CTkOptionMenu(voice_row, values=[DEFAULT_DEVICE] + list(devices), command=choose,
                                 font=self.font(11), dropdown_font=self.font(12), width=150, height=28,
                                 fg_color=SURFACE, text_color=TEXT, button_color=ACCENT,
                                 button_hover_color=ACCENT_HOVER, dynamic_resizing=False)
        menu.set(saved if saved in devices else DEFAULT_DEVICE)
        menu.pack(side="left")
        ctk.CTkButton(voice_row, text="🔈", width=34, height=28, corner_radius=14, font=self.font(13),
                      fg_color=SURFACE, text_color=TEXT, hover_color=BORDER,
                      command=lambda: main.speak("第二の自分の声のテストです。聞こえていますか。")).pack(
            side="left", padx=(6, 0))
        ctk.CTkLabel(bottom, text="テーマ", font=self.font(11), text_color=MUTED).pack(anchor="w")
        theme = ctk.CTkSegmentedButton(bottom, values=["自動", "ライト", "ダーク"], font=self.font(12),
                                       selected_color=ACCENT, selected_hover_color=ACCENT_HOVER,
                                       command=lambda v: ctk.set_appearance_mode(
                                           {"自動": "System", "ライト": "Light", "ダーク": "Dark"}[v]))
        theme.set("自動")
        theme.pack(fill="x", pady=(2, 12))
        self.brain_label = ctk.CTkLabel(bottom, text="", font=self.font(11), text_color=MUTED, anchor="w")
        self.brain_label.pack(anchor="w")

    def build_chat(self):
        area = ctk.CTkFrame(self.root, fg_color="transparent")
        area.grid(row=0, column=1, sticky="nsew")
        area.grid_columnconfigure(0, weight=1)
        area.grid_rowconfigure(0, weight=1)

        self.chat = ctk.CTkScrollableFrame(area, fg_color="transparent")
        self.chat.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=(16, 0))
        self.chat.bind("<Configure>", lambda e: self.rewrap(e.width))

        chips = ctk.CTkFrame(area, fg_color="transparent")
        chips.grid(row=1, column=0, sticky="ew", padx=28, pady=(8, 4))
        for i, text in enumerate(EXAMPLES):
            ctk.CTkButton(chips, text=text, command=lambda t=text: self.fill(t), height=30, corner_radius=15,
                          font=self.font(12), fg_color=SURFACE, text_color=TEXT, hover_color=PLAN_BG,
                          border_width=1, border_color=BORDER).grid(row=i // 2, column=i % 2, padx=4, pady=3,
                                                                    sticky="w")

        composer = ctk.CTkFrame(area, fg_color=SURFACE, corner_radius=26, border_width=1, border_color=BORDER)
        composer.grid(row=2, column=0, sticky="ew", padx=24, pady=(4, 20))
        composer.grid_columnconfigure(0, weight=1)
        self.entry = ctk.CTkEntry(composer, height=48, border_width=0, fg_color="transparent",
                                  font=self.font(15), text_color=TEXT,
                                  placeholder_text="頼みたいことを入力…（例: 〜しといて）")
        self.entry.grid(row=0, column=0, sticky="ew", padx=(20, 6), pady=6)
        self.entry.bind("<Return>", lambda _: self.send())
        self.mic_btn = ctk.CTkButton(composer, text="🎤", width=44, height=44, corner_radius=22,
                                     font=self.font(18), fg_color=BG, text_color=TEXT, hover_color=BORDER,
                                     command=self.toggle_voice, state="disabled")
        self.mic_btn.grid(row=0, column=1, padx=4)
        self.send_btn = ctk.CTkButton(composer, text="➤", width=44, height=44, corner_radius=22,
                                      font=self.font(18, "bold"), fg_color=ACCENT, hover_color=ACCENT_HOVER,
                                      command=self.send)
        self.send_btn.grid(row=0, column=2, padx=(4, 8))

    # ============================================================
    # チャットの表示（吹き出し）
    # ============================================================
    def rewrap(self, width):
        for label, ratio in self.wrapped:
            if label.winfo_exists():
                label.configure(wraplength=max(260, int(width * ratio)))

    def scroll_down(self):
        self.root.after(50, lambda: self.chat._parent_canvas.yview_moveto(1.0))

    def row(self):
        frame = ctk.CTkFrame(self.chat, fg_color="transparent")
        frame.pack(fill="x", padx=8, pady=5)
        return frame

    def label(self, parent, text, ratio=0.62, **kw):
        kw.setdefault("font", self.font(14))
        kw.setdefault("text_color", TEXT)
        lbl = ctk.CTkLabel(parent, text=text, justify="left", anchor="w",
                           wraplength=max(260, int(self.chat.winfo_width() * ratio)), **kw)
        self.wrapped.append((lbl, ratio))
        return lbl

    def circle(self, parent, size, font_size):
        """アバター（大きさを固定した枠の中に文字を置き、正円にする）"""
        frame = ctk.CTkFrame(parent, width=size, height=size, corner_radius=size // 2, fg_color=ACCENT)
        frame.pack_propagate(False)
        frame.grid_propagate(False)
        ctk.CTkLabel(frame, text="自", text_color="white", font=self.font(font_size, "bold"),
                     fg_color="transparent").place(relx=0.5, rely=0.5, anchor="center")
        return frame

    def avatar(self, parent):
        self.circle(parent, 34, 14).pack(side="left", anchor="n", padx=(0, 10))

    def bubble_user(self, text):
        bubble = ctk.CTkFrame(self.row(), fg_color=ACCENT, corner_radius=18)
        bubble.pack(side="right")
        self.label(bubble, text, ratio=0.55, text_color="white").pack(padx=16, pady=10)

    def bubble_ai(self, text):
        row = self.row()
        self.avatar(row)
        bubble = ctk.CTkFrame(row, fg_color=SURFACE, corner_radius=18, border_width=1, border_color=BORDER)
        bubble.pack(side="left")
        self.label(bubble, text).pack(padx=16, pady=10)

    def show_plan_card(self, steps):
        row = self.row()
        card = ctk.CTkFrame(row, fg_color=PLAN_BG, corner_radius=16)
        card.pack(side="left", padx=(44, 0))
        ctk.CTkLabel(card, text="📝  やることを考えました", font=self.font(13, "bold"),
                     text_color=ACCENT).pack(anchor="w", padx=16, pady=(10, 4))
        self.plan_items = []
        # 「(get_schedule)」のような道具の名前は、画面には出さない
        steps = [re.sub(r"\s*[(（][^()（）]*[a-z]+_[a-z_]+[^()（）]*[)）]", "", s).strip() or s for s in steps]
        for i, step in enumerate(steps, 1):
            item = self.label(card, f"○  {i}. {step}", ratio=0.55, font=self.font(13))
            item.pack(anchor="w", padx=16, pady=1)
            self.plan_items.append([item, f"{i}. {step}", False])
        ctk.CTkFrame(card, height=8, fg_color="transparent").pack()

    def show_tool(self, label):
        for item in self.plan_items:  # やることリストの次の項目にチェックを付ける
            if not item[2]:
                item[0].configure(text=f"✓  {item[1]}", text_color=TOOL_FG)
                item[2] = True
                break
        pill = ctk.CTkLabel(self.row(), text=f"🔧  {label}…", font=self.font(12), text_color=TOOL_FG,
                            fg_color=TOOL_BG, corner_radius=12)
        pill.pack(side="left", padx=(44, 0), ipadx=10, ipady=2)

    def show_note(self, text):
        if text.count("\n") >= 3:  # 実行結果や下書きは、等幅のカードで見やすく
            card = ctk.CTkFrame(self.row(), fg_color=SURFACE, corner_radius=12, border_width=1, border_color=BORDER)
            card.pack(side="left", padx=(44, 0), fill="x", expand=True)
            self.label(card, text, ratio=0.7, font=ctk.CTkFont(family="Consolas", size=12),
                       text_color=MUTED).pack(anchor="w", padx=14, pady=10)
            return
        color = DANGER if text.startswith("⚠") else MUTED
        self.label(self.row(), text, ratio=0.7, font=self.font(12), text_color=color).pack(
            side="left", padx=(44, 0))

    def show_alert(self, text):
        card = ctk.CTkFrame(self.row(), fg_color=ALERT_BG, corner_radius=16)
        card.pack(side="left", padx=(44, 0))
        self.label(card, f"🔔  {text}", font=self.font(14, "bold"), text_color=ALERT_FG).pack(padx=16, pady=10)

    def show_typing(self, show):
        if show and self.typing is None:
            row = self.row()
            self.avatar(row)
            self.typing = ctk.CTkLabel(row, text="", font=self.font(13), text_color=MUTED, fg_color=SURFACE,
                                       corner_radius=16)
            self.typing.pack(side="left", ipadx=14, ipady=6)
            self.typing_row = row
        elif not show and self.typing is not None:
            self.typing_row.destroy()
            self.typing = None

    def poll(self):
        items = []
        while not self.queue.empty():
            items.append(self.queue.get())
        if items:
            typing = self.typing is not None
            self.show_typing(False)  # 「考え中」は常に最後に表示する
            for kind, payload in items:
                {"user": self.bubble_user, "ai": self.bubble_ai, "plan": self.show_plan_card,
                 "tool": self.show_tool, "note": self.show_note, "alert": self.show_alert}[kind](payload)
            self.show_typing(typing and self.busy)
            self.scroll_down()
        if self.busy:  # 考え中は「●」を動かして、止まっていないことを伝える
            self.dots = (self.dots + 1) % 3
            dots = "".join("●" if i == self.dots else "○" for i in range(3))
            if self.typing is not None:
                self.typing.configure(text=f"{self.busy_text}  {dots}")
        self.root.after(350 if self.busy else 100, self.poll)

    def set_status(self, text, color):
        self.status_label.configure(text=text)
        self.status_dot.configure(text_color=color)

    def set_busy(self, busy, text="考え中"):
        self.busy, self.busy_text = busy, text
        self.send_btn.configure(state="disabled" if busy else "normal")
        self.mic_btn.configure(state="normal" if not busy and main.WHISPER is not None else "disabled")
        self.show_typing(busy and self.agent is not None)
        if busy:
            self.set_status(f"{text}…", BUSY)
        else:
            self.set_status("準備できました" if self.agent else "AIに接続できません",
                            READY if self.agent else OFF)
            self.plan_items = [[lbl, t, True] for lbl, t, done in self.plan_items]  # 残りの手順は付け替えない

    def fill(self, text):
        self.entry.delete(0, "end")
        self.entry.insert(0, text)
        self.entry.focus_set()

    # ============================================================
    # 起動時
    # ============================================================
    def startup(self):
        self.queue.put(("ai", "こんにちは、第二の自分です。頼みたいことを下の欄に入力してください。\n"
                              "「〜しといて」と頼めば、あなたが休んでいる間にこなしておきます。"))
        self.set_busy(True, "準備中")
        threading.Thread(target=self.prepare, daemon=True).start()
        if not os.path.exists(main.PROFILE_FILE):
            self.root.after(800, lambda: self.open_profile(first=True))

    def prepare(self):
        """Ollama が止まっていれば起動し、エージェントと音声認識を準備する"""
        if not ollama_running() and os.path.exists(OLLAMA_APP):
            subprocess.Popen([OLLAMA_APP])
            for _ in range(30):
                if ollama_running():
                    break
                time.sleep(1)
        if not ollama_running():
            print("⚠ AI（Ollama）を起動できませんでした。スタートメニューから「Ollama」を起動してください。")
        else:
            self.agent = Agent(on_tool=lambda label: self.queue.put(("tool", label)),
                               on_plan=lambda steps: self.queue.put(("plan", steps)))
            if self.agent.history:
                print("前回の会話を覚えています。続きから話せます（忘れさせるときは「＋ 新しい会話」）。")
            self.root.after(60 * 1000, self.monitor)
        if self.mic:
            main.WHISPER = main.WhisperModel(main.WHISPER_MODEL, device="cpu", compute_type="int8")
        mic = "音声入力: 使えます" if main.WHISPER is not None else "音声入力: マイクなし"
        brain = f"頭脳: {self.agent.model}" if self.agent else "頭脳: 未接続"
        self.root.after(0, lambda: (self.brain_label.configure(text=f"{brain}\n{mic}"), self.set_busy(False)))

    # ============================================================
    # 入力
    # ============================================================
    def send(self):
        text = self.entry.get().strip()
        if not text or self.busy or self.agent is None:
            return
        self.entry.delete(0, "end")
        self.run_request(text)

    def toggle_voice(self):
        if self.recorder.stream is None:
            self.recorder.start()
            self.mic_btn.configure(text="■", fg_color=DANGER, text_color="white")
            self.send_btn.configure(state="disabled")
            self.set_status("録音中…もう一度押すと止まります", OFF)
            return
        audio = self.recorder.stop()
        self.mic_btn.configure(text="🎤", fg_color=BG, text_color=TEXT)
        if audio is None:
            self.set_busy(False)
            return
        self.set_busy(True, "聞き取り中")

        def work():
            text = transcribe(audio)
            self.root.after(0, lambda: self.run_request(text) if text else self.set_busy(False))
        threading.Thread(target=work, daemon=True).start()

    def run_request(self, text):
        self.queue.put(("user", text))
        self.set_busy(True)

        def work():
            try:
                answer = self.agent.run(text)
                self.queue.put(("ai", answer))
                if self.voice.get():
                    main.speak(answer)
            except requests.RequestException:
                print("⚠ AI（Ollama）に接続できません。スタートメニューから「Ollama」を起動してください。")
            except Exception as e:
                print(f"⚠ うまくいきませんでした: {e}")
            finally:
                self.root.after(0, lambda: self.set_busy(False))
        threading.Thread(target=work, daemon=True).start()

    def new_conversation(self):
        if self.agent and messagebox.askyesno("新しい会話", "これまでの会話を忘れて、新しく始めますか？", parent=self.root):
            self.agent.forget()
            for child in self.chat.winfo_children():
                child.destroy()
            self.wrapped, self.plan_items, self.typing = [], [], None
            self.queue.put(("ai", "新しい会話を始めます。ご用件をどうぞ。"))

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
                self.queue.put(("alert", body))
            if alerts and self.voice.get():
                main.speak("。".join(body for _, body in alerts))
        if not self.busy:
            threading.Thread(target=work, daemon=True).start()
        self.root.after(MONITOR_MINUTES * 60 * 1000, self.monitor)

    # ============================================================
    # main.py / agent.py から呼ばれる確認・質問（作業スレッドから画面スレッドに頼む）
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

    def dialog(self, title):
        win = ctk.CTkToplevel(self.root)
        win.title(title)
        win.configure(fg_color=BG)
        win.transient(self.root)
        win.resizable(False, False)
        win.geometry(f"+{self.root.winfo_rootx() + 260}+{self.root.winfo_rooty() + 80}")
        win.after(100, lambda: (win.lift(), win.focus_force(), win.grab_set()))
        return win

    def heading(self, win, title, sub=None):
        ctk.CTkLabel(win, text=title, font=self.font(20, "bold"), text_color=TEXT).pack(
            anchor="w", padx=28, pady=(24, 2))
        if sub:
            ctk.CTkLabel(win, text=sub, font=self.font(12), text_color=MUTED, justify="left").pack(
                anchor="w", padx=28)

    def buttons(self, win, primary, on_primary, secondary, on_secondary):
        row = ctk.CTkFrame(win, fg_color="transparent")
        row.pack(fill="x", padx=28, pady=(12, 24))
        ctk.CTkButton(row, text=primary, command=on_primary, height=40, corner_radius=20, font=self.font(14, "bold"),
                      fg_color=ACCENT, hover_color=ACCENT_HOVER).pack(side="right")
        ctk.CTkButton(row, text=secondary, command=on_secondary, height=40, corner_radius=20, font=self.font(14),
                      fg_color="transparent", text_color=TEXT, border_width=1, border_color=BORDER,
                      hover_color=SURFACE).pack(side="right", padx=8)

    def confirm(self, question, code=None):
        if not main.CONFIRM_BEFORE_RUN:
            return True
        explanation = ""
        if code and not code.lstrip().startswith("<"):
            self.busy_text = "内容を確認中"
            explanation = main.explain_code(code)
        return bool(self.on_ui(self._confirm_dialog, question, code, explanation))

    def _confirm_dialog(self, question, code, explanation):
        win = self.dialog("確認")
        self.heading(win, question)
        body = ctk.CTkFrame(win, fg_color="transparent")
        body.pack(fill="x", padx=28, pady=(10, 0))
        if explanation:
            card = ctk.CTkFrame(body, fg_color=SURFACE, corner_radius=14, border_width=1, border_color=BORDER)
            card.pack(fill="x")
            ctk.CTkLabel(card, text="この作業ですること", font=self.font(12, "bold"), text_color=ACCENT).pack(
                anchor="w", padx=16, pady=(12, 2))
            ctk.CTkLabel(card, text=explanation, font=self.font(14), text_color=TEXT, justify="left",
                         wraplength=520).pack(anchor="w", padx=16, pady=(0, 12))
        if code and main.DANGEROUS.search(code):
            ctk.CTkLabel(body, text="⚠ ファイルの削除やPCの設定変更が含まれています。よく確認してください。",
                         font=self.font(13, "bold"), text_color=DANGER, wraplength=520, justify="left").pack(
                anchor="w", pady=(10, 0))
        if code:
            box = ctk.CTkTextbox(body, width=560, height=220, font=ctk.CTkFont(family="Consolas", size=12),
                                 corner_radius=12, fg_color=SURFACE, border_width=1, border_color=BORDER)
            box.insert("1.0", code)
            box.configure(state="disabled")
            toggle = ctk.CTkButton(body, text="▶ 中身（コード）を見る", fg_color="transparent", text_color=ACCENT,
                                   hover_color=PLAN_BG, font=self.font(12), anchor="w", width=10)

            def show():
                if box.winfo_ismapped():
                    box.pack_forget()
                    toggle.configure(text="▶ 中身（コード）を見る")
                else:
                    box.pack(fill="x", pady=(4, 0), after=toggle)
                    toggle.configure(text="▼ 中身（コード）を隠す")
            toggle.configure(command=show)
            toggle.pack(anchor="w", pady=(8, 0))
            if not explanation:
                show()
        answer = {"value": False}
        self.buttons(win, "はい、お願いします", lambda: (answer.update(value=True), win.destroy()),
                     "やめる", win.destroy)
        self.root.wait_window(win)
        return answer["value"]

    def ask_user(self, question):
        return self.on_ui(self._ask_dialog, question) or ""

    def _ask_dialog(self, question):
        win = self.dialog("入力")
        self.heading(win, question, "分からなければ「スキップ」を押してください")
        row = ctk.CTkFrame(win, fg_color="transparent")
        row.pack(fill="x", padx=28, pady=(14, 0))
        entry = ctk.CTkEntry(row, width=340, height=42, corner_radius=12, font=self.font(15),
                             border_color=BORDER, fg_color=SURFACE, text_color=TEXT)
        entry.pack(side="left")
        win.after(200, entry.focus_set)
        if main.WHISPER is not None:
            self.voice_button(row, entry).pack(side="left", padx=(8, 0))
        answer = {"value": ""}

        def ok():
            answer["value"] = entry.get().strip()
            win.destroy()
        entry.bind("<Return>", lambda _: ok())
        self.buttons(win, "OK", ok, "スキップ", win.destroy)
        self.root.wait_window(win)
        return answer["value"]

    def voice_button(self, parent, entry):
        """押して話すと、聞き取った文字を entry に入れるボタン"""
        recorder = Recorder()

        def toggle():
            if recorder.stream is None:
                recorder.start()
                mic.configure(text="■ 止める", fg_color=DANGER, text_color="white")
                return
            audio = recorder.stop()
            mic.configure(text="聞き取り中…", fg_color=BG, text_color=TEXT, state="disabled")

            def work():
                text = transcribe(audio) if audio is not None else ""
                self.root.after(0, lambda: (entry.delete(0, "end"), entry.insert(0, text),
                                            mic.configure(text="🎤 声で入力", state="normal")))
            threading.Thread(target=work, daemon=True).start()
        mic = ctk.CTkButton(parent, text="🎤 声で入力", command=toggle, height=42, corner_radius=12,
                            font=self.font(13), fg_color=BG, text_color=TEXT, hover_color=BORDER)
        return mic

    def ask_message(self):
        """返信したいメッセージを貼り付けてもらう（コピー済みなら最初から入れておく）"""
        return self.on_ui(self._message_dialog) or ""

    def _message_dialog(self):
        win = self.dialog("返信したいメッセージ")
        self.heading(win, "返信したいメッセージを貼り付けてください",
                     "メールやチャットの文章をコピーして、下の欄で Ctrl+V を押します")
        box = ctk.CTkTextbox(win, width=560, height=220, corner_radius=12, font=self.font(14),
                             fg_color=SURFACE, border_width=1, border_color=BORDER, text_color=TEXT, wrap="word")
        box.pack(padx=28, pady=(14, 0))
        try:
            box.insert("1.0", self.root.clipboard_get())
        except tk.TclError:
            pass
        win.after(200, box.focus_set)
        answer = {"value": ""}
        self.buttons(win, "このメッセージに返信", lambda: (answer.update(value=box.get("1.0", "end").strip()),
                                                         win.destroy()), "やめる", win.destroy)
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
            self.heading(win, "はじめに、あなたのことを教えてください",
                         "返信の署名や、予定が重なったときの判断に使います。あとから変更できます。")
        else:
            self.heading(win, "自分の情報")
        form = ctk.CTkFrame(win, fg_color="transparent")
        form.pack(fill="x", padx=28, pady=(14, 0))
        entries = {}
        for i, (key, hint) in enumerate((("名前", "例: 山田 太郎"), ("所属", "例: 神戸電子専門学校 AI学科"),
                                         ("返信の署名", "例: 山田"))):
            ctk.CTkLabel(form, text=key, font=self.font(13), text_color=TEXT).grid(row=i, column=0, sticky="w", pady=6)
            entry = ctk.CTkEntry(form, width=360, height=38, corner_radius=10, font=self.font(14),
                                 placeholder_text=hint, border_color=BORDER, fg_color=SURFACE, text_color=TEXT)
            if field(key):
                entry.insert(0, field(key))
            entry.grid(row=i, column=1, sticky="w", padx=(14, 0), pady=6)
            entries[key] = entry
        ctk.CTkLabel(form, text="予定が重なった\nときの優先順位", font=self.font(13), text_color=TEXT,
                     justify="left").grid(row=3, column=0, sticky="nw", pady=6)
        prio = ctk.CTkTextbox(form, width=360, height=130, corner_radius=10, font=self.font(14),
                              fg_color=SURFACE, border_width=1, border_color=BORDER, text_color=TEXT)
        prio.insert("1.0", "\n".join(priorities or ["授業・試験", "発表会・コンテスト", "アルバイト", "友人との約束"]))
        prio.grid(row=3, column=1, sticky="w", padx=(14, 0), pady=6)
        ctk.CTkLabel(form, text="大事なものから順に、1行に1つ", font=self.font(11), text_color=MUTED).grid(
            row=4, column=1, sticky="w", padx=14)

        def save():
            items = [line.strip() for line in prio.get("1.0", "end").splitlines() if line.strip()]
            body = ["# 自分の情報", ""] + [f"- {k}: {e.get().strip()}" for k, e in entries.items() if e.get().strip()]
            body += ["", "## 予定が重なったときの優先順位"] + [f"{n}. {item}" for n, item in enumerate(items, 1)]
            with open(main.PROFILE_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(body) + "\n")
            win.destroy()
            print("✓ 自分の情報を保存しました。")
        self.buttons(win, "保存", save, "あとで" if first else "閉じる", win.destroy)

    def open_calendar(self):
        win = self.dialog("予定を読み込む")
        self.heading(win, "予定（カレンダー）を読み込む",
                     "Googleカレンダー: パソコンで「設定 → インポート/エクスポート → エクスポート」で .ics を保存\n"
                     "Outlook: 「予定表の保存」で .ics を保存")
        listing = ctk.CTkScrollableFrame(win, width=520, height=180, fg_color=SURFACE, corner_radius=14,
                                         border_width=1, border_color=BORDER)
        listing.pack(padx=28, pady=(14, 0))

        def refresh():
            for child in listing.winfo_children():
                child.destroy()
            files = sorted(n for n in os.listdir(main.CALENDAR_DIR) if n.endswith(".ics"))
            if not files:
                ctk.CTkLabel(listing, text="まだ読み込まれていません", font=self.font(13), text_color=MUTED).pack(pady=20)
            for name in files:
                row = ctk.CTkFrame(listing, fg_color="transparent")
                row.pack(fill="x", pady=2)
                ctk.CTkLabel(row, text=f"📅  {name}", font=self.font(13), text_color=TEXT).pack(side="left", padx=6)
                ctk.CTkButton(row, text="外す", width=60, height=28, corner_radius=14, font=self.font(12),
                              fg_color="transparent", text_color=DANGER, border_width=1, border_color=BORDER,
                              hover_color=ALERT_BG, command=lambda n=name: remove(n)).pack(side="right", padx=6)

        def add():
            paths = filedialog.askopenfilenames(parent=win, title="予定のファイルを選ぶ",
                                                filetypes=[("カレンダー", "*.ics")])
            for path in paths:
                shutil.copy(path, main.CALENDAR_DIR)
            if paths:
                print(f"✓ 予定のファイルを {len(paths)} 個読み込みました。")
            refresh()

        def remove(name):
            os.remove(os.path.join(main.CALENDAR_DIR, name))
            refresh()
        refresh()
        self.buttons(win, "ファイルを選ぶ", add, "閉じる", win.destroy)

    def open_tasks(self):
        win = self.dialog("お任せ一覧")
        self.heading(win, "お任せした作業")
        listing = ctk.CTkScrollableFrame(win, width=640, height=300, fg_color="transparent")
        listing.pack(padx=20, pady=(10, 0))

        def refresh():
            for child in listing.winfo_children():
                child.destroy()
            entries = main.load_delegated()
            if not entries:
                ctk.CTkLabel(listing, text="まだ何もお任せされていません。\n「毎朝9時に〜しといて」のように頼むと、ここに表示されます。",
                             font=self.font(13), text_color=MUTED).pack(pady=40)
                return
            loading = ctk.CTkLabel(listing, text="状況を確認しています…", font=self.font(13), text_color=MUTED)
            loading.pack(pady=20)

            def work():
                rows = [(e, main.task_status(e, os.path.exists(e["log"]))) for e in reversed(entries)]
                self.root.after(0, lambda: show(rows, loading))
            threading.Thread(target=work, daemon=True).start()

        def show(rows, loading):
            if not win.winfo_exists():
                return
            loading.destroy()
            for entry, status in rows:
                card = ctk.CTkFrame(listing, fg_color=SURFACE, corner_radius=14, border_width=1, border_color=BORDER)
                card.pack(fill="x", pady=4, padx=4)
                info = ctk.CTkFrame(card, fg_color="transparent")
                info.pack(side="left", fill="x", expand=True, padx=14, pady=10)
                ctk.CTkLabel(info, text=entry["task"], font=self.font(14, "bold"), text_color=TEXT,
                             wraplength=440, justify="left", anchor="w").pack(anchor="w")
                color = DANGER if status.startswith("エラー") else (TOOL_FG if status.startswith(("成功", "実行済み")) else MUTED)
                ctk.CTkLabel(info, text=f"🕒 {entry['when']}　　{status}", font=self.font(12), text_color=color,
                             wraplength=440, justify="left", anchor="w").pack(anchor="w", pady=(2, 0))
                ctk.CTkButton(card, text="取り消す", width=84, height=32, corner_radius=16, font=self.font(12),
                              fg_color="transparent", text_color=DANGER, border_width=1, border_color=BORDER,
                              hover_color=ALERT_BG, command=lambda e=entry: cancel(e)).pack(side="right", padx=12)

        def cancel(entry):
            if messagebox.askyesno("確認", f"「{entry['task']}」のお任せを取り消しますか？", parent=win):
                main.cancel_delegated(entry)
                print(f"✓ 「{entry['task']}」のお任せを取り消しました。")
                refresh()
        refresh()
        self.buttons(win, "閉じる", win.destroy, "更新", refresh)


if __name__ == "__main__":
    root = ctk.CTk()
    App(root)
    root.mainloop()
