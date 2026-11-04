"""第二の自分 — 初心者向けの画面（「起動.bat」をダブルクリックで起動）

処理の中身は main.py の4段階構成（Whisper → Ollama → Python連携コア → PowerShell実行）をそのまま使い、
この画面は入力・確認・結果表示だけを担当する。
"""

import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext

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
except ImportError as e:
    tk.Tk().withdraw()
    messagebox.showerror("第二の自分", f"必要なパッケージが入っていません。\n\n"
                                      f"「pip install -r requirements.txt」を実行してください。\n\n({e})")
    sys.exit(1)

FONT = ("Yu Gothic UI", 12)
FONT_BIG = ("Yu Gothic UI", 18, "bold")
FONT_CODE = ("Consolas", 10)

EXAMPLES = [
    "任せてた作業どうだった？",
    "ダブルブッキングしてないか見て",
    "毎朝9時にディスクの空き容量を記録しといて",
    "コピーしたメッセージに参加できると返事して",
]

INTENT_LABELS = {"tool": "ツールを作って実行", "reply": "返信の下書き", "schedule": "予定の調整",
                 "delegate": "お任せ（休んでいる間に実行）", "report": "お任せした作業の報告"}


def has_microphone():
    try:
        sd.query_devices(kind="input")
        return True
    except Exception:
        return False


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
        self.q.put(text)

    def flush(self):
        pass


class App:
    def __init__(self, root):
        self.root = root
        self.log_queue = queue.Queue()
        self.recorder = Recorder()
        self.mic = has_microphone()
        self.busy = False

        root.title("第二の自分")
        scale = root.winfo_fpixels("1i") / 96  # 画面の拡大率に合わせてウィンドウの大きさを決める
        root.geometry(f"{int(820 * scale)}x{int(680 * scale)}")
        root.minsize(int(560 * scale), int(480 * scale))
        root.configure(bg="#f5f6f8")

        tk.Label(root, text="第二の自分", font=FONT_BIG, bg="#f5f6f8").pack(pady=(14, 0))
        tk.Label(root, text="「〜しといて」と頼むと、休んでいる間に代わりにこなします",
                 font=FONT, bg="#f5f6f8", fg="#555").pack()

        # 下の部品から先に配置して、ウィンドウが小さくても入力欄が隠れないようにする
        self.status = tk.Label(root, text="", font=("Yu Gothic UI", 10), bg="#f5f6f8", fg="#555", anchor="w")
        self.status.pack(side="bottom", fill="x", padx=16, pady=(0, 10))

        bar = tk.Frame(root, bg="#f5f6f8")
        bar.pack(side="bottom", fill="x", padx=16, pady=(4, 6))

        chips = tk.Frame(root, bg="#f5f6f8")
        chips.pack(side="bottom", fill="x", padx=16)
        tk.Label(chips, text="例:", font=FONT, bg="#f5f6f8").grid(row=0, column=0, rowspan=2, sticky="n")
        for i, text in enumerate(EXAMPLES):
            tk.Button(chips, text=text, font=("Yu Gothic UI", 10), relief="groove", bg="white",
                      command=lambda t=text: self.fill(t)).grid(row=i // 2, column=1 + i % 2,
                                                                 sticky="w", padx=3, pady=2)

        self.log = scrolledtext.ScrolledText(root, font=FONT, wrap="word", state="disabled",
                                             relief="flat", padx=12, pady=10)
        self.log.pack(fill="both", expand=True, padx=16, pady=10)
        self.log.tag_config("you", foreground="#1a5fb4", font=(FONT[0], FONT[1], "bold"))
        self.log.tag_config("note", foreground="#888")

        self.entry = tk.Entry(bar, font=FONT, relief="solid", bd=1)
        self.entry.pack(side="left", fill="x", expand=True, ipady=6)
        self.entry.bind("<Return>", lambda _: self.send())
        self.send_btn = tk.Button(bar, text="送信", font=FONT, width=6, bg="#1a5fb4", fg="white",
                                  relief="flat", command=self.send)
        self.send_btn.pack(side="left", padx=(6, 0))
        self.mic_btn = tk.Button(bar, text="🎤 話す", font=FONT, width=8, relief="flat",
                                 bg="#e0e0e0", command=self.toggle_voice, state="disabled")
        self.mic_btn.pack(side="left", padx=(6, 0))

        sys.stdout = sys.stderr = QueueWriter(self.log_queue)
        main.confirm = self.confirm
        main.ask_user = self.ask_user
        self.poll_log()
        self.startup()

    # --- 起動時のチェック ---
    def startup(self):
        self.say("こんにちは。下の欄に頼みたいことを書いて「送信」を押してください。\n"
                 "例のボタンを押すと、入力欄に例文が入ります。\n")
        try:
            requests.get(f"{main.OLLAMA_BASE}/api/version", timeout=3)
        except requests.RequestException:
            self.say("⚠ Ollama が起動していません。スタートメニューから「Ollama」を起動してから使ってください。\n", "note")
        if not self.mic:
            self.set_status("マイクが見つからないため、文字入力で使えます（マイクをつなぐと🎤が使えます）")
            return
        self.set_status("音声認識を準備中...（文字入力はすぐ使えます）")
        threading.Thread(target=self.load_whisper, daemon=True).start()

    def load_whisper(self):
        main.WHISPER = main.WhisperModel(main.WHISPER_MODEL, device="cpu", compute_type="int8")
        self.root.after(0, lambda: (self.mic_btn.config(state="normal"), self.set_status("準備できました")))

    # --- 表示 ---
    def say(self, text, tag=None):
        self.log.config(state="normal")
        self.log.insert("end", text, tag)
        self.log.see("end")
        self.log.config(state="disabled")

    def poll_log(self):
        while not self.log_queue.empty():
            self.say(self.log_queue.get())
        self.root.after(100, self.poll_log)

    def set_status(self, text):
        self.status.config(text=text)

    def fill(self, text):
        self.entry.delete(0, "end")
        self.entry.insert(0, text)
        self.entry.focus_set()

    def set_busy(self, busy):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.send_btn.config(state=state)
        if main.WHISPER is not None:
            self.mic_btn.config(state=state)
        self.set_status("考え中です。少しお待ちください..." if busy else "準備できました")

    # --- 入力 ---
    def send(self):
        text = self.entry.get().strip()
        if not text or self.busy:
            return
        self.entry.delete(0, "end")
        self.run_request(text)

    def toggle_voice(self):
        if self.recorder.stream is None:
            self.recorder.start()
            self.mic_btn.config(text="■ 止める", bg="#e01b24", fg="white")
            self.send_btn.config(state="disabled")
            self.set_status("録音中です。話し終えたら「止める」を押してください")
            return
        audio = self.recorder.stop()
        self.mic_btn.config(text="🎤 話す", bg="#e0e0e0", fg="black")
        if audio is None:
            self.send_btn.config(state="normal")
            return
        self.set_busy(True)

        def work():
            text = main.transcribe(main.WHISPER, audio)
            if text:
                self.root.after(0, lambda: self.run_request(text, transcribed=True))
            else:
                self.root.after(0, lambda: self.set_busy(False))
        threading.Thread(target=work, daemon=True).start()

    def run_request(self, text, transcribed=False):
        if not transcribed:
            self.say(f"\nあなた: {text}\n", "you")
        self.set_busy(True)

        def work():
            try:
                info = main.classify(text)
                print(f"[AI] 「{INTENT_LABELS[info['intent']]}」として対応します。\n")
                main.HANDLERS[info["intent"]](text, info)
            except requests.RequestException:
                print("⚠ Ollama に接続できません。スタートメニューから「Ollama」を起動してください。\n")
            except Exception as e:
                print(f"⚠ うまくいきませんでした: {e}\n")
            finally:
                self.root.after(0, lambda: self.set_busy(False))
        threading.Thread(target=work, daemon=True).start()

    # --- main.py から呼ばれる確認・質問（作業スレッドから画面スレッドに頼む） ---
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
        return bool(self.on_ui(self._confirm_dialog, question, code))

    def _confirm_dialog(self, question, code):
        win = self.dialog("確認")
        tk.Label(win, text=question, font=FONT_BIG).pack(padx=20, pady=(16, 6))
        if code and main.DANGEROUS.search(code):
            tk.Label(win, text="⚠ ファイルの削除やPCの設定変更が含まれています。内容をよく確認してください。",
                     font=FONT, fg="#e01b24", wraplength=560).pack(padx=20)
        if code:
            tk.Label(win, text="実行する内容:", font=FONT, anchor="w").pack(fill="x", padx=20, pady=(8, 0))
            box = scrolledtext.ScrolledText(win, font=FONT_CODE, height=12, width=72)
            box.insert("1.0", code)
            box.config(state="disabled")
            box.pack(padx=20, pady=4)
        answer = {"value": False}
        buttons = tk.Frame(win)
        buttons.pack(pady=14)
        tk.Button(buttons, text="はい、お願いします", font=FONT, bg="#1a5fb4", fg="white", relief="flat",
                  command=lambda: (answer.update(value=True), win.destroy())).pack(side="left", padx=6)
        tk.Button(buttons, text="やめる", font=FONT, relief="flat", bg="#e0e0e0",
                  command=win.destroy).pack(side="left", padx=6)
        self.root.wait_window(win)
        return answer["value"]

    def ask_user(self, question):
        return self.on_ui(self._ask_dialog, question) or ""

    def _ask_dialog(self, question):
        win = self.dialog("入力")
        tk.Label(win, text=question, font=FONT_BIG, wraplength=520).pack(padx=20, pady=(16, 4))
        tk.Label(win, text="分からなければ「スキップ」を押してください", font=FONT, fg="#555").pack()
        row = tk.Frame(win)
        row.pack(padx=20, pady=10)
        entry = tk.Entry(row, font=FONT, width=28, relief="solid", bd=1)
        entry.pack(side="left", ipady=4)
        entry.focus_set()
        answer = {"value": ""}

        if main.WHISPER is not None:
            recorder = Recorder()

            def toggle():
                if recorder.stream is None:
                    recorder.start()
                    mic.config(text="■ 止める", bg="#e01b24", fg="white")
                    return
                audio = recorder.stop()
                mic.config(text="聞き取り中...", bg="#e0e0e0", fg="black", state="disabled")

                def work():
                    text = (main.transcribe(main.WHISPER, audio, label="聞き取った内容") or "") if audio is not None else ""
                    self.root.after(0, lambda: (entry.delete(0, "end"), entry.insert(0, text.rstrip("。、 ")),
                                                mic.config(text="🎤 声で入力", state="normal")))
                threading.Thread(target=work, daemon=True).start()
            mic = tk.Button(row, text="🎤 声で入力", font=FONT, relief="flat", bg="#e0e0e0", command=toggle)
            mic.pack(side="left", padx=(6, 0))

        def ok():
            answer["value"] = entry.get().strip()
            win.destroy()
        entry.bind("<Return>", lambda _: ok())
        buttons = tk.Frame(win)
        buttons.pack(pady=(0, 14))
        tk.Button(buttons, text="OK", font=FONT, width=8, bg="#1a5fb4", fg="white", relief="flat",
                  command=ok).pack(side="left", padx=6)
        tk.Button(buttons, text="スキップ", font=FONT, relief="flat", bg="#e0e0e0",
                  command=win.destroy).pack(side="left", padx=6)
        self.root.wait_window(win)
        return answer["value"]

    def dialog(self, title):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.transient(self.root)
        win.grab_set()
        win.resizable(False, False)
        return win


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
