"""赤ペン先生デスクを配信し、ローカルの Ollama へ取り次ぎ、音声を Whisper で文字にする小さなサーバー。

127.0.0.1 にだけ待ち受けるので、ほかの PC からは接続できない。
音声認識だけ faster-whisper が必要（py -m pip install faster-whisper）。なくても他の機能は動く。
起動: py server.py （または start.bat をダブルクリック）
"""
import http.server
import io
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

HOST, PORT = "127.0.0.1", 8765
OLLAMA_URL = "http://127.0.0.1:11434"
WEB_DIR = Path(__file__).parent / "web"
PROXY_PATHS = {"GET": {"/api/tags"}, "POST": {"/api/chat"}}
# 学校の事務用 PC を想定し、既定は CPU で動く small モデルを int8 に量子化して使う
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")

_whisper = None
_whisper_lock = threading.Lock()


def get_whisper():
    """初回だけモデルを読み込む（初回はモデルのダウンロードがある）。"""
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            from faster_whisper import WhisperModel

            compute = "int8" if WHISPER_DEVICE == "cpu" else "float16"
            _whisper = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=compute)
        return _whisper


def decode_audio(audio: bytes):
    """録音（webm・wav など）を 16kHz・モノラルの波形にする。メモリ上だけで扱い、ファイルには保存しない。

    faster-whisper 付属の読み込み処理は新しい PyAV で動かないため、ここで PyAV を直接使う。
    """
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    with av.open(io.BytesIO(audio)) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def transcribe(audio: bytes, hint: str) -> str:
    segments, _ = get_whisper().transcribe(
        decode_audio(audio), language="ja", initial_prompt=hint or None, vad_filter=True
    )
    return "".join(s.text for s in segments).strip()


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _proxy(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        req = urllib.request.Request(
            OLLAMA_URL + self.path, data=body, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as res:
                data, status = res.read(), res.status
        except urllib.error.HTTPError as e:
            data, status = e.read(), e.code
        except urllib.error.URLError:
            msg = "Ollama に接続できません。Ollama が起動しているか確認してください。"
            data, status = json.dumps({"error": msg}, ensure_ascii=False).encode("utf-8"), 502
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _transcribe(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= 50 * 1024 * 1024:
            return self._json(400, {"error": "音声が空か、大きすぎます（50MBまで）。"})
        audio = self.rfile.read(length)
        hint = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("hint", [""])[0][:500]
        try:
            text = transcribe(audio, hint)
        except ImportError:
            return self._json(501, {"error": "音声認識が入っていません。py -m pip install faster-whisper を実行してください。"})
        except Exception as e:  # 音声の形式やモデル読み込みの失敗を画面に伝える
            return self._json(500, {"error": f"音声を文字にできませんでした：{e}"})
        self._json(200, {"text": text})

    def do_GET(self):
        if self.path in PROXY_PATHS["GET"]:
            return self._proxy("GET")
        if self.path == "/api/voice":
            try:
                import faster_whisper  # noqa: F401
                return self._json(200, {"ok": True, "model": WHISPER_MODEL, "device": WHISPER_DEVICE})
            except ImportError:
                return self._json(200, {"ok": False})
        return super().do_GET()

    def do_POST(self):
        if self.path in PROXY_PATHS["POST"]:
            return self._proxy("POST")
        if urllib.parse.urlsplit(self.path).path == "/api/transcribe":
            return self._transcribe()
        self.send_error(404)

    def log_message(self, format, *args):
        pass


def main() -> None:
    server = http.server.ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}/"
    print(f"赤ペン先生デスクを起動しました: {url}")
    print("終了するときは、この画面で Ctrl+C を押してください。")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
