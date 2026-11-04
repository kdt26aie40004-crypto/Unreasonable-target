"""ローカルの Ollama サーバーと通信する最小クライアント（外部ネットワークには出ない）。"""
import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"


class OllamaError(RuntimeError):
    pass


def chat(system: str, user: str, model: str = DEFAULT_MODEL, temperature: float = 0.3) -> str:
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as res:
            body = json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise OllamaError(f"Ollama がエラーを返しました: {e.code} {e.read().decode('utf-8', 'ignore')}") from e
    except urllib.error.URLError as e:
        raise OllamaError("Ollama に接続できません。`ollama serve` が起動しているか確認してください。") from e
    return body["message"]["content"]


def list_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as res:
            return [m["name"] for m in json.loads(res.read())["models"]]
    except (urllib.error.URLError, KeyError, ValueError):
        return []
