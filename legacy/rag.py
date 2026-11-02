"""学校ルール文書の簡易 RAG。

低スペック PC でも動くよう埋め込みモデルは使わず、
日本語向けの文字 bi-gram の重なりで関連チャンクを検索する。
"""
import re
from collections import Counter


def split_chunks(text: str, max_chars: int = 200) -> list[str]:
    # 空行、または行頭の見出し（【…】）で段落に分ける
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n(?=【)", text) if p.strip()]
    chunks, buf = [], ""
    for p in paragraphs:
        if buf and len(buf) + len(p) > max_chars:
            chunks.append(buf)
            buf = ""
        buf = f"{buf}\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks


def _bigrams(text: str) -> Counter:
    s = re.sub(r"\s+", "", text)
    return Counter(s[i : i + 2] for i in range(len(s) - 1))


def retrieve(query: str, chunks: list[str], k: int = 3) -> list[str]:
    q = _bigrams(query)
    scored = []
    for c in chunks:
        overlap = sum((q & _bigrams(c)).values())
        scored.append((overlap, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for score, c in scored[:k] if score > 0]
