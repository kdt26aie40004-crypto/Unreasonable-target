"""教務事務総合DXコックピット（MVP）

起動: streamlit run app.py
"""
from pathlib import Path

import streamlit as st

import ollama_client
import rag

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="先生のデジタル分身", page_icon="📝", layout="wide")


def read_upload(file) -> str:
    raw = file.getvalue()
    for enc in ("utf-8", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "ignore")


def run_llm(system: str, user: str) -> str | None:
    try:
        return ollama_client.chat(system, user, model=st.session_state.model)
    except ollama_client.OllamaError as e:
        st.error(str(e))
        return None


# ---------- サイドバー：設定・学校ルール・先生の文体 ----------
with st.sidebar:
    st.header("⚙️ 設定")
    models = ollama_client.list_models()
    if models:
        default = models.index(ollama_client.DEFAULT_MODEL) if ollama_client.DEFAULT_MODEL in models else 0
        st.session_state.model = st.selectbox("モデル", models, index=default)
        st.success("Ollama 接続OK（ローカル処理）")
    else:
        st.session_state.model = st.text_input("モデル", ollama_client.DEFAULT_MODEL)
        st.warning("Ollama に接続できません。`ollama serve` を起動してください。")

    st.subheader("📘 学校ルール（RAG）")
    rules_file = st.file_uploader("学生便覧・審査マニュアル（.txt）", type=["txt"], key="rules")
    rules_text = read_upload(rules_file) if rules_file else (DATA_DIR / "rules_sample.txt").read_text("utf-8")
    rules_text = st.text_area("ルール本文", rules_text, height=200)

    st.subheader("✉️ 先生の文体（パーソナルデータ）")
    style_text = st.text_area(
        "普段のメール例文", (DATA_DIR / "style_sample.txt").read_text("utf-8"), height=200
    )

rule_chunks = rag.split_chunks(rules_text)

st.title("📝 先生のデジタル分身 — 教務事務DXコックピット")
st.caption("すべての処理はこのPC内のローカルLLM（Ollama）で完結し、学生の個人情報は外部に送信されません。")

tab_check, tab_sort, tab_notice = st.tabs(
    ["① 書類不備チェック＆返信メール", "② 一括仕分け＆3行要約", "④ 案内文・公式文書作成"]
)

# ---------- ① 書類不備チェック ＆ 返信メール ----------
with tab_check:
    doc_type = st.selectbox("書類の種類", ["日報", "志望理由書", "欠席届", "その他"])
    student = st.text_input("学生の氏名", placeholder="例：山田 太郎")
    doc_text = st.text_area("提出された書類の本文", height=220)

    if st.button("チェックする", type="primary", disabled=not doc_text.strip()):
        related = rag.retrieve(f"{doc_type} {doc_text}", rule_chunks, k=3) or rule_chunks[:3]
        rules_ctx = "\n\n".join(related)
        with st.expander("参照した学校ルール"):
            st.text(rules_ctx)

        left, right = st.columns(2)
        with left:
            st.subheader("不備の指摘")
            with st.spinner("ルールと照合中..."):
                issues = run_llm(
                    "あなたは学校の教務担当者です。提示された学校ルールだけを根拠に、提出書類の不備を日本語で箇条書きにしてください。"
                    "各項目に根拠となるルールを括弧で添えてください。ルールに書かれていないことで減点しないでください。"
                    "不備がなければ「不備はありません」とだけ答えてください。",
                    f"# 学校ルール\n{rules_ctx}\n\n# 書類の種類\n{doc_type}\n\n# 提出書類\n{doc_text}",
                )
            if issues:
                st.markdown(issues)
        with right:
            st.subheader("返信メールの下書き")
            if issues:
                with st.spinner("先生の文体で作成中..."):
                    mail = run_llm(
                        "あなたは教務の先生本人として、学生への返信メールを書きます。"
                        "以下の『先生の文体例』の口調・挨拶・締め方をそのまま真似てください。メール本文だけを出力してください。",
                        f"# 先生の文体例\n{style_text}\n\n# 宛先の学生\n{student or '学生'}さん\n\n"
                        f"# 書類の種類\n{doc_type}\n\n# 伝えるべき不備\n{issues}",
                    )
                if mail:
                    st.text_area("下書き（編集してコピーできます）", mail, height=320)

# ---------- ② 一括仕分け ＆ 3行要約 ----------
with tab_sort:
    files = st.file_uploader("提出ファイルをまとめてドロップ（.txt）", type=["txt"], accept_multiple_files=True)
    if st.button("仕分け・要約する", type="primary", disabled=not files):
        progress = st.progress(0.0)
        for i, f in enumerate(files, 1):
            text = read_upload(f)
            result = run_llm(
                "あなたは教務アシスタントです。提出書類を読み、次の形式だけで答えてください。\n"
                "提出者: （本文から分かる氏名。不明なら「不明」）\n"
                "種類: （日報/志望理由書/欠席届/その他）\n"
                "要約:\n・\n・\n・\n"
                "要対応: （至急の確認が必要なら「あり：理由」、なければ「なし」）",
                text,
            )
            if result is None:
                break
            urgent = "要対応: あり" in result or "要対応：あり" in result
            with st.container(border=True):
                st.markdown(f"**{'🚩 ' if urgent else ''}{f.name}**")
                st.text(result)
            progress.progress(i / len(files))

# ---------- ④ 案内文・公式文書作成 ----------
with tab_notice:
    memo = st.text_area("メモ（キーワードでOK）", placeholder="例：来週月曜1限、遅刻厳禁、持ち物は筆記用具と学生証", height=120)
    fmt = st.radio("出力形式", ["学校公式の通知文", "学生ポータル・掲示板向け"], horizontal=True)
    if st.button("文章を作成", type="primary", disabled=not memo.strip()):
        style = (
            "保護者や学生に配布する学校の公式通知文として、表題・日付欄・本文・記・以上 の形式で、格調ある丁寧な文体で書いてください。"
            if fmt == "学校公式の通知文"
            else "学生用ポータルにそのまま貼れる、簡潔で分かりやすい案内文を書いてください。重要事項は【】で強調してください。"
        )
        with st.spinner("作成中..."):
            notice = run_llm(
                f"あなたは学校の教務担当者です。先生のメモを元に文章を作成します。{style}"
                "メモにない日時や場所を勝手に補わず、必要なら「（要記入）」と書いてください。",
                memo,
            )
        if notice:
            st.text_area("作成結果", notice, height=360)
