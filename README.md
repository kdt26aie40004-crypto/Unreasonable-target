# 第二の自分 — 声で頼むと、休んでいる間に業務をこなす

声で「これをしといて」と頼むと、ローカルLLMが作業を引き受け、本人が休んでいる間に実行します。
外部APIには一切データを送信せず、すべてローカルで完結します。構想は [docs/system_architecture_concept.md](docs/system_architecture_concept.md) を参照。

```
[音声] → 1. Whisper（文字起こし） → 2. Ollama（要望の振り分け + 生成）
       → 3. Python（抽出・BOM付きUTF-8で保存） → 4. PowerShell（実行・予約・下書きを開く）
```

## できること

| 話しかけ方の例 | 機能 |
|---|---|
| 「デスクトップの画像を整理するツールを作って」 | ツールを作って、その場で実行 |
| 「休んでいる間にダウンロードフォルダを仕分けしといて」 | **お任せ**：今すぐバックグラウンドで実行 |
| 「1時間後にPCの稼働時間をメモしといて」 | **お任せ**：指定時刻に1回実行（タスクスケジューラ） |
| 「毎朝9時にディスクの空き容量を記録しといて」 | **お任せ**：毎日・毎週実行（タスクスケジューラ） |
| 「任せてた作業どうだった？」 | お任せした作業の結果を報告 |
| 「来週ダブルブッキングしてないか見て」 | 予定の重複を検出し、調整案と連絡文の下書きを作る |
| 「このメールに参加できると返事して」 | クリップボードのメッセージに、自分の文体で返信の下書きを作る（送信はしない） |

お任せの確認は依頼したときに1回だけ。その後は本人がいなくても動き、結果は `my_powershell_tools/*.log` に残ります。

## セットアップ

1. [Ollama](https://ollama.com/) をインストールし、モデルを取得
   ```
   ollama pull qwen2.5-coder:7b
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ```
   `qwen2.5:7b` は返信・調整案などの文章用です（無い場合は `qwen2.5-coder:7b` で代用しますが、日本語の質が落ちます）。
2. Python の依存パッケージをインストール
   ```
   pip install -r requirements.txt
   ```
3. パーソナルデータを `data/` に置く（`data/` はGitHubに上げない設定です）
   - `data/profile.md` — 自分の情報・予定の優先順位（[examples/profile.example.md](examples/profile.example.md) をコピー）
   - `data/calendar/*.ics` — Google カレンダーや Outlook から書き出した予定
   - `data/my_messages/*.txt` — 自分が過去に書いたメールやチャット（返信の文体の参考）

## 使い方

**`起動.bat` をダブルクリック**すると画面が開きます。

1. 下の入力欄に頼みたいことを書いて「送信」（マイクがあれば「🎤 話す」で声でも頼めます）
2. 実行やお任せの前には確認画面が出るので、内容を見て「はい、お願いします」か「やめる」を選ぶ
3. 宛先などを聞かれたら、文字で入力するか「🎤 声で入力」で答える

「例」のボタンを押すと、入力欄に例文が入ります。

ターミナルで使う場合:
```
python main.py          # 音声で話しかける
python main.py --text   # マイクが無いときはキーボードで入力
```

お任せで登録したタスクの確認・削除（PowerShell）:
```
Get-ScheduledTask -TaskName "SecondSelf_*"
Unregister-ScheduledTask -TaskName "SecondSelf_<名前>"
```

`main.py` 冒頭の設定:
- `MODEL_NAME` / `TEXT_MODEL` — コード用 / 文章用のローカルLLM
- `WHISPER_MODEL` — 文字起こしの精度と速度（`small` が標準）
- `CONFIRM_BEFORE_RUN` — 実行・お任せの前に確認するか
- `WORK_HOURS` — 予定の調整案で、空き時間として扱う時間帯
