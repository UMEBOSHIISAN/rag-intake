# rag-intake

汎用 RAG 取り込みパイプライン。

音声メモ・テキスト・PDF・Web ページを、ローカルで要約・整形して RAG ナレッジベースに自動登録します。
全てローカル完結（faster-whisper + Ollama）、課金ゼロ。

## パイプライン

```
入力ソース
├── 音声 (.wav/.m4a/.mp3/.flac) → faster-whisper で文字起こし
├── テキスト (.txt/.md)         → そのまま読み込み
├── PDF (.pdf)                  → pymupdf でテキスト抽出
└── URL (http/https)            → trafilatura で本文抽出
    ↓
Ollama で要約・タグ付け・整形（共通）
    ↓
出力ディレクトリに Markdown 保存（共通）
    ↓
ask.py --rebuild-cache で検索可能に
```

## セットアップ

### 前提条件

- Python 3.9+
- ffmpeg（音声デコードに必要）
- Ollama（要約機能を使う場合）

```bash
brew install ffmpeg
```

### インストール

```bash
cd rag-intake
pip3 install -r requirements.txt
cp config.yml.example config.yml
# config.yml を環境に合わせて編集
```

## 使い方

```bash
# 音声ファイル
python3 ingest.py audio.wav

# テキストファイル
python3 ingest.py memo.txt note.md

# PDF
python3 ingest.py document.pdf

# Web ページ（URL）
python3 ingest.py https://example.com/article

# 混在OK
python3 ingest.py *.wav *.txt *.pdf https://example.com

# 監視モード（ファイルを自動検出、Ctrl+C で停止）
python3 ingest.py --watch

# 監視ディレクトリを指定
python3 ingest.py --watch --watch-dir ~/Downloads

# 処理済みファイル一覧
python3 ingest.py --list
```

## 設定

`config.yml` で各種設定を変更できます（`config.yml.example` を参照）。

| セクション | キー | デフォルト | 説明 |
|------------|------|-----------|------|
| whisper.model | | base | Whisper モデルサイズ (tiny/base/small/medium/large) |
| whisper.language | | ja | 文字起こし言語 |
| ollama.model | | gemma3 | 要約に使う LLM モデル |
| output.dir | | ./output | Markdown 出力先 |
| output.prefix | | voice | ファイル名プレフィックス（音声用デフォルト） |
| text.extensions | | [".txt", ".md"] | テキストとして扱う拡張子 |
| pdf.max_pages | | 50 | PDF 抽出の最大ページ数 |
| clip.timeout | | 30 | URL フェッチのタイムアウト（秒） |
| watch.dir | | ./watch | 監視ディレクトリ |
| watch.extensions | | (音声+txt+md+pdf) | 監視対象の拡張子 |

## 出力ファイル名

ソースタイプに応じてプレフィックスが自動設定されます:

| ソース | プレフィックス | 例 |
|--------|---------------|-----|
| 音声 | `voice` | `2026-02-11_voice_会議メモ.md` |
| テキスト | `text` | `2026-02-11_text_買い物リスト.md` |
| PDF | `pdf` | `2026-02-11_pdf_論文要約.md` |
| URL | `clip` | `2026-02-11_clip_技術記事.md` |

## Ollama フォールバック

Ollama が未起動・未インストールの場合、要約なしで生テキストのみ Markdown に保存します。
後から Ollama を起動して再処理することも可能です。

## テスト

```bash
python3 -m pytest tests/ -v
```

## ライセンス

MIT
