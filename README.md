# voice-to-rag

音声メモ → RAG 自動取り込みパイプライン。

PLAUD 等で録音した音声メモを、ローカルで文字起こし → 要約・整形 → RAG ナレッジベースに自動登録します。
全てローカル完結（faster-whisper + Ollama）、課金ゼロ。

## パイプライン

```
音声ファイル (.wav/.m4a/.mp3/.flac)
  → faster-whisper で文字起こし（ローカル）
  → Ollama で要約・タグ付け・整形（ローカル）
  → RAG/drafts/ に Markdown 保存
  → ask.py --rebuild-cache で検索可能に
```

## セットアップ

### 前提条件

- Python 3.9+
- ffmpeg（m4a/mp3 デコードに必要）
- Ollama（要約機能を使う場合）

```bash
brew install ffmpeg
```

### インストール

```bash
cd voice-to-rag
pip3 install -r requirements.txt
cp config.yml.example config.yml
# config.yml を環境に合わせて編集
```

## 使い方

```bash
# 1ファイル処理
python3 ingest.py audio.wav

# 複数ファイル処理
python3 ingest.py *.wav

# inbox 監視モード（音声ファイルを自動検出、Ctrl+C で停止）
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
| whisper.model | base | Whisper モデルサイズ (tiny/base/small/medium/large) |
| whisper.language | ja | 文字起こし言語 |
| ollama.model | gemma3 | 要約に使う LLM モデル |
| output.dir | ./output | Markdown 出力先 |
| output.prefix | voice | ファイル名プレフィックス |
| watch.dir | ./watch | 監視ディレクトリ |

## Ollama フォールバック

Ollama が未起動・未インストールの場合、要約なしで生テキストのみ Markdown に保存します。
後から Ollama を起動して再処理することも可能です。

## テスト

```bash
python3 -m pytest tests/ -v
```

## ライセンス

MIT
