#!/usr/bin/env python3
"""voice-to-rag: 音声メモ → RAG 自動取り込みパイプライン"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from datetime import date
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# ログ設定
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# デフォルト設定
# ---------------------------------------------------------------------------
DEFAULTS = {
    "whisper": {
        "model": "base",
        "device": "cpu",
        "compute_type": "int8",
        "language": "ja",
    },
    "ollama": {
        "model": "gemma3",
        "host": "http://localhost:11434",
    },
    "output": {
        "dir": "./output",
        "prefix": "voice",
    },
    "watch": {
        "dir": "./watch",
        "extensions": [".wav", ".m4a", ".mp3", ".flac"],
        "processed_log": "~/.voice-to-rag/processed.json",
    },
}

AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3", ".flac"}


# ---------------------------------------------------------------------------
# 設定読み込み
# ---------------------------------------------------------------------------
def load_config(config_path: Path | None = None) -> dict:
    """config.yml を読み込み、デフォルト値とマージして返す"""
    cfg = _deep_copy_dict(DEFAULTS)
    if config_path and config_path.exists():
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        _deep_merge(cfg, user_cfg)
    return cfg


def _deep_copy_dict(d: dict) -> dict:
    """dict をディープコピー（リストも複製）"""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            result[k] = v[:]
        else:
            result[k] = v
    return result


def _deep_merge(base: dict, override: dict) -> None:
    """override の値で base を上書き（ネスト対応）"""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def get(cfg: dict, section: str, key: str, default=None):
    """設定値を取得"""
    return cfg.get(section, {}).get(key, default)


# ---------------------------------------------------------------------------
# 文字起こし（faster-whisper）
# ---------------------------------------------------------------------------
def transcribe(audio_path: Path, cfg: dict) -> str:
    """音声ファイルを faster-whisper で文字起こし"""
    from faster_whisper import WhisperModel

    model_size = get(cfg, "whisper", "model", "base")
    device = get(cfg, "whisper", "device", "cpu")
    compute_type = get(cfg, "whisper", "compute_type", "int8")
    language = get(cfg, "whisper", "language", "ja")

    logger.info("faster-whisper モデルロード: %s (%s/%s)", model_size, device, compute_type)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    logger.info("文字起こし開始: %s", audio_path.name)
    t0 = time.monotonic()
    segments, info = model.transcribe(str(audio_path), language=language)

    text_parts = []
    for segment in segments:
        text_parts.append(segment.text)

    result = "".join(text_parts).strip()
    elapsed = time.monotonic() - t0
    logger.info("文字起こし完了: %.1f秒, テキスト長=%d", elapsed, len(result))
    return result


# ---------------------------------------------------------------------------
# 要約・整形（Ollama）
# ---------------------------------------------------------------------------
SUMMARIZE_PROMPT = """\
以下は音声メモの文字起こしテキストです。
これを構造化された Markdown ドキュメントに整形してください。

## 出力フォーマット（厳守）

```
# タイトル（内容を端的に表す短いタイトル、英数字とアンダースコアのみ）

## 要約
（3〜5文で内容を要約）

## キーポイント
- ポイント1
- ポイント2
- ...

## 全文
（元のテキストを読みやすく段落分けしたもの）
```

重要:
- タイトルは必ず1行目に `# ` で始めてください
- タイトルはファイル名に使うので、日本語OK・スペースはアンダースコアに置換
- 内容の追加や創作はしないでください

---

文字起こしテキスト:
{text}
"""


def summarize(text: str, cfg: dict) -> dict | None:
    """Ollama で文字起こしテキストを要約・整形。失敗時は None を返す"""
    try:
        from ollama import chat
    except ImportError:
        logger.warning("ollama パッケージ未インストール — 生テキストで保存します")
        return None

    model = get(cfg, "ollama", "model", "gemma3")
    prompt = SUMMARIZE_PROMPT.format(text=text)

    try:
        logger.info("Ollama で要約中 (モデル: %s)...", model)
        response = chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response["message"]["content"]
        return parse_summary(content)
    except Exception as e:
        logger.warning("Ollama 接続失敗: %s — 生テキストで保存します", e)
        return None


def parse_summary(content: str) -> dict:
    """Ollama の出力からタイトルと本文を抽出"""
    lines = content.strip().split("\n")
    title = "untitled"
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break
    # タイトルをファイル名安全な形式に
    safe_title = sanitize_title(title)
    return {"title": safe_title, "body": content}


def sanitize_title(title: str) -> str:
    """タイトルをファイル名に安全な形式に変換"""
    # スペースをアンダースコアに
    title = title.replace(" ", "_").replace("　", "_")
    # ファイル名に使えない文字を除去
    title = re.sub(r'[\\/:*?"<>|]', "", title)
    # 長すぎる場合は切り詰め
    if len(title) > 60:
        title = title[:60]
    return title or "untitled"


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------
def save_markdown(text: str, summary: dict | None, cfg: dict) -> Path:
    """Markdown ファイルを出力ディレクトリに保存"""
    output_dir = Path(get(cfg, "output", "dir", "~/Workspace/RAG/drafts")).expanduser()
    prefix = get(cfg, "output", "prefix", "voice")
    today = date.today().isoformat()

    if summary:
        title = summary["title"]
        body = summary["body"]
    else:
        title = "memo"
        body = f"# 音声メモ\n\n## 全文\n\n{text}\n"

    filename = f"{today}_{prefix}_{title}.md"
    output_path = output_dir / filename

    # 同名ファイルが存在する場合はサフィックスを追加
    counter = 1
    while output_path.exists():
        counter += 1
        filename = f"{today}_{prefix}_{title}_{counter}.md"
        output_path = output_dir / filename

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body, encoding="utf-8")
    logger.info("保存完了: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 処理済みログ
# ---------------------------------------------------------------------------
def load_processed_log(cfg: dict) -> dict:
    """処理済みファイルのログを読み込む"""
    log_path = Path(get(cfg, "watch", "processed_log", "~/.voice-to-rag/processed.json")).expanduser()
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return {}


def save_processed_log(log: dict, cfg: dict) -> None:
    """処理済みログを保存"""
    log_path = Path(get(cfg, "watch", "processed_log", "~/.voice-to-rag/processed.json")).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def file_hash(path: Path) -> str:
    """ファイルの SHA-256 ハッシュを計算"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_processed(audio_path: Path, log: dict) -> bool:
    """ファイルが処理済みかチェック"""
    key = str(audio_path.resolve())
    if key not in log:
        return False
    return log[key]["hash"] == file_hash(audio_path)


def mark_processed(audio_path: Path, output_path: Path, log: dict) -> None:
    """ファイルを処理済みとしてログに記録"""
    key = str(audio_path.resolve())
    log[key] = {
        "hash": file_hash(audio_path),
        "output": str(output_path),
        "processed_at": date.today().isoformat(),
    }


# ---------------------------------------------------------------------------
# パイプライン（1ファイル処理）
# ---------------------------------------------------------------------------
def process_file(audio_path: Path, cfg: dict) -> Path | None:
    """音声ファイル1つを処理して Markdown を生成"""
    if not audio_path.exists():
        logger.error("ファイルが見つかりません: %s", audio_path)
        return None

    if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
        logger.error("非対応の形式です: %s", audio_path.suffix)
        return None

    # 1. 文字起こし
    text = transcribe(audio_path, cfg)
    if not text:
        logger.warning("文字起こし結果が空です: %s", audio_path.name)
        return None

    # 2. 要約・整形（失敗時は None）
    summary = summarize(text, cfg)

    # 3. 保存
    output_path = save_markdown(text, summary, cfg)
    return output_path


# ---------------------------------------------------------------------------
# watch モード
# ---------------------------------------------------------------------------
def watch_directory(cfg: dict) -> None:
    """ディレクトリを監視して音声ファイルを自動処理"""
    watch_dir = Path(get(cfg, "watch", "dir", "~/Workspace/inbox")).expanduser()
    extensions = set(get(cfg, "watch", "extensions", [".wav", ".m4a", ".mp3", ".flac"]))

    if not watch_dir.exists():
        logger.error("監視ディレクトリが存在しません: %s", watch_dir)
        sys.exit(1)

    logger.info("監視開始: %s (拡張子: %s)", watch_dir, ", ".join(extensions))
    logger.info("Ctrl+C で停止")

    log = load_processed_log(cfg)

    try:
        while True:
            for ext in extensions:
                for audio_path in watch_dir.glob(f"*{ext}"):
                    if is_processed(audio_path, log):
                        continue
                    logger.info("新規ファイル検出: %s", audio_path.name)
                    output_path = process_file(audio_path, cfg)
                    if output_path:
                        mark_processed(audio_path, output_path, log)
                        save_processed_log(log, cfg)
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("監視を停止しました")


# ---------------------------------------------------------------------------
# --list: 処理済みファイル一覧
# ---------------------------------------------------------------------------
def list_processed(cfg: dict) -> None:
    """処理済みファイルの一覧を表示"""
    log = load_processed_log(cfg)
    if not log:
        print("処理済みファイルはありません。")
        return

    print(f"処理済みファイル ({len(log)} 件):")
    print("-" * 60)
    for path, info in sorted(log.items(), key=lambda x: x[1].get("processed_at", "")):
        print(f"  {info.get('processed_at', '?')}  {Path(path).name}")
        print(f"    → {info.get('output', '?')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="voice-to-rag: 音声メモ → RAG 自動取り込みパイプライン",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="処理する音声ファイル (.wav/.m4a/.mp3/.flac)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="inbox 監視モード（音声ファイルを自動検出）",
    )
    parser.add_argument(
        "--watch-dir",
        type=str,
        default=None,
        help="監視ディレクトリを指定（デフォルト: config の watch.dir）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_processed",
        help="処理済みファイル一覧を表示",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="設定ファイルのパス（デフォルト: ./config.yml）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # 設定読み込み
    config_path = Path(args.config) if args.config else Path(__file__).parent / "config.yml"
    cfg = load_config(config_path)

    # --list モード
    if args.list_processed:
        list_processed(cfg)
        return 0

    # --watch モード
    if args.watch:
        if args.watch_dir:
            cfg["watch"]["dir"] = args.watch_dir
        watch_directory(cfg)
        return 0

    # ファイル指定モード
    if not args.files:
        parser.print_help()
        return 1

    log = load_processed_log(cfg)
    success_count = 0

    for filepath in args.files:
        audio_path = Path(filepath).resolve()
        if is_processed(audio_path, log):
            logger.info("スキップ（処理済み）: %s", audio_path.name)
            continue

        output_path = process_file(audio_path, cfg)
        if output_path:
            mark_processed(audio_path, output_path, log)
            success_count += 1

    save_processed_log(log, cfg)
    logger.info("完了: %d/%d ファイル処理", success_count, len(args.files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
