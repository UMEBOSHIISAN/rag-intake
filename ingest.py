#!/usr/bin/env python3
"""rag-intake: 汎用 RAG 取り込みパイプライン（音声・テキスト・PDF・URL → Markdown）"""

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
    "text": {
        "extensions": [".txt", ".md"],
    },
    "pdf": {
        "max_pages": 50,
    },
    "clip": {
        "timeout": 30,
    },
    "watch": {
        "dir": "./watch",
        "extensions": [".wav", ".m4a", ".mp3", ".flac", ".txt", ".md", ".pdf"],
        "processed_log": "~/.rag-intake/processed.json",
    },
}

AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3", ".flac"}
TEXT_EXTENSIONS = {".txt", ".md"}
PDF_EXTENSIONS = {".pdf"}


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
# ソースタイプ判定
# ---------------------------------------------------------------------------
def detect_source_type(source: str) -> str:
    """入力ソースのタイプを判定して返す: 'audio' / 'text' / 'pdf' / 'url'"""
    if source.startswith("http://") or source.startswith("https://"):
        return "url"
    p = Path(source)
    ext = p.suffix.lower()
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in TEXT_EXTENSIONS:
        return "text"
    if ext in PDF_EXTENSIONS:
        return "pdf"
    raise ValueError(f"非対応の入力ソースです: {source}")


# ---------------------------------------------------------------------------
# テキスト抽出（タイプ別）
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


def extract_textfile(path: Path, cfg: dict) -> str:
    """テキストファイル (.txt/.md) を読み込んで返す"""
    logger.info("テキスト読み込み: %s", path.name)
    return path.read_text(encoding="utf-8").strip()


def extract_pdf(path: Path, cfg: dict) -> str:
    """PDF ファイルからテキストを抽出（pymupdf 使用）"""
    import pymupdf

    max_pages = get(cfg, "pdf", "max_pages", 50)

    logger.info("PDF テキスト抽出: %s (最大 %d ページ)", path.name, max_pages)
    doc = pymupdf.open(str(path))
    text_parts = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            logger.warning("最大ページ数 (%d) に達したため中断", max_pages)
            break
        text_parts.append(page.get_text())
    doc.close()

    result = "\n".join(text_parts).strip()
    logger.info("PDF 抽出完了: %d ページ, テキスト長=%d", min(len(doc), max_pages), len(result))
    return result


def extract_url(url: str, cfg: dict) -> str:
    """URL から本文を抽出（trafilatura 使用）"""
    import trafilatura

    timeout = get(cfg, "clip", "timeout", 30)

    logger.info("Web ページ取得: %s", url)
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise RuntimeError(f"URL の取得に失敗しました: {url}")

    text = trafilatura.extract(downloaded)
    if not text:
        raise RuntimeError(f"本文の抽出に失敗しました: {url}")

    logger.info("Web 本文抽出完了: テキスト長=%d", len(text))
    return text


def extract_text(source: str, cfg: dict) -> tuple[str, str]:
    """入力ソースからテキストを抽出。(text, source_type) を返す"""
    source_type = detect_source_type(source)

    if source_type == "audio":
        path = Path(source)
        return transcribe(path, cfg), "audio"
    elif source_type == "text":
        path = Path(source)
        return extract_textfile(path, cfg), "text"
    elif source_type == "pdf":
        path = Path(source)
        return extract_pdf(path, cfg), "pdf"
    elif source_type == "url":
        return extract_url(source, cfg), "url"
    else:
        raise ValueError(f"非対応のソースタイプ: {source_type}")


# ---------------------------------------------------------------------------
# 要約・整形（Ollama）
# ---------------------------------------------------------------------------
SUMMARIZE_PROMPTS = {
    "audio": "以下は音声メモの文字起こしテキストです。",
    "text": "以下はテキストメモです。",
    "pdf": "以下は PDF ドキュメントから抽出したテキストです。",
    "url": "以下は Web ページから抽出したテキストです。",
}

SUMMARIZE_TEMPLATE = """\
{source_description}
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
{extra_instructions}
---

テキスト:
{text}
"""

# 後方互換: 旧テストから参照される場合用
SUMMARIZE_PROMPT = SUMMARIZE_TEMPLATE

SOURCE_TYPE_PREFIX = {
    "audio": "voice",
    "text": "text",
    "pdf": "pdf",
    "url": "clip",
}

FALLBACK_LABELS = {
    "audio": "音声メモ",
    "text": "テキストメモ",
    "pdf": "PDF ドキュメント",
    "url": "Web クリップ",
}


def summarize(text: str, cfg: dict, source_type: str = "audio", source_url: str | None = None) -> dict | None:
    """Ollama でテキストを要約・整形。失敗時は None を返す"""
    try:
        from ollama import chat
    except ImportError:
        logger.warning("ollama パッケージ未インストール — 生テキストで保存します")
        return None

    model = get(cfg, "ollama", "model", "gemma3")
    source_description = SUMMARIZE_PROMPTS.get(source_type, SUMMARIZE_PROMPTS["text"])

    extra_instructions = ""
    if source_type == "url" and source_url:
        extra_instructions = f"\n- 出典 URL: {source_url}\n"

    prompt = SUMMARIZE_TEMPLATE.format(
        source_description=source_description,
        text=text,
        extra_instructions=extra_instructions,
    )

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
def save_markdown(text: str, summary: dict | None, cfg: dict, source_type: str = "audio") -> Path:
    """Markdown ファイルを出力ディレクトリに保存"""
    output_dir = Path(get(cfg, "output", "dir", "~/Workspace/RAG/drafts")).expanduser()
    prefix = SOURCE_TYPE_PREFIX.get(source_type, get(cfg, "output", "prefix", "voice"))
    today = date.today().isoformat()

    if summary:
        title = summary["title"]
        body = summary["body"]
    else:
        label = FALLBACK_LABELS.get(source_type, "メモ")
        title = "memo"
        body = f"# {label}\n\n## 全文\n\n{text}\n"

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
    log_path = Path(get(cfg, "watch", "processed_log", "~/.rag-intake/processed.json")).expanduser()
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return {}


def save_processed_log(log: dict, cfg: dict) -> None:
    """処理済みログを保存"""
    log_path = Path(get(cfg, "watch", "processed_log", "~/.rag-intake/processed.json")).expanduser()
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


def url_hash(url: str) -> str:
    """URL 文字列の SHA-256 ハッシュを計算"""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def is_processed(source: str | Path, log: dict) -> bool:
    """ソースが処理済みかチェック"""
    source_str = str(source)
    if source_str.startswith("http://") or source_str.startswith("https://"):
        key = source_str
        return key in log
    path = Path(source_str).resolve()
    key = str(path)
    if key not in log:
        return False
    return log[key]["hash"] == file_hash(path)


def mark_processed(source: str | Path, output_path: Path, log: dict) -> None:
    """ソースを処理済みとしてログに記録"""
    source_str = str(source)
    if source_str.startswith("http://") or source_str.startswith("https://"):
        key = source_str
        log[key] = {
            "hash": url_hash(source_str),
            "output": str(output_path),
            "processed_at": date.today().isoformat(),
        }
    else:
        path = Path(source_str).resolve()
        key = str(path)
        log[key] = {
            "hash": file_hash(path),
            "output": str(output_path),
            "processed_at": date.today().isoformat(),
        }


# ---------------------------------------------------------------------------
# パイプライン（1ソース処理）
# ---------------------------------------------------------------------------
def process_file(audio_path: Path, cfg: dict) -> Path | None:
    """後方互換: process_source のラッパー"""
    return process_source(str(audio_path), cfg)


def process_source(source: str, cfg: dict) -> Path | None:
    """入力ソース1つを処理して Markdown を生成"""
    source_type = None
    try:
        source_type = detect_source_type(source)
    except ValueError as e:
        logger.error("%s", e)
        return None

    # ファイルの場合は存在チェック
    if source_type != "url":
        path = Path(source)
        if not path.exists():
            logger.error("ファイルが見つかりません: %s", source)
            return None

    # 1. テキスト抽出
    try:
        text, source_type = extract_text(source, cfg)
    except Exception as e:
        logger.error("テキスト抽出失敗: %s — %s", source, e)
        return None

    if not text:
        logger.warning("抽出結果が空です: %s", source)
        return None

    # 2. 要約・整形
    source_url = source if source_type == "url" else None
    summary = summarize(text, cfg, source_type=source_type, source_url=source_url)

    # 3. 保存
    output_path = save_markdown(text, summary, cfg, source_type=source_type)
    return output_path


# ---------------------------------------------------------------------------
# watch モード
# ---------------------------------------------------------------------------
def watch_directory(cfg: dict) -> None:
    """ディレクトリを監視してファイルを自動処理"""
    watch_dir = Path(get(cfg, "watch", "dir", "~/Workspace/inbox")).expanduser()
    extensions = set(get(cfg, "watch", "extensions",
                        [".wav", ".m4a", ".mp3", ".flac", ".txt", ".md", ".pdf"]))

    if not watch_dir.exists():
        logger.error("監視ディレクトリが存在しません: %s", watch_dir)
        sys.exit(1)

    logger.info("監視開始: %s (拡張子: %s)", watch_dir, ", ".join(sorted(extensions)))
    logger.info("Ctrl+C で停止")

    log = load_processed_log(cfg)

    try:
        while True:
            for ext in extensions:
                for file_path in watch_dir.glob(f"*{ext}"):
                    if is_processed(str(file_path), log):
                        continue
                    logger.info("新規ファイル検出: %s", file_path.name)
                    output_path = process_source(str(file_path), cfg)
                    if output_path:
                        mark_processed(str(file_path), output_path, log)
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
        # URL の場合はそのまま、ファイルの場合はファイル名のみ
        if path.startswith("http://") or path.startswith("https://"):
            display = path
        else:
            display = Path(path).name
        print(f"  {info.get('processed_at', '?')}  {display}")
        print(f"    → {info.get('output', '?')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="rag-intake: 汎用 RAG 取り込みパイプライン（音声・テキスト・PDF・URL → Markdown）",
    )
    parser.add_argument(
        "sources",
        nargs="*",
        help="処理する入力ソース（音声/テキスト/PDF ファイル、または URL）",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="監視モード（ファイルを自動検出）",
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

    # ソース指定モード
    if not args.sources:
        parser.print_help()
        return 1

    log = load_processed_log(cfg)
    success_count = 0

    for source in args.sources:
        # URL はそのまま、ファイルは resolve
        if source.startswith("http://") or source.startswith("https://"):
            source_key = source
        else:
            source_key = str(Path(source).resolve())

        if is_processed(source_key, log):
            logger.info("スキップ（処理済み）: %s", source)
            continue

        output_path = process_source(source, cfg)
        if output_path:
            mark_processed(source_key, output_path, log)
            success_count += 1

    save_processed_log(log, cfg)
    logger.info("完了: %d/%d ソース処理", success_count, len(args.sources))
    return 0


if __name__ == "__main__":
    sys.exit(main())
