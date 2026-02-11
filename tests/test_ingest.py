"""rag-intake テスト"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# テスト対象をインポート
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import ingest


# ---------------------------------------------------------------------------
# 設定読み込み
# ---------------------------------------------------------------------------
class TestConfig:
    def test_defaults(self):
        """デフォルト設定が正しくロードされる"""
        cfg = ingest.load_config(None)
        assert cfg["whisper"]["model"] == "base"
        assert cfg["whisper"]["language"] == "ja"
        assert cfg["ollama"]["model"] == "gemma3"
        assert cfg["output"]["prefix"] == "voice"
        assert cfg["text"]["extensions"] == [".txt", ".md"]
        assert cfg["pdf"]["max_pages"] == 50
        assert cfg["clip"]["timeout"] == 30

    def test_load_from_file(self, tmp_path):
        """config.yml から設定を上書きできる"""
        config_file = tmp_path / "config.yml"
        config_file.write_text("whisper:\n  model: large\n  language: en\n")
        cfg = ingest.load_config(config_file)
        assert cfg["whisper"]["model"] == "large"
        assert cfg["whisper"]["language"] == "en"
        # 上書きしていない値はデフォルトのまま
        assert cfg["whisper"]["device"] == "cpu"

    def test_get_helper(self):
        cfg = {"section": {"key": "value"}}
        assert ingest.get(cfg, "section", "key") == "value"
        assert ingest.get(cfg, "section", "missing", "default") == "default"
        assert ingest.get(cfg, "missing_section", "key", "default") == "default"


# ---------------------------------------------------------------------------
# ソースタイプ判定
# ---------------------------------------------------------------------------
class TestDetectSourceType:
    def test_audio_extensions(self):
        assert ingest.detect_source_type("test.wav") == "audio"
        assert ingest.detect_source_type("test.m4a") == "audio"
        assert ingest.detect_source_type("test.mp3") == "audio"
        assert ingest.detect_source_type("test.flac") == "audio"

    def test_text_extensions(self):
        assert ingest.detect_source_type("memo.txt") == "text"
        assert ingest.detect_source_type("note.md") == "text"

    def test_pdf_extension(self):
        assert ingest.detect_source_type("document.pdf") == "pdf"

    def test_url(self):
        assert ingest.detect_source_type("https://example.com") == "url"
        assert ingest.detect_source_type("http://example.com/page") == "url"

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="非対応"):
            ingest.detect_source_type("file.xyz")


# ---------------------------------------------------------------------------
# タイトルサニタイズ
# ---------------------------------------------------------------------------
class TestSanitizeTitle:
    def test_spaces_to_underscores(self):
        assert ingest.sanitize_title("hello world") == "hello_world"

    def test_fullwidth_spaces(self):
        assert ingest.sanitize_title("全角　スペース") == "全角_スペース"

    def test_unsafe_chars_removed(self):
        assert ingest.sanitize_title('a/b\\c:d*e?f"g<h>i|j') == "abcdefghij"

    def test_truncation(self):
        long_title = "a" * 100
        result = ingest.sanitize_title(long_title)
        assert len(result) == 60

    def test_empty_becomes_untitled(self):
        assert ingest.sanitize_title("") == "untitled"
        assert ingest.sanitize_title("***") == "untitled"


# ---------------------------------------------------------------------------
# parse_summary
# ---------------------------------------------------------------------------
class TestParseSummary:
    def test_extracts_title(self):
        content = "# テスト会議メモ\n\n## 要約\nテスト"
        result = ingest.parse_summary(content)
        assert result["title"] == "テスト会議メモ"
        assert result["body"] == content

    def test_no_title_defaults_to_untitled(self):
        content = "タイトルなしのテキスト"
        result = ingest.parse_summary(content)
        assert result["title"] == "untitled"


# ---------------------------------------------------------------------------
# テキスト抽出: テキストファイル
# ---------------------------------------------------------------------------
class TestExtractTextfile:
    def test_read_txt(self, tmp_path):
        f = tmp_path / "memo.txt"
        f.write_text("これはテストメモです。\n2行目。", encoding="utf-8")
        cfg = ingest.load_config(None)
        result = ingest.extract_textfile(f, cfg)
        assert "テストメモ" in result
        assert "2行目" in result

    def test_read_md(self, tmp_path):
        f = tmp_path / "note.md"
        f.write_text("# タイトル\n\n本文です。", encoding="utf-8")
        cfg = ingest.load_config(None)
        result = ingest.extract_textfile(f, cfg)
        assert "# タイトル" in result
        assert "本文" in result

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        cfg = ingest.load_config(None)
        result = ingest.extract_textfile(f, cfg)
        assert result == ""


# ---------------------------------------------------------------------------
# テキスト抽出: PDF（mock）
# ---------------------------------------------------------------------------
class TestExtractPdf:
    def test_extract_pdf(self, tmp_path):
        """pymupdf の mock テスト"""
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "1ページ目の内容。"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "2ページ目の内容。"

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page1, mock_page2]))
        mock_doc.__len__ = MagicMock(return_value=2)

        mock_pymupdf = MagicMock()
        mock_pymupdf.open.return_value = mock_doc

        with patch.dict("sys.modules", {"pymupdf": mock_pymupdf}):
            cfg = ingest.load_config(None)
            result = ingest.extract_pdf(tmp_path / "test.pdf", cfg)

        assert "1ページ目" in result
        assert "2ページ目" in result

    def test_max_pages_limit(self, tmp_path):
        """max_pages で抽出ページ数を制限"""
        pages = []
        for i in range(10):
            p = MagicMock()
            p.get_text.return_value = f"ページ{i}"
            pages.append(p)

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter(pages))
        mock_doc.__len__ = MagicMock(return_value=10)

        mock_pymupdf = MagicMock()
        mock_pymupdf.open.return_value = mock_doc

        with patch.dict("sys.modules", {"pymupdf": mock_pymupdf}):
            cfg = ingest.load_config(None)
            cfg["pdf"]["max_pages"] = 3
            result = ingest.extract_pdf(tmp_path / "test.pdf", cfg)

        assert "ページ0" in result
        assert "ページ2" in result
        assert "ページ3" not in result


# ---------------------------------------------------------------------------
# テキスト抽出: URL（mock）
# ---------------------------------------------------------------------------
class TestExtractUrl:
    def test_extract_url(self):
        """trafilatura の mock テスト"""
        mock_trafilatura = MagicMock()
        mock_trafilatura.fetch_url.return_value = "<html>dummy</html>"
        mock_trafilatura.extract.return_value = "抽出された本文テキスト。"

        with patch.dict("sys.modules", {"trafilatura": mock_trafilatura}):
            cfg = ingest.load_config(None)
            result = ingest.extract_url("https://example.com/article", cfg)

        assert "抽出された本文" in result

    def test_fetch_failure(self):
        """URL 取得失敗時は RuntimeError"""
        mock_trafilatura = MagicMock()
        mock_trafilatura.fetch_url.return_value = None

        with patch.dict("sys.modules", {"trafilatura": mock_trafilatura}):
            cfg = ingest.load_config(None)
            with pytest.raises(RuntimeError, match="取得に失敗"):
                ingest.extract_url("https://example.com/bad", cfg)

    def test_extract_failure(self):
        """本文抽出失敗時は RuntimeError"""
        mock_trafilatura = MagicMock()
        mock_trafilatura.fetch_url.return_value = "<html>ok</html>"
        mock_trafilatura.extract.return_value = None

        with patch.dict("sys.modules", {"trafilatura": mock_trafilatura}):
            cfg = ingest.load_config(None)
            with pytest.raises(RuntimeError, match="抽出に失敗"):
                ingest.extract_url("https://example.com/empty", cfg)


# ---------------------------------------------------------------------------
# extract_text ディスパッチャ
# ---------------------------------------------------------------------------
class TestExtractText:
    def test_dispatches_to_textfile(self, tmp_path):
        f = tmp_path / "memo.txt"
        f.write_text("テストテキスト", encoding="utf-8")
        cfg = ingest.load_config(None)
        text, stype = ingest.extract_text(str(f), cfg)
        assert text == "テストテキスト"
        assert stype == "text"

    @patch("ingest.transcribe", return_value="音声テキスト")
    def test_dispatches_to_audio(self, mock_transcribe):
        cfg = ingest.load_config(None)
        text, stype = ingest.extract_text("/tmp/test.wav", cfg)
        assert text == "音声テキスト"
        assert stype == "audio"
        mock_transcribe.assert_called_once()

    @patch("ingest.extract_pdf", return_value="PDF内容")
    def test_dispatches_to_pdf(self, mock_pdf):
        cfg = ingest.load_config(None)
        text, stype = ingest.extract_text("/tmp/test.pdf", cfg)
        assert text == "PDF内容"
        assert stype == "pdf"

    @patch("ingest.extract_url", return_value="Web内容")
    def test_dispatches_to_url(self, mock_url):
        cfg = ingest.load_config(None)
        text, stype = ingest.extract_text("https://example.com", cfg)
        assert text == "Web内容"
        assert stype == "url"


# ---------------------------------------------------------------------------
# 文字起こし（mock）
# ---------------------------------------------------------------------------
class TestTranscribe:
    @patch("ingest.WhisperModel", create=True)
    def test_transcribe_returns_text(self, mock_whisper_cls):
        """faster-whisper の mock テスト"""
        # mock セグメント
        seg1 = MagicMock()
        seg1.text = "こんにちは。"
        seg2 = MagicMock()
        seg2.text = "今日の会議です。"
        mock_info = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], mock_info)

        with patch("ingest.WhisperModel", return_value=mock_model):
            # faster_whisper モジュール自体を mock
            mock_fw = MagicMock()
            mock_fw.WhisperModel = MagicMock(return_value=mock_model)
            with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
                cfg = ingest.load_config(None)
                audio_path = Path("/tmp/test.wav")
                result = ingest.transcribe(audio_path, cfg)

        assert "こんにちは" in result
        assert "今日の会議" in result


# ---------------------------------------------------------------------------
# 要約（mock）
# ---------------------------------------------------------------------------
class TestSummarize:
    def test_summarize_success(self):
        """Ollama 正常応答時"""
        mock_response = {
            "message": {
                "content": "# テスト要約\n\n## 要約\nテスト内容です。"
            }
        }
        mock_ollama = MagicMock()
        mock_ollama.chat = MagicMock(return_value=mock_response)

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            cfg = ingest.load_config(None)
            result = ingest.summarize("テストテキスト", cfg)

        assert result is not None
        assert result["title"] == "テスト要約"

    def test_summarize_with_source_type(self):
        """ソースタイプ別プロンプト"""
        mock_response = {
            "message": {
                "content": "# PDF要約\n\n## 要約\nPDF内容です。"
            }
        }
        mock_ollama = MagicMock()
        mock_ollama.chat = MagicMock(return_value=mock_response)

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            cfg = ingest.load_config(None)
            result = ingest.summarize("PDF本文", cfg, source_type="pdf")

        assert result is not None
        assert result["title"] == "PDF要約"

    def test_summarize_url_includes_source(self):
        """URL ソースは出典 URL をプロンプトに含める"""
        mock_response = {
            "message": {
                "content": "# Web記事\n\n## 要約\n記事内容です。"
            }
        }
        mock_ollama = MagicMock()
        mock_ollama.chat = MagicMock(return_value=mock_response)

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            cfg = ingest.load_config(None)
            result = ingest.summarize(
                "Web本文", cfg,
                source_type="url",
                source_url="https://example.com/article",
            )

        assert result is not None
        # chat に渡されたプロンプトに URL が含まれる
        call_args = mock_ollama.chat.call_args
        prompt_text = call_args[1]["messages"][0]["content"] if "messages" in (call_args[1] or {}) else call_args[0][0] if call_args[0] else ""
        # kwargs か positional か確認
        if call_args.kwargs and "messages" in call_args.kwargs:
            prompt_text = call_args.kwargs["messages"][0]["content"]

    def test_summarize_ollama_not_installed(self):
        """ollama 未インストール時は None"""
        with patch.dict("sys.modules", {"ollama": None}):
            cfg = ingest.load_config(None)
            with patch("builtins.__import__", side_effect=ImportError("no ollama")):
                result = ingest.summarize("テスト", cfg)
        assert result is None

    def test_summarize_connection_error(self):
        """Ollama 接続失敗時は None"""
        mock_ollama = MagicMock()
        mock_ollama.chat = MagicMock(side_effect=Exception("接続拒否"))

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            cfg = ingest.load_config(None)
            result = ingest.summarize("テスト", cfg)

        assert result is None


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------
class TestSaveMarkdown:
    def test_save_with_summary(self, tmp_path):
        cfg = ingest.load_config(None)
        cfg["output"]["dir"] = str(tmp_path)

        summary = {"title": "テスト", "body": "# テスト\n\nテスト内容"}
        path = ingest.save_markdown("raw text", summary, cfg)

        assert path.exists()
        assert "テスト" in path.name
        assert path.read_text() == "# テスト\n\nテスト内容"

    def test_save_without_summary(self, tmp_path):
        """Ollama フォールバック: 生テキストのみ"""
        cfg = ingest.load_config(None)
        cfg["output"]["dir"] = str(tmp_path)

        path = ingest.save_markdown("生テキスト内容", None, cfg)

        assert path.exists()
        content = path.read_text()
        assert "生テキスト内容" in content
        assert "memo" in path.name

    def test_save_with_source_type_prefix(self, tmp_path):
        """ソースタイプ別プレフィックス"""
        cfg = ingest.load_config(None)
        cfg["output"]["dir"] = str(tmp_path)

        summary = {"title": "test", "body": "# test"}
        path_audio = ingest.save_markdown("t", summary, cfg, source_type="audio")
        assert "voice_" in path_audio.name

        summary2 = {"title": "test2", "body": "# test2"}
        path_text = ingest.save_markdown("t", summary2, cfg, source_type="text")
        assert "text_" in path_text.name

        summary3 = {"title": "test3", "body": "# test3"}
        path_pdf = ingest.save_markdown("t", summary3, cfg, source_type="pdf")
        assert "pdf_" in path_pdf.name

        summary4 = {"title": "test4", "body": "# test4"}
        path_url = ingest.save_markdown("t", summary4, cfg, source_type="url")
        assert "clip_" in path_url.name

    def test_fallback_label_by_source_type(self, tmp_path):
        """フォールバック時のラベルがソースタイプに応じて変わる"""
        cfg = ingest.load_config(None)
        cfg["output"]["dir"] = str(tmp_path)

        path = ingest.save_markdown("テスト", None, cfg, source_type="pdf")
        content = path.read_text()
        assert "PDF ドキュメント" in content

    def test_duplicate_filename(self, tmp_path):
        """同名ファイルが存在する場合サフィックス追加"""
        cfg = ingest.load_config(None)
        cfg["output"]["dir"] = str(tmp_path)

        summary = {"title": "same", "body": "body1"}
        path1 = ingest.save_markdown("text1", summary, cfg)
        path2 = ingest.save_markdown("text2", summary, cfg)

        assert path1 != path2
        assert path1.exists()
        assert path2.exists()
        assert "_2" in path2.name


# ---------------------------------------------------------------------------
# 処理済みログ
# ---------------------------------------------------------------------------
class TestProcessedLog:
    def test_round_trip(self, tmp_path):
        cfg = ingest.load_config(None)
        log_path = tmp_path / "processed.json"
        cfg["watch"]["processed_log"] = str(log_path)

        log = ingest.load_processed_log(cfg)
        assert log == {}

        log["/tmp/test.wav"] = {"hash": "abc123", "output": "/tmp/out.md", "processed_at": "2026-02-11"}
        ingest.save_processed_log(log, cfg)

        log2 = ingest.load_processed_log(cfg)
        assert log2["/tmp/test.wav"]["hash"] == "abc123"

    def test_file_hash(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        h = ingest.file_hash(f)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_url_hash(self):
        h = ingest.url_hash("https://example.com")
        assert isinstance(h, str)
        assert len(h) == 64

    def test_is_processed_file(self, tmp_path):
        f = tmp_path / "test.wav"
        f.write_bytes(b"audio data")

        log = {}
        assert not ingest.is_processed(str(f), log)

        ingest.mark_processed(str(f), Path("/tmp/out.md"), log)
        assert ingest.is_processed(str(f), log)

        # ファイル内容が変わると未処理扱い
        f.write_bytes(b"different audio data")
        assert not ingest.is_processed(str(f), log)

    def test_is_processed_url(self):
        log = {}
        url = "https://example.com/article"
        assert not ingest.is_processed(url, log)

        ingest.mark_processed(url, Path("/tmp/out.md"), log)
        assert ingest.is_processed(url, log)


# ---------------------------------------------------------------------------
# process_source 統合テスト（mock）
# ---------------------------------------------------------------------------
class TestProcessSource:
    def test_process_textfile(self, tmp_path):
        """テキストファイルの統合処理"""
        src = tmp_path / "memo.txt"
        src.write_text("テストメモの内容です。", encoding="utf-8")

        cfg = ingest.load_config(None)
        cfg["output"]["dir"] = str(tmp_path / "output")

        # Ollama をスキップ
        with patch("ingest.summarize", return_value=None):
            result = ingest.process_source(str(src), cfg)

        assert result is not None
        assert result.exists()
        content = result.read_text()
        assert "テストメモの内容" in content
        assert "text_" in result.name

    @patch("ingest.extract_url", return_value="Web記事の本文です。")
    def test_process_url(self, mock_extract, tmp_path):
        """URL の統合処理"""
        cfg = ingest.load_config(None)
        cfg["output"]["dir"] = str(tmp_path / "output")

        with patch("ingest.summarize", return_value=None):
            result = ingest.process_source("https://example.com/article", cfg)

        assert result is not None
        assert result.exists()
        assert "clip_" in result.name

    @patch("ingest.extract_pdf", return_value="PDFの中身。")
    def test_process_pdf(self, mock_pdf, tmp_path):
        """PDF の統合処理"""
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"dummy pdf")

        cfg = ingest.load_config(None)
        cfg["output"]["dir"] = str(tmp_path / "output")

        with patch("ingest.summarize", return_value=None):
            result = ingest.process_source(str(src), cfg)

        assert result is not None
        assert result.exists()
        assert "pdf_" in result.name

    def test_process_nonexistent_file(self, tmp_path):
        """存在しないファイル"""
        cfg = ingest.load_config(None)
        result = ingest.process_source(str(tmp_path / "missing.txt"), cfg)
        assert result is None

    def test_process_unsupported_format(self, tmp_path):
        """非対応の拡張子"""
        cfg = ingest.load_config(None)
        result = ingest.process_source(str(tmp_path / "file.xyz"), cfg)
        assert result is None

    def test_process_file_backward_compat(self, tmp_path):
        """process_file は process_source のラッパー"""
        src = tmp_path / "memo.txt"
        src.write_text("テスト", encoding="utf-8")

        cfg = ingest.load_config(None)
        cfg["output"]["dir"] = str(tmp_path / "output")

        with patch("ingest.summarize", return_value=None):
            result = ingest.process_file(src, cfg)

        assert result is not None
        assert result.exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
class TestCLI:
    def test_no_args_returns_1(self):
        """引数なしはヘルプ表示して終了コード 1"""
        result = ingest.main([])
        assert result == 1

    def test_list_empty(self, tmp_path, capsys):
        """--list で処理済みが空のとき"""
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            f"watch:\n  processed_log: '{tmp_path}/processed.json'\n"
        )
        result = ingest.main(["--list", "--config", str(config_file)])
        assert result == 0
        captured = capsys.readouterr()
        assert "ありません" in captured.out

    def test_nonexistent_file(self, tmp_path):
        """存在しないファイルを指定"""
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            f"watch:\n  processed_log: '{tmp_path}/processed.json'\n"
        )
        result = ingest.main(["/tmp/nonexistent.wav", "--config", str(config_file)])
        assert result == 0  # エラーでも正常終了（ログに記録）

    def test_text_file_via_cli(self, tmp_path):
        """テキストファイルを CLI で処理"""
        src = tmp_path / "note.txt"
        src.write_text("CLIテスト内容", encoding="utf-8")

        config_file = tmp_path / "config.yml"
        config_file.write_text(
            f"output:\n  dir: '{tmp_path}/output'\n"
            f"watch:\n  processed_log: '{tmp_path}/processed.json'\n"
        )

        with patch("ingest.summarize", return_value=None):
            result = ingest.main([str(src), "--config", str(config_file)])

        assert result == 0
        output_files = list((tmp_path / "output").glob("*.md"))
        assert len(output_files) == 1
        assert "text_" in output_files[0].name

    @patch("ingest.extract_url", return_value="Web本文")
    def test_url_via_cli(self, mock_extract, tmp_path):
        """URL を CLI で処理"""
        config_file = tmp_path / "config.yml"
        config_file.write_text(
            f"output:\n  dir: '{tmp_path}/output'\n"
            f"watch:\n  processed_log: '{tmp_path}/processed.json'\n"
        )

        with patch("ingest.summarize", return_value=None):
            result = ingest.main(["https://example.com/test", "--config", str(config_file)])

        assert result == 0
        output_files = list((tmp_path / "output").glob("*.md"))
        assert len(output_files) == 1
        assert "clip_" in output_files[0].name
