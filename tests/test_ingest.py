"""voice-to-rag テスト"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# テスト対象をインポート
sys_path_added = False
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

    def test_summarize_ollama_not_installed(self):
        """ollama 未インストール時は None"""
        with patch.dict("sys.modules", {"ollama": None}):
            cfg = ingest.load_config(None)
            # ImportError を発生させる
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

    def test_is_processed(self, tmp_path):
        f = tmp_path / "test.wav"
        f.write_bytes(b"audio data")

        log = {}
        assert not ingest.is_processed(f, log)

        ingest.mark_processed(f, Path("/tmp/out.md"), log)
        assert ingest.is_processed(f, log)

        # ファイル内容が変わると未処理扱い
        f.write_bytes(b"different audio data")
        assert not ingest.is_processed(f, log)


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
