import os
import tempfile
import pytest
import typer
from typer.testing import CliRunner
from src.cli import app, validate_api_key, load_env, parse_retry_delay, handle_embedding_error, get_source_files, ensure_gitignore

runner = CliRunner()

@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Ensure clean environment and isolated working directory before each test."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

def test_validate_api_key_missing():
    with pytest.raises(typer.Exit) as exc_info:
        validate_api_key()
    assert exc_info.value.exit_code == 1

def test_validate_api_key_empty_or_whitespace(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    with pytest.raises(typer.Exit) as exc_info:
        validate_api_key()
    assert exc_info.value.exit_code == 1

def test_validate_api_key_placeholder(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "your_api_key_here")
    with pytest.raises(typer.Exit) as exc_info:
        validate_api_key()
    assert exc_info.value.exit_code == 1

def test_validate_api_key_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestGeminiKey123")
    key = validate_api_key()
    assert key == "AIzaSyTestGeminiKey123"

def test_validate_api_key_google(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyTestGoogleKey456")
    key = validate_api_key()
    assert key == "AIzaSyTestGoogleKey456"

def test_validate_api_key_loads_from_target_dir_env():
    with tempfile.TemporaryDirectory() as temp_dir:
        env_file = os.path.join(temp_dir, ".env")
        with open(env_file, "w", encoding="utf-8") as f:
            f.write('# Comment line\n')
            f.write('GEMINI_API_KEY="AIzaSyFromEnvFile789"\n')
            
        key = validate_api_key(temp_dir)
        assert key == "AIzaSyFromEnvFile789"
        assert os.environ.get("GEMINI_API_KEY") == "AIzaSyFromEnvFile789"

def test_cli_index_aborts_without_api_key():
    with tempfile.TemporaryDirectory() as temp_dir:
        result = runner.invoke(app, ["index", temp_dir])
        assert result.exit_code == 1
        assert "Gemini API key not found" in result.output
        
        # Verify no database or state files were created
        db_path = os.path.join(temp_dir, ".code_search_db")
        state_path = os.path.join(temp_dir, ".index_state.json")
        assert not os.path.exists(db_path)
        assert not os.path.exists(state_path)

def test_cli_index_proceeds_with_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "valid_test_key")
    with tempfile.TemporaryDirectory() as temp_dir:
        sample_file = os.path.join(temp_dir, "sample.py")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("def sample():\n    pass\n")
            
        from unittest.mock import patch
        with patch("src.cli.get_embedding", return_value=[0.1] * 3072):
            result = runner.invoke(app, ["index", temp_dir])
            assert result.exit_code == 0
            assert "Indexing complete" in result.output
            
            db_path = os.path.join(temp_dir, ".code_search_db")
            state_path = os.path.join(temp_dir, ".index_state.json")
            assert os.path.exists(db_path)
            assert os.path.exists(state_path)

def test_validate_api_key_override_param():
    key = validate_api_key(api_key="direct_param_key")
    assert key == "direct_param_key"
    assert os.environ.get("GEMINI_API_KEY") == "direct_param_key"

def test_cli_index_with_api_key_flag():
    with tempfile.TemporaryDirectory() as temp_dir:
        sample_file = os.path.join(temp_dir, "sample.py")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("def sample():\n    pass\n")
            
        from unittest.mock import patch
        with patch("src.cli.get_embedding", return_value=[0.1] * 3072):
            result = runner.invoke(app, ["index", temp_dir, "--api-key", "my_flag_key"])
            assert result.exit_code == 0
            assert "Indexing complete" in result.output

def test_parse_retry_delay():
    msg1 = "Please retry in 23.874270388s."
    assert parse_retry_delay(msg1) == "23.874270388s"

    msg2 = "{'error': ..., 'details': [{'@type': ..., 'retryDelay': '45s'}]}"
    assert parse_retry_delay(msg2) == "45s"

    msg3 = "Unknown generic failure without delay"
    assert parse_retry_delay(msg3) is None

def test_handle_embedding_error_quota():
    from google.genai.errors import APIError
    err = APIError(429, "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: free_tier_requests, limit: 1000. Please retry in 23.8s.")
    
    with pytest.raises(typer.Exit) as exc_info:
        handle_embedding_error(err, file_path="test_file.py")
    assert exc_info.value.exit_code == 1

def test_cli_index_aborts_on_quota_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "valid_test_key")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create 2 sample files
        f1 = os.path.join(temp_dir, "file1.py")
        f2 = os.path.join(temp_dir, "file2.py")
        with open(f1, "w", encoding="utf-8") as f:
            f.write("def func1(): pass\n")
        with open(f2, "w", encoding="utf-8") as f:
            f.write("def func2(): pass\n")
            
        from google.genai.errors import APIError
        from unittest.mock import patch
        
        # Simulate 429 quota exhaustion on get_embedding
        quota_err = APIError(429, "429 RESOURCE_EXHAUSTED. Quota exceeded for free_tier_requests. Please retry in 25s.")
        with patch("src.cli.get_embedding", side_effect=quota_err):
            result = runner.invoke(app, ["index", temp_dir])
            # Process must stop immediately with exit code 1
            assert result.exit_code == 1
            assert "API Quota Exceeded (429 RESOURCE_EXHAUSTED)" in result.output
            assert "Suggested Wait Time: 25s" in result.output
            assert "1,000 embedding requests per day" in result.output

def test_get_source_files_ignores_target_and_build_dirs():
    with tempfile.TemporaryDirectory() as temp_dir:
        src_dir = os.path.join(temp_dir, "src")
        target_dir = os.path.join(temp_dir, "target", "classes")
        node_modules = os.path.join(temp_dir, "node_modules", "pkg")
        
        os.makedirs(src_dir)
        os.makedirs(target_dir)
        os.makedirs(node_modules)
        
        valid_file = os.path.join(src_dir, "app.py")
        target_file = os.path.join(target_dir, "Generated.html")
        node_file = os.path.join(node_modules, "index.js")
        
        with open(valid_file, "w") as f: f.write("print('hi')")
        with open(target_file, "w") as f: f.write("<html></html>")
        with open(node_file, "w") as f: f.write("console.log('hi')")
        
        files = get_source_files(temp_dir)
        assert valid_file in files
        assert target_file not in files
        assert node_file not in files

def test_ensure_gitignore_creates_new_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        ensure_gitignore(temp_dir)
        gitignore_path = os.path.join(temp_dir, ".gitignore")
        assert os.path.exists(gitignore_path)
        with open(gitignore_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert ".code_search_db/" in content
        assert ".index_state.json" in content

def test_ensure_gitignore_appends_missing():
    with tempfile.TemporaryDirectory() as temp_dir:
        gitignore_path = os.path.join(temp_dir, ".gitignore")
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write("node_modules/\n.env")  # Note: no trailing newline
        ensure_gitignore(temp_dir)
        with open(gitignore_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "node_modules/" in content
        assert ".env" in content
        assert ".code_search_db/" in content
        assert ".index_state.json" in content
        # Ensure it wasn't merged onto the same line as .env
        lines = [line.strip() for line in content.splitlines()]
        assert ".env" in lines
        assert ".code_search_db/" in lines
        assert ".index_state.json" in lines

def test_ensure_gitignore_idempotent():
    with tempfile.TemporaryDirectory() as temp_dir:
        gitignore_path = os.path.join(temp_dir, ".gitignore")
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write(".code_search_db/\n.index_state.json\n")
        ensure_gitignore(temp_dir)
        with open(gitignore_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Should not duplicate lines
        assert content.count(".code_search_db") == 1
        assert content.count(".index_state.json") == 1

def test_cli_index_updates_gitignore(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "valid_test_key")
    with tempfile.TemporaryDirectory() as temp_dir:
        sample_file = os.path.join(temp_dir, "app.py")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("def app(): pass\n")
            
        from unittest.mock import patch
        with patch("src.cli.get_embedding", return_value=[0.1] * 3072):
            result = runner.invoke(app, ["index", temp_dir])
            assert result.exit_code == 0
            
            gitignore_path = os.path.join(temp_dir, ".gitignore")
            assert os.path.exists(gitignore_path)
            with open(gitignore_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert ".code_search_db/" in content
            assert ".index_state.json" in content




