"""Tests for tools/rust_execution_tool.py — Rust code execution handlers."""

import json
import pytest
import shutil

from tools.rust_execution_tool import (
    _handle_rust_compile as _rust_compile_handler,
    _handle_rust_cargo_run as _rust_cargo_run_handler,
    _handle_rust_version as _rust_version_handler,
    _check_rust_available,
)


RUST_AVAILABLE = shutil.which("rustc") is not None and shutil.which("cargo") is not None


# =========================================================================
# Requirements / availability
# =========================================================================

class TestRustRequirements:
    def test_check_rust_available_matches_path(self):
        assert _check_rust_available() == RUST_AVAILABLE


# =========================================================================
# rust_compile
# =========================================================================

@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust toolchain not installed")
class TestRustCompile:
    def test_hello_world(self):
        code = 'fn main() { println!("Hello, Rust!"); }'
        args = {"code": code, "edition": "2021"}
        result = json.loads(_rust_compile_handler(args))
        assert "error" not in result
        assert result["success"] is True
        assert "Hello, Rust!" in result["stdout"]

    def test_with_args(self):
        code = 'fn main() { println!("optimized"); }'
        args = {"code": code, "args": ["-O"]}
        result = json.loads(_rust_compile_handler(args))
        assert "error" not in result
        assert result["success"] is True
        assert "optimized" in result["stdout"]

    def test_with_stdin(self):
        code = '''
use std::io::{self, BufRead};
fn main() {
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        if let Ok(l) = line {
            println!("Echo: {}", l);
        }
    }
}
'''
        args = {"code": code, "stdin": "hello world"}
        result = json.loads(_rust_compile_handler(args))
        assert "error" not in result
        assert result["success"] is True
        assert "Echo: hello world" in result["stdout"]

    def test_syntax_error(self):
        code = 'fn main() { println!("missing paren" }'
        args = {"code": code}
        result = json.loads(_rust_compile_handler(args))
        # Syntax errors cause compilation failure → success=False
        assert result["success"] is False

    def test_empty_code(self):
        args = {"code": ""}
        result = json.loads(_rust_compile_handler(args))
        assert "error" in result

    def test_invalid_edition(self):
        code = 'fn main() {}'
        args = {"code": code, "edition": "9999"}
        result = json.loads(_rust_compile_handler(args))
        # Bad edition causes compilation failure
        assert result["success"] is False


# =========================================================================
# rust_cargo_run
# =========================================================================

@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust toolchain not installed")
class TestRustCargoRun:
    def test_basic_cargo_run(self):
        args = {
            "files": {
                "Cargo.toml": '[package]\nname = "test"\nversion = "0.1.0"\nedition = "2021"\n[dependencies]\n',
                "src/main.rs": 'fn main() { println!("Cargo OK"); }',
            },
            "command": "run",
        }
        result = json.loads(_rust_cargo_run_handler(args))
        assert "error" not in result
        assert result["success"] is True
        assert "Cargo OK" in result["stdout"]

    def test_cargo_test(self):
        args = {
            "files": {
                "Cargo.toml": '[package]\nname = "test"\nversion = "0.1.0"\nedition = "2021"\n',
                "src/main.rs": 'fn main() {}\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn it_works() { assert_eq!(2 + 2, 4); }\n}',
            },
            "command": "test",
        }
        result = json.loads(_rust_cargo_run_handler(args))
        assert "error" not in result
        assert result["success"] is True
        assert "test result: ok" in result["stdout"]

    def test_path_traversal_rejection(self):
        args = {
            "files": {
                "../evil.rs": "fn main() {}",
            },
            "command": "run",
        }
        result = json.loads(_rust_cargo_run_handler(args))
        assert "error" in result or result["success"] is False

    def test_empty_files(self):
        args = {"files": {}, "command": "run"}
        result = json.loads(_rust_cargo_run_handler(args))
        assert "error" in result


# =========================================================================
# rust_version
# =========================================================================

class TestRustVersion:
    def test_rust_version(self):
        result = json.loads(_rust_version_handler({}))
        assert "error" not in result
        assert "rustc_version" in result
        assert "cargo_version" in result
        assert "available" in result
        assert result["available"] == RUST_AVAILABLE

    def test_rust_version_when_not_available(self, monkeypatch):
        monkeypatch.setattr("tools.rust_execution_tool.shutil.which", lambda x: None)
        result = json.loads(_rust_version_handler({}))
        assert result["available"] is False
