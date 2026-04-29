#!/usr/bin/env python3
"""Rust Execution Tool — compile and run Rust code via rustc and cargo.

Registered tools:
- ``rust_compile``   — compile a single .rs file with rustc and run the binary
- ``rust_cargo_run`` — run a cargo project from a provided files dict
- ``rust_version``   — check rustc and cargo versions

Requires the Rust toolchain (rustc + cargo) to be installed and on PATH.
"""

import json
import logging
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_OUTPUT_CHARS = 50_000
_COMPILE_TIMEOUT = 60
_RUN_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_rust_available() -> bool:
    """Return True when rustc and cargo are both in PATH."""
    return shutil.which("rustc") is not None and shutil.which("cargo") is not None


def _trim_output(text: str, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
    """Trim output text to *max_chars*, keeping head and tail."""
    if len(text) <= max_chars:
        return text
    head_chars = int(max_chars * 0.4)
    tail_chars = max_chars - head_chars
    head = text[:head_chars]
    tail = text[-tail_chars:]
    omitted = len(text) - len(head) - len(tail)
    return (
        head
        + f"\n\n... [OUTPUT TRUNCATED - {omitted:,} chars omitted "
        f"out of {len(text):,} total] ...\n\n"
        + tail
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def rust_compile(
    code: str,
    edition: str = "2021",
    args: Optional[List[str]] = None,
    stdin: Optional[str] = None,
) -> str:
    """Compile and run a single Rust source file.

    Args:
        code: Rust source code to compile and run.
        edition: Rust edition (e.g. '2015', '2018', '2021', '2024').
        args: Additional CLI arguments for rustc.
        stdin: Input to feed to the compiled binary via stdin.

    Returns:
        JSON string: {"stdout": "...", "stderr": "...", "success": true, "duration_ms": 123}
        or {"error": "..."}.
    """
    if not code:
        return json.dumps({"error": "Missing required parameter: code"}, ensure_ascii=False)

    tmpdir = tempfile.mkdtemp(prefix="hermes_rust_compile_")
    source_path = os.path.join(tmpdir, "main.rs")
    binary_name = "main.exe" if platform.system() == "Windows" else "main"
    binary_path = os.path.join(tmpdir, binary_name)

    try:
        with open(source_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Build rustc command
        cmd = ["rustc", "--edition", edition, source_path, "-o", binary_path]
        if args:
            cmd.extend(str(a) for a in args)

        logger.info(
            "Running rustc: %s",
            " ".join(shlex.quote(str(c)) for c in cmd),
        )

        compile_start = time.monotonic()
        compile_result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_COMPILE_TIMEOUT,
        )
        compile_duration_ms = round((time.monotonic() - compile_start) * 1000)

        if compile_result.returncode != 0:
            stderr = _trim_output(compile_result.stderr or "")
            return json.dumps(
                {
                    "success": False,
                    "stdout": "",
                    "stderr": stderr,
                    "duration_ms": compile_duration_ms,
                },
                ensure_ascii=False,
            )

        # Run the compiled binary
        run_cmd = [binary_path]
        run_start = time.monotonic()
        run_result = subprocess.run(
            run_cmd,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT,
        )
        run_duration_ms = round((time.monotonic() - run_start) * 1000)

        stdout = _trim_output(run_result.stdout or "")
        stderr = _trim_output(run_result.stderr or "")

        return json.dumps(
            {
                "success": run_result.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "duration_ms": compile_duration_ms + run_duration_ms,
            },
            ensure_ascii=False,
        )

    except subprocess.TimeoutExpired as e:
        logger.warning("rust_compile timed out after %ss", e.timeout)
        return json.dumps(
            {"error": f"Rust compilation or execution timed out after {e.timeout}s"},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.exception("rust_compile failed")
        return json.dumps(
            {"error": f"Rust compilation failed: {type(e).__name__}: {e}"},
            ensure_ascii=False,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def rust_cargo_run(
    files: Dict[str, str],
    command: str = "run",
    args: Optional[List[str]] = None,
    timeout: int = 120,
) -> str:
    """Run a Cargo project from provided files.

    Args:
        files: Mapping of relative file paths to file contents. Must include
               Cargo.toml and source files (e.g. src/main.rs).
        command: Cargo subcommand to run (e.g. 'run', 'build', 'test').
        args: Additional arguments to pass to the cargo command.
        timeout: Maximum execution time in seconds.

    Returns:
        JSON string: {"stdout": "...", "stderr": "...", "success": true, "duration_ms": 123}
        or {"error": "..."}.
    """
    if not files:
        return json.dumps({"error": "Missing required parameter: files"}, ensure_ascii=False)

    tmpdir = tempfile.mkdtemp(prefix="hermes_rust_cargo_")

    try:
        for rel_path, content in files.items():
            # Prevent path traversal outside the temp directory
            if ".." in rel_path or rel_path.startswith("/"):
                return json.dumps(
                    {"error": f"Invalid file path (path traversal attempt): {rel_path}"},
                    ensure_ascii=False,
                )
            full_path = os.path.join(tmpdir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

        cmd = ["cargo", command]
        if args:
            cmd.extend(str(a) for a in args)

        logger.info(
            "Running cargo in %s: %s",
            tmpdir,
            " ".join(shlex.quote(str(c)) for c in cmd),
        )

        start = time.monotonic()
        result = subprocess.run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration_ms = round((time.monotonic() - start) * 1000)

        stdout = _trim_output(result.stdout or "")
        stderr = _trim_output(result.stderr or "")

        return json.dumps(
            {
                "success": result.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "duration_ms": duration_ms,
            },
            ensure_ascii=False,
        )

    except subprocess.TimeoutExpired as e:
        logger.warning("rust_cargo_run timed out after %ss", e.timeout)
        return json.dumps(
            {"error": f"Cargo command timed out after {e.timeout}s"},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.exception("rust_cargo_run failed")
        return json.dumps(
            {"error": f"Cargo execution failed: {type(e).__name__}: {e}"},
            ensure_ascii=False,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def rust_version() -> str:
    """Check installed rustc and cargo versions.

    Returns:
        JSON string: {"rustc_version": "...", "cargo_version": "...", "available": true}
        or {"available": false}.
    """
    rustc = shutil.which("rustc")
    cargo = shutil.which("cargo")

    rustc_version: Optional[str] = None
    cargo_version: Optional[str] = None

    if rustc:
        try:
            result = subprocess.run(
                ["rustc", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                rustc_version = (result.stdout or "").strip() or None
        except Exception as e:
            logger.debug("Failed to get rustc version: %s", e)

    if cargo:
        try:
            result = subprocess.run(
                ["cargo", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                cargo_version = (result.stdout or "").strip() or None
        except Exception as e:
            logger.debug("Failed to get cargo version: %s", e)

    return json.dumps(
        {
            "rustc_version": rustc_version,
            "cargo_version": cargo_version,
            "available": bool(rustc and cargo),
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Handlers (args, **kw) -> str
# ---------------------------------------------------------------------------

def _handle_rust_compile(args: dict, **kw) -> str:
    return rust_compile(
        code=args.get("code", ""),
        edition=args.get("edition", "2021"),
        args=args.get("args"),
        stdin=args.get("stdin"),
    )


def _handle_rust_cargo_run(args: dict, **kw) -> str:
    return rust_cargo_run(
        files=args.get("files", {}),
        command=args.get("command", "run"),
        args=args.get("args"),
        timeout=args.get("timeout", 120),
    )


def _handle_rust_version(args: dict, **kw) -> str:
    return rust_version()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

RUST_COMPILE_SCHEMA = {
    "name": "rust_compile",
    "description": (
        "Compile and run a single Rust source file using rustc. "
        "The source code is written to a temporary file, compiled, and the resulting binary is executed. "
        "Returns stdout, stderr, success status, and execution duration."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Rust source code to compile and run.",
            },
            "edition": {
                "type": "string",
                "description": "Rust edition to use (e.g., '2015', '2018', '2021', '2024'). Defaults to '2021'.",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional CLI arguments to pass to rustc. Optional.",
            },
            "stdin": {
                "type": "string",
                "description": "Input to provide to the compiled binary via stdin. Optional.",
            },
        },
        "required": ["code"],
    },
}

RUST_CARGO_RUN_SCHEMA = {
    "name": "rust_cargo_run",
    "description": (
        "Run a Cargo project from a set of provided files. "
        "Files are written to a temporary directory and the specified cargo command is executed. "
        "Use this for multi-file Rust projects or when dependencies are needed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "files": {
                "type": "object",
                "description": (
                    "Dictionary mapping file paths (relative to project root) to their contents. "
                    "Must include at least Cargo.toml and src/main.rs (or equivalent)."
                ),
            },
            "command": {
                "type": "string",
                "description": "Cargo command to run (e.g., 'run', 'build', 'test'). Defaults to 'run'.",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional arguments to pass to the cargo command. Optional.",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum execution time in seconds. Defaults to 120.",
            },
        },
        "required": ["files"],
    },
}

RUST_VERSION_SCHEMA = {
    "name": "rust_version",
    "description": (
        "Check the installed versions of rustc and cargo. "
        "Use this to verify that the Rust toolchain is available before attempting compilation."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="rust_compile",
    toolset="rust_execution",
    schema=RUST_COMPILE_SCHEMA,
    handler=_handle_rust_compile,
    check_fn=_check_rust_available,
    description="Compile and run a single Rust source file",
    emoji="⚙️",
    max_result_size_chars=100_000,
)

registry.register(
    name="rust_cargo_run",
    toolset="rust_execution",
    schema=RUST_CARGO_RUN_SCHEMA,
    handler=_handle_rust_cargo_run,
    check_fn=_check_rust_available,
    description="Run a Cargo project from provided files",
    emoji="📦",
    max_result_size_chars=100_000,
)

registry.register(
    name="rust_version",
    toolset="rust_execution",
    schema=RUST_VERSION_SCHEMA,
    handler=_handle_rust_version,
    check_fn=_check_rust_available,
    description="Check rustc and cargo versions",
    emoji="🦀",
    max_result_size_chars=10_000,
)
