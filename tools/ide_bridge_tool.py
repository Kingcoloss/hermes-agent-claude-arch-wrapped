#!/usr/bin/env python3
"""IDE Bridge Tool — remote-control bridge for VS Code / JetBrains via JSON-RPC over TCP.

Communicates with a compatible IDE plugin/extension listening on localhost.
Default port is 9876 (override with HERMES_IDE_PORT env var).

Registered tools:
- ``ide_read_file``     — read a file from the IDE workspace
- ``ide_edit_file``     — replace lines in a file
- ``ide_navigate``      — open/navigate to a position
- ``ide_run_command``   — run a command in the IDE terminal

When no IDE is connected, tools return a clear error message and a
setup hint so the model can fall back to local file/terminal tools.
"""

import json
import logging
import os
import socket
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_PORT = 9876
_TIMEOUT_SECONDS = 10


def _get_ide_host_port() -> tuple[str, int]:
    """Return (host, port) for the IDE bridge TCP connection."""
    raw = os.getenv("HERMES_IDE_PORT", str(_DEFAULT_PORT))
    if ":" in raw:
        host, port_str = raw.rsplit(":", 1)
        return host, int(port_str)
    return "127.0.0.1", int(raw)


# ---------------------------------------------------------------------------
# Low-level JSON-RPC-style transport
# ---------------------------------------------------------------------------

def _send_request(method: str, params: dict) -> dict:
    """Send a JSON request to the IDE and return the parsed response.

    Raises ConnectionRefusedError when no IDE is listening.
    Raises TimeoutError when the IDE does not respond in time.
    Raises OSError for other socket-level failures.
    """
    host, port = _get_ide_host_port()
    payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(_TIMEOUT_SECONDS)
    try:
        sock.connect((host, port))
        # Send length-prefixed message (4-byte big-endian length + JSON bytes)
        msg_len = len(data)
        sock.sendall(msg_len.to_bytes(4, "big") + data)

        # Read response length
        len_bytes = _recv_all(sock, 4)
        resp_len = int.from_bytes(len_bytes, "big")
        if resp_len > 10_000_000:  # 10 MB sanity ceiling
            raise ValueError(f"Response length {resp_len} exceeds sanity ceiling")

        resp_data = _recv_all(sock, resp_len)
        resp = json.loads(resp_data.decode("utf-8", errors="replace"))

        if "error" in resp:
            raise RuntimeError(resp["error"].get("message", str(resp["error"])))
        return resp.get("result", {})
    finally:
        sock.close()


def _recv_all(sock: socket.socket, n: int) -> bytes:
    """Receive exactly *n* bytes from *sock*."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionResetError("IDE closed connection before sending full response")
        buf.extend(chunk)
    return bytes(buf)


def _try_send_request(method: str, params: dict) -> tuple[bool, Any]:
    """Wrap _send_request with user-friendly error handling.

    Returns (success, result_or_error_string).
    """
    host, port = _get_ide_host_port()
    try:
        return True, _send_request(method, params)
    except ConnectionRefusedError:
        logger.debug("IDE bridge refused connection at %s:%s", host, port)
        return False, (
            f"No IDE is listening on {host}:{port}. "
            "Start your IDE with the Hermes bridge plugin enabled, "
            f"or set HERMES_IDE_PORT to the correct port."
        )
    except TimeoutError:
        logger.debug("IDE bridge timed out at %s:%s", host, port)
        return False, f"IDE at {host}:{port} did not respond within {_TIMEOUT_SECONDS}s."
    except OSError as e:
        logger.debug("IDE bridge socket error at %s:%s: %s", host, port, e)
        return False, f"Could not reach IDE at {host}:{port}: {e}"
    except Exception as e:
        logger.exception("IDE bridge request failed")
        return False, f"IDE bridge error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def ide_read_file(path: str) -> str:
    """Read a file from the IDE workspace.

    Args:
        path: Absolute or workspace-relative path to the file.

    Returns:
        JSON string: {"content": "..."} or {"error": "..."}.
    """
    if not path:
        return json.dumps({"error": "Missing required parameter: path"}, ensure_ascii=False)
    ok, result = _try_send_request("readFile", {"path": path})
    if not ok:
        return json.dumps({"error": result}, ensure_ascii=False)
    return json.dumps({"content": result.get("content", "")}, ensure_ascii=False)


def ide_edit_file(
    path: str,
    content: str,
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
) -> str:
    """Replace lines in a file within the IDE workspace.

    Args:
        path: Absolute or workspace-relative path to the file.
        content: New content to insert.
        line_start: 1-based start line for replacement (inclusive).
        line_end: 1-based end line for replacement (inclusive).
            If both line_start and line_end are omitted, the entire file is replaced.

    Returns:
        JSON string: {"success": true} or {"error": "..."}.
    """
    if not path:
        return json.dumps({"error": "Missing required parameter: path"}, ensure_ascii=False)
    params: Dict[str, Any] = {"path": path, "content": content}
    if line_start is not None:
        params["lineStart"] = line_start
    if line_end is not None:
        params["lineEnd"] = line_end
    ok, result = _try_send_request("editFile", params)
    if not ok:
        return json.dumps({"error": result}, ensure_ascii=False)
    return json.dumps({"success": result.get("success", True)}, ensure_ascii=False)


def ide_navigate(path: str, line: Optional[int] = None, column: Optional[int] = None) -> str:
    """Open/navigate to a position in the IDE.

    Args:
        path: Absolute or workspace-relative file path.
        line: 1-based line number.
        column: 1-based column number.

    Returns:
        JSON string: {"success": true} or {"error": "..."}.
    """
    if not path:
        return json.dumps({"error": "Missing required parameter: path"}, ensure_ascii=False)
    params: Dict[str, Any] = {"path": path}
    if line is not None:
        params["line"] = line
    if column is not None:
        params["column"] = column
    ok, result = _try_send_request("navigate", params)
    if not ok:
        return json.dumps({"error": result}, ensure_ascii=False)
    return json.dumps({"success": result.get("success", True)}, ensure_ascii=False)


def ide_run_command(command: str) -> str:
    """Run a command in the IDE integrated terminal.

    Args:
        command: Shell command to execute.

    Returns:
        JSON string: {"output": "...", "exit_code": 0} or {"error": "..."}.
    """
    if not command:
        return json.dumps({"error": "Missing required parameter: command"}, ensure_ascii=False)
    ok, result = _try_send_request("runCommand", {"command": command})
    if not ok:
        return json.dumps({"error": result}, ensure_ascii=False)
    return json.dumps(
        {
            "output": result.get("output", ""),
            "exit_code": result.get("exitCode", 0),
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_ide_available() -> bool:
    """Return True if an IDE bridge is currently reachable."""
    host, port = _get_ide_host_port()
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect((host, port))
        sock.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Handlers (args, **kw) -> str
# ---------------------------------------------------------------------------

def _handle_ide_read_file(args: dict, **kw) -> str:
    return ide_read_file(args.get("path", ""))


def _handle_ide_edit_file(args: dict, **kw) -> str:
    return ide_edit_file(
        path=args.get("path", ""),
        content=args.get("content", ""),
        line_start=args.get("line_start"),
        line_end=args.get("line_end"),
    )


def _handle_ide_navigate(args: dict, **kw) -> str:
    return ide_navigate(
        path=args.get("path", ""),
        line=args.get("line"),
        column=args.get("column"),
    )


def _handle_ide_run_command(args: dict, **kw) -> str:
    return ide_run_command(args.get("command", ""))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

IDE_READ_FILE_SCHEMA = {
    "name": "ide_read_file",
    "description": (
        "Read a file from the IDE workspace. The IDE must be running with the "
        "Hermes bridge plugin. Falls back to read_file if no IDE is connected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or workspace-relative path to the file.",
            },
        },
        "required": ["path"],
    },
}

IDE_EDIT_FILE_SCHEMA = {
    "name": "ide_edit_file",
    "description": (
        "Replace lines in a file within the IDE workspace. If line_start and line_end "
        "are omitted, the entire file is replaced. The IDE must be running with the "
        "Hermes bridge plugin."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or workspace-relative path to the file.",
            },
            "content": {
                "type": "string",
                "description": "New content to insert.",
            },
            "line_start": {
                "type": "integer",
                "description": "1-based start line for replacement (inclusive). Optional.",
            },
            "line_end": {
                "type": "integer",
                "description": "1-based end line for replacement (inclusive). Optional.",
            },
        },
        "required": ["path", "content"],
    },
}

IDE_NAVIGATE_SCHEMA = {
    "name": "ide_navigate",
    "description": (
        "Open/navigate to a file position in the IDE. Useful for showing the user "
        "where changes were made or where to look next."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or workspace-relative file path.",
            },
            "line": {
                "type": "integer",
                "description": "1-based line number. Optional.",
            },
            "column": {
                "type": "integer",
                "description": "1-based column number. Optional.",
            },
        },
        "required": ["path"],
    },
}

IDE_RUN_COMMAND_SCHEMA = {
    "name": "ide_run_command",
    "description": (
        "Run a command in the IDE integrated terminal. The IDE must be running with the "
        "Hermes bridge plugin. Falls back to terminal() if no IDE is connected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute in the IDE terminal.",
            },
        },
        "required": ["command"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="ide_read_file",
    toolset="ide",
    schema=IDE_READ_FILE_SCHEMA,
    handler=_handle_ide_read_file,
    check_fn=_check_ide_available,
    requires_env=["HERMES_IDE_PORT"],
    emoji="🖥️",
    max_result_size_chars=100_000,
)

registry.register(
    name="ide_edit_file",
    toolset="ide",
    schema=IDE_EDIT_FILE_SCHEMA,
    handler=_handle_ide_edit_file,
    check_fn=_check_ide_available,
    requires_env=["HERMES_IDE_PORT"],
    emoji="🖥️",
    max_result_size_chars=50_000,
)

registry.register(
    name="ide_navigate",
    toolset="ide",
    schema=IDE_NAVIGATE_SCHEMA,
    handler=_handle_ide_navigate,
    check_fn=_check_ide_available,
    requires_env=["HERMES_IDE_PORT"],
    emoji="🖥️",
    max_result_size_chars=10_000,
)

registry.register(
    name="ide_run_command",
    toolset="ide",
    schema=IDE_RUN_COMMAND_SCHEMA,
    handler=_handle_ide_run_command,
    check_fn=_check_ide_available,
    requires_env=["HERMES_IDE_PORT"],
    emoji="🖥️",
    max_result_size_chars=50_000,
)
