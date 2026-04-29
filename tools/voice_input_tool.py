#!/usr/bin/env python3
"""Voice Input Tool — transcribe audio files to text.

Supports two backends:
1. OpenAI Whisper API (cloud) — requires OPENAI_API_KEY.
2. Local ``whisper`` CLI (openai-whisper / faster-whisper / whisper.cpp) —
   auto-detected via ``shutil.which("whisper")``.

Priority:
- OPENAI_API_KEY present → use OpenAI Whisper API.
- ``whisper`` CLI in PATH → use local CLI.
- Neither → tool is unavailable.

Registered tools:
- ``voice_transcribe``  — transcribe an audio file to text
- ``voice_list_models`` — list available STT models
"""

import json
import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_OPENAI_WHISPER_MODELS = [
    "whisper-1",
]

_LOCAL_WHISPER_MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "turbo",
]


def _has_openai_api() -> bool:
    """Return True when OPENAI_API_KEY is present."""
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def _has_local_whisper() -> bool:
    """Return True when a ``whisper`` executable is in PATH."""
    return shutil.which("whisper") is not None


def _get_backend() -> str:
    """Determine which STT backend to use."""
    if _has_openai_api():
        return "openai"
    if _has_local_whisper():
        return "local"
    return "none"


def _check_voice_available() -> bool:
    """Return True when at least one STT backend is available."""
    return _get_backend() != "none"


# ---------------------------------------------------------------------------
# OpenAI Whisper API backend
# ---------------------------------------------------------------------------

def _transcribe_openai(audio_path: str, model: str = "whisper-1") -> Dict[str, Any]:
    """Transcribe using OpenAI Whisper API.

    Raises RuntimeError on API or network failure.
    """
    try:
        import openai
    except ImportError as e:
        raise RuntimeError("openai package is not installed: pip install openai") from e

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = openai.OpenAI(api_key=api_key)
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        client.base_url = base_url

    audio_file = Path(audio_path)
    if not audio_file.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with audio_file.open("rb") as f:
        response = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="json",
        )

    # Response is either a dict or an API object with a text attribute
    if hasattr(response, "text"):
        return {"text": response.text, "model": model, "backend": "openai"}
    if isinstance(response, dict):
        return {"text": response.get("text", ""), "model": model, "backend": "openai"}
    return {"text": str(response), "model": model, "backend": "openai"}


# ---------------------------------------------------------------------------
# Local whisper CLI backend
# ---------------------------------------------------------------------------

def _transcribe_local(
    audio_path: str,
    model: str = "base",
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe using local ``whisper`` CLI.

    Supports openai-whisper, faster-whisper, and whisper.cpp wrappers
    that expose a ``whisper`` CLI with ``--model`` and ``--output_format json``.

    Raises RuntimeError on transcription failure.
    """
    whisper_exe = shutil.which("whisper")
    if not whisper_exe:
        raise RuntimeError("Local whisper CLI not found in PATH")

    audio_file = Path(audio_path)
    if not audio_file.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Build command
    cmd = [
        whisper_exe,
        str(audio_file),
        "--model", model,
        "--output_format", "json",
        "--output_dir", str(audio_file.parent),
    ]
    if language:
        cmd.extend(["--language", language])

    logger.info("Running local whisper: %s", " ".join(shlex.quote(str(c)) for c in cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        err = (result.stderr or "").strip() or "Unknown error"
        raise RuntimeError(f"whisper CLI failed (exit {result.returncode}): {err}")

    # whisper CLI writes <audio_name>.json next to the audio file
    json_path = audio_file.with_suffix(".json")
    if not json_path.is_file():
        # Some wrappers write to a different naming scheme — look for any .json in the same dir
        candidates = list(audio_file.parent.glob(f"{audio_file.stem}*.json"))
        if candidates:
            json_path = candidates[0]
        else:
            raise RuntimeError(
                f"whisper CLI did not produce expected JSON output: {json_path}"
            )

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to read whisper JSON output: {e}") from e

    # Extract text — openai-whisper writes {"text": "...", "segments": [...]}
    text = data.get("text", "").strip() if isinstance(data, dict) else ""
    return {
        "text": text,
        "model": model,
        "backend": "local",
        "segments": data.get("segments") if isinstance(data, dict) else None,
    }


# ---------------------------------------------------------------------------
# Public tool implementations
# ---------------------------------------------------------------------------

def voice_transcribe(
    audio_path: str,
    model: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """Transcribe an audio file to text.

    Args:
        audio_path: Path to the audio file (wav, mp3, m4a, ogg, flac, etc.).
        model: Model to use. For OpenAI: only "whisper-1" is supported.
               For local: one of tiny/base/small/medium/large/turbo (default: base).
        language: ISO-639-1 language code (e.g. "en", "es", "fr"). Optional.

    Returns:
        JSON string: {"text": "...", "model": "...", "backend": "..."}
        or {"error": "..."}.
    """
    if not audio_path:
        return json.dumps({"error": "Missing required parameter: audio_path"}, ensure_ascii=False)

    backend = _get_backend()
    if backend == "none":
        return json.dumps(
            {
                "error": (
                    "No STT backend available. Set OPENAI_API_KEY for cloud Whisper, "
                    "or install a local whisper CLI (openai-whisper, faster-whisper, whisper.cpp)."
                )
            },
            ensure_ascii=False,
        )

    try:
        if backend == "openai":
            effective_model = model if model else "whisper-1"
            result = _transcribe_openai(audio_path, model=effective_model)
        else:
            effective_model = model if model else "base"
            result = _transcribe_local(audio_path, model=effective_model, language=language)

        return json.dumps(result, ensure_ascii=False)
    except FileNotFoundError as e:
        logger.warning("voice_transcribe file not found: %s", e)
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    except TimeoutError as e:
        logger.warning("voice_transcribe timed out: %s", e)
        return json.dumps({"error": f"Transcription timed out: {e}"}, ensure_ascii=False)
    except Exception as e:
        logger.exception("voice_transcribe failed")
        return json.dumps(
            {"error": f"Transcription failed: {type(e).__name__}: {e}"},
            ensure_ascii=False,
        )


def voice_list_models() -> str:
    """List available STT models for the currently active backend.

    Returns:
        JSON string: {"models": [...], "backend": "openai|local|none"}.
    """
    backend = _get_backend()
    if backend == "openai":
        models = _OPENAI_WHISPER_MODELS
    elif backend == "local":
        models = _LOCAL_WHISPER_MODELS
    else:
        models = []

    return json.dumps(
        {
            "models": models,
            "backend": backend,
            "note": (
                "Set OPENAI_API_KEY for cloud Whisper, "
                "or install a local whisper CLI for local STT."
                if backend == "none"
                else None
            ),
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Handlers (args, **kw) -> str
# ---------------------------------------------------------------------------

def _handle_voice_transcribe(args: dict, **kw) -> str:
    return voice_transcribe(
        audio_path=args.get("audio_path", ""),
        model=args.get("model"),
        language=args.get("language"),
    )


def _handle_voice_list_models(args: dict, **kw) -> str:
    return voice_list_models()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

VOICE_TRANSCRIBE_SCHEMA = {
    "name": "voice_transcribe",
    "description": (
        "Transcribe an audio file to text using OpenAI Whisper API (cloud) "
        "or a local whisper CLI. Supports wav, mp3, m4a, ogg, flac, and more. "
        "Requires OPENAI_API_KEY for cloud mode, or a local whisper installation "
        "(openai-whisper, faster-whisper, whisper.cpp) for offline mode."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "audio_path": {
                "type": "string",
                "description": "Path to the audio file to transcribe.",
            },
            "model": {
                "type": "string",
                "description": (
                    "Model to use. For OpenAI: 'whisper-1'. "
                    "For local: 'tiny', 'base', 'small', 'medium', 'large', 'turbo'. "
                    "Defaults to 'whisper-1' (OpenAI) or 'base' (local)."
                ),
            },
            "language": {
                "type": "string",
                "description": "ISO-639-1 language code, e.g. 'en', 'es', 'fr'. Optional.",
            },
        },
        "required": ["audio_path"],
    },
}

VOICE_LIST_MODELS_SCHEMA = {
    "name": "voice_list_models",
    "description": (
        "List available speech-to-text models for the currently active backend. "
        "Use this to discover which models can be passed to voice_transcribe."
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
    name="voice_transcribe",
    toolset="voice",
    schema=VOICE_TRANSCRIBE_SCHEMA,
    handler=_handle_voice_transcribe,
    check_fn=_check_voice_available,
    requires_env=["OPENAI_API_KEY"],
    emoji="🎙️",
    max_result_size_chars=50_000,
)

registry.register(
    name="voice_list_models",
    toolset="voice",
    schema=VOICE_LIST_MODELS_SCHEMA,
    handler=_handle_voice_list_models,
    check_fn=_check_voice_available,
    requires_env=["OPENAI_API_KEY"],
    emoji="🎙️",
    max_result_size_chars=10_000,
)
