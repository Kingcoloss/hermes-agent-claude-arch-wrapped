#!/usr/bin/env python3
"""
Auto Memory Extraction -- extracts key facts after each conversation turn
and stores them in a lightweight per-session memory store.

Uses the auxiliary client with a fast/cheap model for lightweight extraction.
Stores facts as timestamped JSON entries in ~/.hermes/memories/<session_id>.jsonl
with per-session deduplication.

Designed to be called from run_agent.py after each assistant turn.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from agent.auxiliary_client import call_llm, extract_content_or_reasoning
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_EXTRACTION_MAX_TOKENS = 800
_EXTRACTION_TEMPERATURE = 0.1
_DEFAULT_MEMORY_DIR = get_hermes_home() / "memories"


class MemoryExtractor:
    """Extracts and stores key facts from conversation turns."""

    def __init__(self, memory_dir: Optional[Path] = None):
        """
        Args:
            memory_dir: Directory to store memory JSONL files.
                        Defaults to ~/.hermes/memories/
        """
        self.memory_dir = Path(memory_dir) if memory_dir else _DEFAULT_MEMORY_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        # Per-instance cache of seen hashes to avoid disk reads
        self._session_hash_cache: Dict[str, Set[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_and_store(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract key facts from a conversation turn and store them.

        Args:
            session_id: The current session identifier.
            user_message: The user's input for this turn.
            assistant_response: The assistant's response for this turn.

        Returns:
            List of memory entry dicts that were stored (empty if none or deduped).
        """
        if not session_id:
            logger.debug("MemoryExtractor: skipping empty session_id")
            return []

        # Build extraction prompt
        try:
            extracted_facts = self._extract_facts(user_message, assistant_response)
        except Exception as exc:
            logger.warning("Memory extraction LLM call failed: %s", exc)
            return []

        if not extracted_facts:
            logger.debug("MemoryExtractor: no facts extracted for session %s", session_id)
            return []

        # Load existing hashes for deduplication
        existing_hashes = self._load_session_hashes(session_id)

        stored: List[Dict[str, Any]] = []
        for fact in extracted_facts:
            fact_hash = self._hash_fact(fact)
            if fact_hash in existing_hashes:
                logger.debug("MemoryExtractor: deduped fact %s", fact_hash[:8])
                continue

            entry = {
                "session_id": session_id,
                "timestamp": time.time(),
                "fact": fact,
                "fact_hash": fact_hash,
            }
            self._append_memory(session_id, entry)
            existing_hashes.add(fact_hash)
            stored.append(entry)

        # Update cache
        self._session_hash_cache[session_id] = existing_hashes
        logger.debug("MemoryExtractor: stored %d new facts for session %s", len(stored), session_id)
        return stored

    def get_memories(
        self,
        session_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve stored memories for a session, newest first."""
        path = self._memory_path(session_id)
        if not path.exists():
            return []

        entries: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            logger.warning("MemoryExtractor: failed to read %s: %s", path, exc)
            return []

        # Sort by timestamp descending, limit
        entries.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
        return entries[:limit]

    def clear_memories(self, session_id: str) -> bool:
        """Delete all stored memories for a session."""
        path = self._memory_path(session_id)
        if path.exists():
            try:
                path.unlink()
                self._session_hash_cache.pop(session_id, None)
                return True
            except OSError as exc:
                logger.warning("MemoryExtractor: failed to delete %s: %s", path, exc)
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_facts(self, user_message: str, assistant_response: str) -> List[str]:
        """Call the auxiliary LLM to extract discrete facts."""
        system_prompt = (
            "You are a fact extraction assistant. Given a user message and an "
            "assistant response, extract 0 to 5 concise, declarative facts that "
            "would be useful to remember later.\n\n"
            "Rules:\n"
            "- Each fact must be self-contained (no pronouns like 'he', 'it', 'this').\n"
            "- Focus on user preferences, important decisions, project structure, "
            "  credentials, constraints, or confirmed facts.\n"
            "- Skip trivial greetings, chit-chat, or obvious general knowledge.\n"
            "- Respond ONLY with a JSON array of strings. No markdown, no prose.\n"
            "- If nothing is worth remembering, return []."
        )
        user_prompt = (
            f"User message:\n{user_message}\n\n"
            f"Assistant response:\n{assistant_response}\n\n"
            "Extract facts as JSON array: [\"fact 1\", \"fact 2\", ...]"
        )

        response = call_llm(
            task="compression",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=_EXTRACTION_TEMPERATURE,
            max_tokens=_EXTRACTION_MAX_TOKENS,
        )
        text = extract_content_or_reasoning(response).strip()

        parsed = self._extract_json(text)
        if isinstance(parsed, list):
            return [str(f).strip() for f in parsed if str(f).strip()]

        # Fallback: try line-split if JSON parsing fails
        lines = [l.strip().lstrip("- ").strip('"') for l in text.splitlines() if l.strip()]
        return [l for l in lines if l and not l.startswith("[") and not l.startswith("]")]

    def _memory_path(self, session_id: str) -> Path:
        """Sanitize session_id and return the JSONL file path."""
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
        return self.memory_dir / f"{safe_id}.jsonl"

    def _append_memory(self, session_id: str, entry: Dict[str, Any]) -> None:
        """Append a single memory entry to the session's JSONL file."""
        path = self._memory_path(session_id)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("MemoryExtractor: failed to write %s: %s", path, exc)

    def _load_session_hashes(self, session_id: str) -> Set[str]:
        """Load all existing fact hashes for a session."""
        if session_id in self._session_hash_cache:
            return self._session_hash_cache[session_id]

        hashes: Set[str] = set()
        path = self._memory_path(session_id)
        if not path.exists():
            return hashes

        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        h = entry.get("fact_hash")
                        if h:
                            hashes.add(h)
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            logger.warning("MemoryExtractor: failed to read hashes from %s: %s", path, exc)

        self._session_hash_cache[session_id] = hashes
        return hashes

    @staticmethod
    def _hash_fact(fact: str) -> str:
        """Create a stable hash for deduplication."""
        normalized = " ".join(fact.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _extract_json(text: str) -> Optional[Any]:
        """Best-effort JSON extraction from text that may be fenced or raw."""
        if not text:
            return None
        # Try fenced JSON first
        fenced = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass
        # Try raw JSON array or object
        raw = re.search(r"(\[.*?\]|\{.*?\})", text, re.DOTALL)
        if raw:
            try:
                return json.loads(raw.group(1))
            except json.JSONDecodeError:
                pass
        # Try the whole text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        return None
