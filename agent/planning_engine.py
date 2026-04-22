#!/usr/bin/env python3
"""
Multi-Agent Planning Engine -- ULTRAPLAN-inspired subagent orchestration.

When the agent detects a complex multi-step task, this engine:
  1. Breaks the task into 1-5 sub-tasks
  2. Delegates each sub-task to a child agent via the existing delegate_tool
  3. Collects results from all subagents
  4. Runs a verification step via an auxiliary LLM
  5. Returns an aggregated result with metadata

Designed to be called conditionally from run_agent.py based on `should_plan()`.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import call_llm, extract_content_or_reasoning
from tools.delegate_tool import delegate_task

logger = logging.getLogger(__name__)

# Heuristic thresholds for complexity detection
_MIN_WORDS_FOR_PLANNING = 50
_MAX_WORDS_FOR_PLANNING = 5000  # sanity upper bound
_VERIFICATION_MAX_TOKENS = 1500
_PLANNING_MAX_TOKENS = 1500


class PlanningEngine:
    """Orchestrates subagent planning, execution, and verification."""

    def __init__(self, parent_agent: Any = None):
        """
        Args:
            parent_agent: The parent AIAgent instance used for delegation.
                         Required for delegate_task to work.
        """
        self.parent_agent = parent_agent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_plan(self, task_description: str) -> bool:
        """Heuristic to decide if a task is complex enough to warrant planning.

        Signals:
          - Length > 100 words (or > 50 with multiple verbs)
          - Mentions multiple files/directories
          - Contains multiple distinct action verbs
          - Has explicit sequencing words (first, then, finally, step)
          - Contains numbered or bulleted sub-tasks
        """
        if not task_description or not isinstance(task_description, str):
            return False

        text = task_description.strip()
        word_count = len(text.split())

        # Very short tasks never need planning
        if word_count < _MIN_WORDS_FOR_PLANNING:
            return False

        # Simple single-action heuristic: count action verbs
        action_verbs = re.findall(
            r"\b(create|write|build|implement|refactor|migrate|convert|"
            r"update|delete|remove|add|fix|debug|test|deploy|analyze|"
            r"research|compare|integrate|configure|install|generate|"
            r"extract|transform|summarize|review|audit|optimize)\b",
            text, re.IGNORECASE,
        )

        # File/path mentions
        file_mentions = re.findall(
            r"\b[\w\-]+\.(py|js|ts|json|yaml|yml|md|txt|sql|sh|rs|go|java|cpp|c|h)\b",
            text, re.IGNORECASE,
        )
        path_mentions = re.findall(r"(?:[\w\-]+/)+[\w\-]+", text)

        # Explicit sequencing signals
        sequencing_signals = re.findall(
            r"\b(first|second|third|then|next|afterward|finally|step \d+|"
            r"phase|stage|part \d+)\b",
            text, re.IGNORECASE,
        )

        score = 0
        if word_count > 100:
            score += 2
        elif word_count > _MIN_WORDS_FOR_PLANNING:
            score += 1

        if len(action_verbs) >= 2:
            score += 2
        if len(file_mentions) + len(path_mentions) >= 2:
            score += 2
        if len(sequencing_signals) >= 1:
            score += 2

        # Require at least a score of 3 to trigger planning
        return score >= 3

    def plan_and_execute(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a complex task, break it into sub-tasks, delegate each to a
        subagent, collect results, verify consistency, and return aggregated output.

        Args:
            task_description: The high-level goal.
            context: Optional dict with extra context (e.g. file paths,
                     current directory, previous errors).

        Returns:
            Dict with keys:
              - plan_steps: list of step dicts {index, description}
              - subagent_results: list of raw result dicts from delegate_task
              - verification: {status, summary, issues}
              - final_response: str -- human-readable aggregated result
              - metadata: {total_duration_seconds, num_subagents, task_description}
        """
        if not self.parent_agent:
            return self._error_result(
                task_description, "PlanningEngine requires a parent_agent for delegation."
            )

        context = context or {}

        # 1. Analyze and break into sub-tasks
        try:
            plan_steps = self._break_into_subtasks(task_description, context)
        except Exception as exc:
            logger.exception("Planning step failed")
            return self._error_result(task_description, f"Planning failed: {exc}")

        if not plan_steps:
            # Fallback: single sub-task with the original description
            plan_steps = [{"index": 0, "description": task_description}]

        # 2. Delegate each sub-task
        try:
            subagent_results = self._delegate_steps(plan_steps, context)
        except Exception as exc:
            logger.exception("Delegation step failed")
            return self._error_result(
                task_description, f"Delegation failed: {exc}", plan_steps=plan_steps
            )

        # 3. Run verification step
        try:
            verification = self._verify_results(task_description, plan_steps, subagent_results)
        except Exception as exc:
            logger.exception("Verification step failed")
            verification = {
                "status": "error",
                "summary": f"Verification failed: {exc}",
                "issues": [str(exc)],
            }

        # 4. Build aggregated response
        final_response = self._aggregate_response(plan_steps, subagent_results, verification)

        return {
            "plan_steps": plan_steps,
            "subagent_results": subagent_results,
            "verification": verification,
            "final_response": final_response,
            "metadata": {
                "task_description": task_description,
                "num_subagents": len(plan_steps),
                "num_completed": sum(
                    1 for r in subagent_results if r.get("status") == "completed"
                ),
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _break_into_subtasks(
        self,
        task_description: str,
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Use an auxiliary LLM to break the task into sub-tasks (1-5 steps)."""
        system_prompt = (
            "You are a planning assistant. Break the user's task into 1 to 5 clear, "
            "self-contained sub-tasks. Each sub-task should be specific enough for a "
            "subagent to execute independently. Respond ONLY with a JSON array of objects, "
            "each with 'index' (0-based int) and 'description' (string). No markdown, no prose."
        )
        user_prompt = (
            f"Task: {task_description}\n\n"
            f"Context: {json.dumps(context, ensure_ascii=False)}\n\n"
            "Break this into sub-tasks and return JSON: [{\"index\": 0, \"description\": \"...\"}, ...]"
        )

        response = call_llm(
            task="compression",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=_PLANNING_MAX_TOKENS,
        )
        text = extract_content_or_reasoning(response).strip()

        # Try to extract JSON from fenced or raw text
        parsed = self._extract_json(text)
        if not parsed or not isinstance(parsed, list):
            # Fallback: heuristic split by sentences with action verbs
            logger.debug("LLM planning returned non-JSON; falling back to heuristic split")
            return self._heuristic_split(task_description)

        steps = []
        for item in parsed:
            if isinstance(item, dict) and "description" in item:
                steps.append({
                    "index": item.get("index", len(steps)),
                    "description": str(item["description"]).strip(),
                })
        return steps[:5]

    def _delegate_steps(
        self,
        plan_steps: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Delegate each plan step to a subagent via delegate_task."""
        if len(plan_steps) == 1:
            # Single task -- use simple goal mode
            result_json = delegate_task(
                goal=plan_steps[0]["description"],
                context=json.dumps(context, ensure_ascii=False),
                parent_agent=self.parent_agent,
            )
        else:
            # Batch mode -- build tasks array
            tasks = []
            for step in plan_steps:
                tasks.append({
                    "goal": step["description"],
                    "context": json.dumps(context, ensure_ascii=False),
                })
            result_json = delegate_task(
                tasks=tasks,
                parent_agent=self.parent_agent,
            )

        # Parse delegate_task JSON output
        try:
            parsed = json.loads(result_json)
        except json.JSONDecodeError as exc:
            logger.warning("delegate_task returned invalid JSON: %s", exc)
            return []

        results = parsed.get("results", [])
        # Ensure each result has expected keys
        for r in results:
            if "status" not in r:
                r["status"] = "unknown"
            if "summary" not in r:
                r["summary"] = ""
        return results

    def _verify_results(
        self,
        task_description: str,
        plan_steps: List[Dict[str, Any]],
        subagent_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Use an auxiliary LLM to verify consistency and correctness."""
        summaries = []
        for i, result in enumerate(subagent_results):
            status = result.get("status", "unknown")
            summary = result.get("summary", "") or ""
            summaries.append(f"Step {i}: [{status}] {summary[:500]}")

        system_prompt = (
            "You are a verification assistant. Review the subagent results below and check:\n"
            "1. Are the results internally consistent?\n"
            "2. Did all steps complete successfully?\n"
            "3. Is anything missing or contradictory?\n"
            "4. Summarize the overall outcome.\n\n"
            "Respond with a JSON object: {\"status\": \"pass\" | \"warn\" | \"fail\", "
            "\"summary\": \"...\", \"issues\": [\"...\", ...]}. No markdown."
        )
        user_prompt = (
            f"Original task: {task_description}\n\n"
            f"Plan steps:\n" + "\n".join(
                f"{s['index']}: {s['description']}" for s in plan_steps
            ) + "\n\n"
            f"Subagent results:\n" + "\n".join(summaries) + "\n\n"
            "Provide verification JSON only."
        )

        response = call_llm(
            task="compression",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=_VERIFICATION_MAX_TOKENS,
        )
        text = extract_content_or_reasoning(response).strip()
        parsed = self._extract_json(text)

        if isinstance(parsed, dict) and "status" in parsed:
            return {
                "status": str(parsed.get("status", "unknown")).lower(),
                "summary": str(parsed.get("summary", "")),
                "issues": list(parsed.get("issues", [])) if isinstance(parsed.get("issues"), list) else [],
            }

        # Fallback: treat raw text as summary
        return {
            "status": "warn",
            "summary": text or "Verification produced unstructured output.",
            "issues": [],
        }

    def _aggregate_response(
        self,
        plan_steps: List[Dict[str, Any]],
        subagent_results: List[Dict[str, Any]],
        verification: Dict[str, Any],
    ) -> str:
        """Build a human-readable final response."""
        lines = ["### Plan Execution Summary", ""]

        for i, step in enumerate(plan_steps):
            result = subagent_results[i] if i < len(subagent_results) else {}
            status = result.get("status", "unknown")
            icon = "[OK]" if status == "completed" else "[ERR]" if status in ("failed", "error") else "[WARN]"
            lines.append(f"{icon} **Step {i + 1}:** {step['description']}")
            summary = (result.get("summary") or "No output").strip()
            lines.append(f"   Result: {summary[:300]}{'...' if len(summary) > 300 else ''}")
            lines.append("")

        lines.append("---")
        lines.append(f"**Verification:** {verification.get('status', 'unknown').upper()}")
        if verification.get("summary"):
            lines.append(f"_Summary:_ {verification['summary']}")
        if verification.get("issues"):
            lines.append("_Issues:_")
            for issue in verification["issues"]:
                lines.append(f"  - {issue}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

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

    @staticmethod
    def _heuristic_split(task_description: str) -> List[Dict[str, Any]]:
        """Fallback: split long task by sentences containing action verbs."""
        sentences = re.split(r"(?<=[.!?])\s+", task_description.strip())
        steps = []
        for i, sent in enumerate(sentences[:5]):
            if re.search(
                r"\b(create|write|build|implement|refactor|migrate|convert|"
                r"update|delete|remove|add|fix|debug|test|deploy|analyze|"
                r"research|compare|integrate|configure|install|generate|"
                r"extract|transform|summarize|review|audit|optimize)\b",
                sent, re.IGNORECASE,
            ):
                steps.append({"index": i, "description": sent.strip()})
        if not steps:
            steps = [{"index": 0, "description": task_description.strip()}]
        return steps

    @staticmethod
    def _error_result(
        task_description: str,
        error: str,
        plan_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return {
            "plan_steps": plan_steps or [],
            "subagent_results": [],
            "verification": {"status": "fail", "summary": error, "issues": [error]},
            "final_response": f"Planning engine failed: {error}",
            "metadata": {
                "task_description": task_description,
                "num_subagents": 0,
                "num_completed": 0,
            },
        }
