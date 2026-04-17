from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import CandidateConfig


def _load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        # Prefer the repo-local `.env` because shell sessions in this workspace
        # may already carry placeholder provider keys.
        os.environ[key] = value


def _clean_answer(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^answer\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    return cleaned


@dataclass
class OpenAIMemoryBackend:
    model: str
    reasoning_effort: str = "medium"
    max_output_tokens: int = 96
    verbosity: str = "low"
    prompt_profile: str = "default"
    refusal_text: str = "I don't know."

    def __post_init__(self) -> None:
        _load_env_file()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for llm.enabled=true. Install it with `python -m pip install openai`."
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to the environment or `.env`.")
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def answer(
        self,
        *,
        question: str,
        evidence_text: str,
        heuristic_answer: Optional[str] = None,
    ) -> str:
        system_prompt = self._system_prompt()
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Heuristic draft:\n{heuristic_answer or '(none)'}\n\n"
            f"Evidence:\n{evidence_text}\n"
        )

        request = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            "max_output_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort and self.reasoning_effort.lower() != "none":
            request["reasoning"] = {"effort": self.reasoning_effort}
        if self.verbosity:
            request["text"] = {"verbosity": self.verbosity}

        response = self._client.responses.create(**request)
        text = getattr(response, "output_text", "") or ""
        if not text and getattr(response, "output", None):
            chunks = []
            for item in response.output:
                for content in (getattr(item, "content", None) or []):
                    piece = getattr(content, "text", None)
                    if hasattr(piece, "value"):
                        piece = piece.value
                    if piece:
                        chunks.append(piece)
            text = "\n".join(chunks)
        return _clean_answer(text) or self.refusal_text

    def _system_prompt(self) -> str:
        if self.prompt_profile == "memeval_short_answer":
            return (
                "You answer questions about conversation memory using only the supplied evidence.\n"
                "Rules:\n"
                "1. Prefer the heuristic draft if the evidence supports it.\n"
                "2. If the draft is wrong or incomplete, correct it using the evidence.\n"
                "3. Give the shortest answer possible, usually 1-8 words.\n"
                "4. Preserve exact dates, times, names, and numbers when supported.\n"
                "5. For yes/no questions, answer with `yes` or `no` when possible.\n"
                "6. If the evidence is insufficient, return exactly `None`.\n"
                "7. Return only the final answer string. Do not explain your reasoning."
            )
        return (
            "You answer memory questions using only the supplied evidence.\n"
            "Rules:\n"
            "1. Prefer the heuristic draft if the evidence supports it.\n"
            "2. If the draft is wrong or incomplete, correct it using the evidence.\n"
            "3. For updates, prefer the newest active fact.\n"
            "4. For yes/no questions, answer with `yes` or `no` when possible.\n"
            "5. For timing questions, preserve exact phrases like `Around turn 15.` if supported.\n"
            "6. Return only the final short answer. Do not explain your reasoning.\n"
            f"7. If the evidence is insufficient, return `{self.refusal_text}`"
        )


@dataclass
class MockMemoryBackend:
    mode: str = "echo_heuristic"
    answer_text: str = "I don't know."

    def answer(
        self,
        *,
        question: str,
        evidence_text: str,
        heuristic_answer: Optional[str] = None,
    ) -> str:
        del question, evidence_text
        if self.mode == "echo_heuristic" and heuristic_answer:
            return heuristic_answer
        return self.answer_text


def build_llm_backend(candidate: CandidateConfig):
    if not candidate.get("llm", "enabled", default=False):
        return None
    provider = str(candidate.get("llm", "provider", default="openai")).lower()
    if provider == "mock":
        return MockMemoryBackend(
            mode=str(candidate.get("llm", "mock_mode", default="echo_heuristic")),
            answer_text=str(candidate.get("llm", "mock_answer", default="I don't know.")),
        )
    if provider != "openai":
        raise ValueError(f"Unsupported llm.provider: {provider}")
    return OpenAIMemoryBackend(
        model=str(candidate.get("llm", "model", default="gpt-5.2")),
        reasoning_effort=str(candidate.get("llm", "reasoning_effort", default="medium")),
        max_output_tokens=int(candidate.get("llm", "max_output_tokens", default=96)),
        verbosity=str(candidate.get("llm", "verbosity", default="low")),
        prompt_profile=str(candidate.get("llm", "prompt_profile", default="default")),
        refusal_text=str(candidate.get("llm", "refusal_text", default="I don't know.")),
    )
