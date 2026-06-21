"""
Automatic narration generation.

Turns a visual generation prompt ("a lone astronaut walking across a
red Martian dune at sunset, cinematic wide shot") into spoken narration
text. Two real strategies are wired up, selected by availability:

  1. If an Anthropic API key is configured, calls Claude to write a
     proper narration line — genuinely better prose than rule-based
     generation, and this app already runs inside infrastructure that
     has the `anthropic` SDK available.
  2. Otherwise, falls back to a deterministic template expansion so the
     feature still works fully offline with zero external dependencies
     (relevant for an airgapped RunPod template).
"""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)


def _fallback_narration(prompt: str) -> str:
    """
    Deterministic, dependency-free narration synthesis: strips
    camera/technical jargon that sounds odd read aloud ("4k", "cinematic
    wide shot", "85mm lens") and reflows the remainder into a short
    spoken sentence. Not as fluent as an LLM rewrite, but fully
    self-contained and instant.
    """
    jargon_pattern = re.compile(
        r"\b(4k|8k|hdr|cinematic|wide shot|close-?up|tracking shot|dolly|"
        r"\d+mm lens|f/\d+(\.\d+)?|depth of field|bokeh|ultra-?realistic|"
        r"hyper-?realistic|octane render|unreal engine|trending on artstation)\b",
        re.IGNORECASE,
    )
    cleaned = jargon_pattern.sub("", prompt)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.-")
    if not cleaned:
        cleaned = prompt.strip()
    cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else cleaned
    if not cleaned.endswith((".", "!", "?")):
        cleaned += "."
    return cleaned


def _llm_narration(prompt: str) -> str | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Write a single short, natural-sounding spoken narration "
                        "line (one to two sentences, no stage directions, no "
                        "quotation marks) for a video clip described as:\n\n"
                        f"{prompt}\n\nReturn only the narration text."
                    ),
                }
            ],
        )
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        return text or None
    except Exception as exc:  # network, auth, rate-limit, etc.
        logger.warning("LLM narration generation failed, falling back to template: %s", exc)
        return None


def generate_narration(prompt: str) -> str:
    text = _llm_narration(prompt)
    if text:
        return text
    return _fallback_narration(prompt)
