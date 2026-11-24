"""Guarded Gemini Flash summary service with future RAG context injection."""

from __future__ import annotations

import logging

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.config import Settings
from app.llm.guardrails import enforce_summary_guardrails
from app.rag.retriever import ContextRetriever, NoOpRetriever, RetrievedChunk
from app.schemas.events import TopicCandidate, TranscriptSegment

logger = logging.getLogger(__name__)


class _SummaryResponse(BaseModel):
    bullets: list[str] = Field(min_length=1, max_length=3)


_SYSTEM_PROMPT = """You are a silent real-time reference assistant for a person on a live call.
Given a detected specialized topic, recent conversation context, and optional reference chunks,
produce the most succinct, high-signal 1 to 3 bullet summary possible.

Strict rules:
1. Return 1 to 3 bullets ONLY. Zero introductory or closing prose.
2. Each bullet MUST be <= 18 words, crisp, and plain-English.
3. Bullet 1: What it is in this domain.
4. Bullet 2: Core mechanism, trade-off, or practical role.
5. Bullet 3 (optional): Why it matters in the current conversation snippet.
6. If reference chunks are provided, ground facts in them without citing chunk IDs.
"""


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(none)"
    return "\n".join(f"- [{c.title}] {c.content}" for c in chunks)


def _format_context(segments: list[TranscriptSegment]) -> str:
    if not segments:
        return "(none)"
    return "\n".join(f"[{s.speaker.upper()}] {s.text}" for s in segments[-8:])


class GeminiSummaryService:
    """Generates guarded 1-3 bullet summaries via Gemini Flash."""

    def __init__(
        self,
        settings: Settings,
        retriever: ContextRetriever | None = None,
        client: genai.Client | None = None,
    ) -> None:
        self._settings = settings
        self._retriever: ContextRetriever = retriever or NoOpRetriever()
        self._client = client

    def _get_client(self) -> genai.Client:
        if self._client is not None:
            return self._client
        if not self._settings.has_valid_gemini_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured in backend/.env."
            )
        self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client

    async def summarize(
        self,
        candidate: TopicCandidate,
        context_window: list[TranscriptSegment],
    ) -> list[str]:
        query = candidate.search_query or candidate.display_name
        chunks = await self._retriever.retrieve(query, context_window)
        client = self._get_client()

        prompt = (
            f"TOPIC: {candidate.display_name} (Category: {candidate.category})\n"
            f"CONVERSATION CONTEXT:\n{_format_context(context_window)}\n\n"
            f"REFERENCE CHUNKS:\n{_format_chunks(chunks)}"
        )

        response = await client.aio.models.generate_content(
            model=self._settings.summary_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.15,
                max_output_tokens=160,
                response_mime_type="application/json",
                response_schema=_SummaryResponse,
            ),
        )

        parsed = getattr(response, "parsed", None)
        if parsed is None and getattr(response, "text", None):
            parsed = _SummaryResponse.model_validate_json(response.text)
        if not isinstance(parsed, _SummaryResponse):
            return ["Summary unavailable."]

        return enforce_summary_guardrails(parsed.bullets)
