"""Cross-domain Gemini Flash-Lite topic gatekeeper with structured JSON output."""

from __future__ import annotations

import logging

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.config import Settings
from app.controller.state import normalize_topic_id
from app.schemas.events import TopicCandidate, TranscriptSegment
from app.topics.prefilter import filter_segments

logger = logging.getLogger(__name__)


class _ExtractedTopic(BaseModel):
    canonical_id: str = Field(
        description="Lowercase hyphenated slug for deduplication, e.g. 'etcd' or 'saas-ltv'."
    )
    display_name: str = Field(
        description="Clean human-readable topic title, e.g. 'etcd' or 'LTV (Customer Lifetime Value)'."
    )
    category: str = Field(
        description="Concise domain/category label, e.g. 'Distributed Systems', 'SaaS Metrics', 'Biotech'."
    )
    search_query: str = Field(
        description="Clean search query capturing domain context for lookup or RAG."
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Common abbreviations or alternate names for this topic.",
    )


class _TopicExtractionResponse(BaseModel):
    topics: list[_ExtractedTopic] = Field(default_factory=list)


_SYSTEM_PROMPT = """You are a passive, real-time topic gatekeeper listening to a live voice call.
Your job is to extract ONLY specialized, technical, or domain-specific concepts, tools, protocols,
frameworks, methodologies, regulations, metrics, or acronyms from the NEW UTTERANCES.

Rules:
1. Broad Specialized Scope: Cover ANY specialized domain (software/systems, finance, biotech/medicine,
   hardware, legal/compliance, data/analytics, science, engineering).
2. Strict Filtering: Ignore everyday conversational English, small talk, scheduling, attendee names,
   and generic verbs/nouns. Return an empty list if no specialized topic is introduced.
3. Context Disambiguation & ASR Repair: Use the CONVERSATION CONTEXT to infer the active domain
   (e.g., 'LTV' in SaaS = Customer Lifetime Value vs. Real Estate = Loan-to-Value) and repair
   phonetic speech-to-text errors (e.g., 'et c d' -> 'etcd', 'e bit da' -> 'EBITDA').
4. Deduplication: Do NOT return any topic that matches or aliases an ID in ALREADY COVERED.
"""


def _format_segments(segments: list[TranscriptSegment]) -> str:
    if not segments:
        return "(none)"
    return "\n".join(f"[{seg.speaker.upper()}] {seg.text}" for seg in segments)


class GeminiTopicDetector:
    """Two-stage topic detector: Python pre-filter followed by Gemini Flash-Lite."""

    def __init__(
        self, settings: Settings, client: genai.Client | None = None
    ) -> None:
        self._settings = settings
        self._client = client
        self._warned_missing_key = False

    def _get_client(self) -> genai.Client | None:
        if self._client is not None:
            return self._client
        if not self._settings.has_valid_gemini_key:
            if not self._warned_missing_key:
                logger.warning(
                    "GEMINI_API_KEY is not configured; topic detection is disabled."
                )
                self._warned_missing_key = True
            return None
        self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client

    async def detect(
        self,
        new_segments: list[TranscriptSegment],
        context_window: list[TranscriptSegment],
        excluded_ids: list[str],
    ) -> list[TopicCandidate]:
        filtered = filter_segments(new_segments)
        if not filtered:
            return []

        client = self._get_client()
        if client is None:
            return []

        prompt = (
            f"ALREADY COVERED (exclude): {', '.join(excluded_ids) if excluded_ids else '(none)'}\n\n"
            f"CONVERSATION CONTEXT:\n{_format_segments(context_window)}\n\n"
            f"NEW UTTERANCES TO INSPECT:\n{_format_segments(filtered)}"
        )

        response = await client.aio.models.generate_content(
            model=self._settings.topic_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=400,
                response_mime_type="application/json",
                response_schema=_TopicExtractionResponse,
            ),
        )

        parsed = getattr(response, "parsed", None)
        if parsed is None and getattr(response, "text", None):
            parsed = _TopicExtractionResponse.model_validate_json(response.text)
        if not isinstance(parsed, _TopicExtractionResponse):
            return []

        source_id = filtered[-1].segment_id
        excluded_set = {normalize_topic_id(x) for x in excluded_ids}
        candidates: list[TopicCandidate] = []
        for item in parsed.topics:
            canon = normalize_topic_id(item.canonical_id or item.display_name)
            if canon in excluded_set:
                continue
            candidates.append(
                TopicCandidate(
                    canonical_id=canon,
                    display_name=item.display_name.strip(),
                    category=item.category.strip() or "General",
                    search_query=item.search_query.strip() or item.display_name.strip(),
                    aliases=item.aliases,
                    source_segment_id=source_id,
                )
            )
        return candidates
