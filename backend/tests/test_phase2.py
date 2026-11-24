"""Phase 2 tests: deterministic pre-filter, guardrails, RAG seam, and Gemini adapters."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config import Settings
from app.controller.orchestrator import BackendController
from app.llm.gemini_summary import GeminiSummaryService, _SummaryResponse
from app.llm.guardrails import enforce_summary_guardrails
from app.main import create_app
from app.rag.retriever import RetrievedChunk
from app.schemas.events import TopicCandidate, TranscriptSegment
from app.topics.gemini_detector import (
    GeminiTopicDetector,
    _ExtractedTopic,
    _TopicExtractionResponse,
)
from app.topics.prefilter import filter_segments, is_segment_worth_inspecting


def test_prefilter_drops_filler_and_logistics_keeps_specialized() -> None:
    dropped = [
        TranscriptSegment(speaker="you", text="um yeah sounds good"),
        TranscriptSegment(speaker="others", text="Can you see my screen?"),
        TranscriptSegment(speaker="you", text="You're on mute"),
        TranscriptSegment(speaker="others", text="Yeah, okay, right."),
        TranscriptSegment(speaker="you", text="Let's go now"),
    ]
    kept = [
        TranscriptSegment(speaker="others", text="What about LTV?"),
        TranscriptSegment(speaker="you", text="Check gRPC"),
        TranscriptSegment(speaker="others", text="We should review our CRISPR delivery vector."),
        TranscriptSegment(speaker="you", text="Look at the borrow_checker error."),
    ]

    for seg in dropped:
        assert not is_segment_worth_inspecting(seg), f"Expected drop: {seg.text}"
    for seg in kept:
        assert is_segment_worth_inspecting(seg), f"Expected keep: {seg.text}"

    assert len(filter_segments(dropped + kept)) == len(kept)


def test_summary_guardrails_strip_cap_and_truncate() -> None:
    raw = [
        "Here is a summary:",
        "- **etcd** is a strongly consistent key-value store.",
        "2. Uses Raft consensus for leader election and distributed state replication across cluster nodes in production.",
        "* Critical for Kubernetes control-plane metadata.",
        "4) Extra fourth bullet that should be dropped.",
    ]
    result = enforce_summary_guardrails(raw, max_bullets=3, max_words=12)
    assert len(result) == 3
    assert result[0].startswith("**etcd**")
    assert result[1].endswith("...")
    assert len(result[1].split()) <= 12
    assert result[2] == "Critical for Kubernetes control-plane metadata."

    assert enforce_summary_guardrails(["   ", "- "]) == ["No concise summary available."]


class _FakeModels:
    def __init__(self, handler) -> None:
        self._handler = handler
        self.calls: list[dict] = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self._handler(kwargs)


class _FakeClient:
    def __init__(self, handler) -> None:
        self.aio = SimpleNamespace(models=_FakeModels(handler))


async def test_gemini_detector_short_circuits_and_parses_structured_output() -> None:
    def handler(kwargs):
        return SimpleNamespace(
            parsed=_TopicExtractionResponse(
                topics=[
                    _ExtractedTopic(
                        canonical_id="saas-ltv",
                        display_name="LTV (Customer Lifetime Value)",
                        category="SaaS Metrics",
                        search_query="Customer Lifetime Value LTV SaaS",
                        aliases=["LTV"],
                    ),
                    _ExtractedTopic(
                        canonical_id="cac",
                        display_name="CAC",
                        category="SaaS Metrics",
                        search_query="Customer Acquisition Cost",
                    ),
                ]
            )
        )

    client = _FakeClient(handler)
    settings = Settings(gemini_api_key="test-key")
    detector = GeminiTopicDetector(settings, client=client)

    # Pure chatter should not call Gemini at all
    chatter = [TranscriptSegment(speaker="you", text="Yeah sounds good, thanks!")]
    assert await detector.detect(chatter, chatter, excluded_ids=[]) == []
    assert len(client.aio.models.calls) == 0

    # Specialized utterance calls Gemini and respects excluded_ids
    tech = [TranscriptSegment(speaker="others", text="Our LTV to CAC ratio improved.")]
    candidates = await detector.detect(tech, tech, excluded_ids=["cac"])
    assert len(client.aio.models.calls) == 1
    assert [c.canonical_id for c in candidates] == ["saas-ltv"]
    assert candidates[0].source_segment_id == tech[0].segment_id


class _FakeRetriever:
    async def retrieve(self, query: str, context_window: list[TranscriptSegment]):
        return [
            RetrievedChunk(
                source="internal-wiki",
                title="SaaS Finance Glossary",
                content="LTV measures total net margin expected from an account.",
            )
        ]


async def test_gemini_summary_injects_rag_and_enforces_guardrails() -> None:
    def handler(kwargs):
        assert "SaaS Finance Glossary" in kwargs["contents"]
        return SimpleNamespace(
            parsed=_SummaryResponse(
                bullets=[
                    "- Total net profit expected over a customer's lifetime.",
                    "- Usually compared against CAC (aim for >= 3:1).",
                ]
            )
        )

    client = _FakeClient(handler)
    settings = Settings(gemini_api_key="test-key")
    service = GeminiSummaryService(settings, retriever=_FakeRetriever(), client=client)

    cand = TopicCandidate(
        canonical_id="saas-ltv",
        display_name="LTV (Customer Lifetime Value)",
        category="SaaS Metrics",
        search_query="Customer Lifetime Value SaaS",
    )
    bullets = await service.summarize(cand, [])
    assert len(bullets) == 2
    assert bullets[0] == "Total net profit expected over a customer's lifetime."


def test_full_pipeline_with_gemini_adapters() -> None:
    def detect_handler(kwargs):
        return SimpleNamespace(
            parsed=_TopicExtractionResponse(
                topics=[
                    _ExtractedTopic(
                        canonical_id="mrna-lnp",
                        display_name="Lipid Nanoparticles (LNP)",
                        category="Biotech",
                        search_query="Lipid Nanoparticles mRNA delivery",
                    )
                ]
            )
        )

    def summary_handler(kwargs):
        return SimpleNamespace(
            parsed=_SummaryResponse(
                bullets=[
                    "Spherical lipid vesicles that encapsulate and protect mRNA.",
                    "Enable cellular uptake and endosomal escape into cytoplasm.",
                ]
            )
        )

    settings = Settings(
        gemini_api_key="test-key",
        topic_debounce_seconds=0.05,
        enable_dev_routes=True,
    )
    detector = GeminiTopicDetector(settings, client=_FakeClient(detect_handler))
    summary = GeminiSummaryService(settings, client=_FakeClient(summary_handler))
    controller = BackendController(
        settings, topic_detector=detector, summary_service=summary
    )
    app = create_app(settings=settings, controller=controller)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            _ = ws.receive_json()  # system.status
            resp = client.post(
                "/api/dev/utterance",
                json={
                    "speaker": "others",
                    "text": "We switched the mRNA payload to a new LNP formulation.",
                    "is_final": True,
                },
            )
            assert resp.status_code == 200
            events = [ws.receive_json() for _ in range(3)]
            assert [e["type"] for e in events] == [
                "transcript.final",
                "topic.detected",
                "topic.summary",
            ]
            assert events[1]["topic_id"] == "mrna-lnp"
            assert events[1]["category"] == "Biotech"
            assert len(events[2]["bullets"]) == 2
