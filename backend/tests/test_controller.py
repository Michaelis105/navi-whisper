"""Phase 1 unit and integration tests for controller state, sinks, orchestrator, and WS API."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.controller.orchestrator import BackendController
from app.controller.sinks import JsonlMarkdownSink, MemoryOnlySink
from app.controller.state import TopicRegistry, TranscriptRingBuffer
from app.main import create_app
from app.schemas.events import TopicCandidate, TopicCard, TranscriptSegment


def test_ring_buffer_prunes_by_time_and_max_items() -> None:
    buf = TranscriptRingBuffer(window_seconds=30.0, max_items=3)
    buf.append(TranscriptSegment(speaker="others", text="oldest", timestamp=100.0))
    buf.append(TranscriptSegment(speaker="you", text="keep-1", timestamp=120.0))
    buf.append(TranscriptSegment(speaker="others", text="keep-2", timestamp=135.0))

    texts = [s.text for s in buf.snapshot()]
    assert texts == ["keep-1", "keep-2"]

    buf.append(TranscriptSegment(speaker="you", text="keep-3", timestamp=136.0))
    buf.append(TranscriptSegment(speaker="others", text="keep-4", timestamp=137.0))
    assert [s.text for s in buf.snapshot()] == ["keep-2", "keep-3", "keep-4"]


def test_topic_registry_dedup_aliases_and_dismiss() -> None:
    reg = TopicRegistry()
    cand = TopicCandidate(
        canonical_id="Kubernetes",
        display_name="Kubernetes",
        category="Infrastructure",
        aliases=["K8s", "kube"],
    )
    card = reg.register_detected(cand)
    assert card.topic_id == "kubernetes"
    assert reg.is_seen("k8s")
    assert reg.is_seen("KUBE")
    assert not reg.is_seen("etcd")

    # Dismiss without allow_redetect keeps topic suppressed
    assert reg.dismiss("k8s", allow_redetect=False) is True
    assert reg.is_seen("kubernetes")

    # Dismiss with allow_redetect clears suppression
    reg.register_detected(cand)
    assert reg.dismiss("kubernetes", allow_redetect=True) is True
    assert not reg.is_seen("kubernetes")


async def test_sinks_memory_and_disk(tmp_path: Path) -> None:
    mem = MemoryOnlySink()
    seg = TranscriptSegment(speaker="others", text="Testing memory sink", timestamp=1.0)
    card = TopicCard(
        topic_id="etcd",
        display_name="etcd",
        category="Distributed Systems",
        status="complete",
        bullets=["Distributed key-value store."],
    )
    await mem.on_transcript(seg)
    await mem.on_topic_card(card)
    assert list(tmp_path.iterdir()) == []

    disk = JsonlMarkdownSink(tmp_path, session_id="test-session")
    await disk.on_transcript(seg)
    await disk.on_topic_card(card)

    transcript_file = tmp_path / "test-session" / "transcript.jsonl"
    summary_file = tmp_path / "test-session" / "summary.md"
    assert transcript_file.exists()
    assert summary_file.exists()
    assert "Testing memory sink" in transcript_file.read_text(encoding="utf-8")
    assert "## etcd (Distributed Systems)" in summary_file.read_text(encoding="utf-8")


class _StubDetector:
    async def detect(
        self,
        new_segments: list[TranscriptSegment],
        context_window: list[TranscriptSegment],
        excluded_ids: list[str],
    ) -> list[TopicCandidate]:
        combined = " ".join(s.text for s in new_segments).lower()
        if "etcd" in combined and "etcd" not in excluded_ids:
            return [
                TopicCandidate(
                    canonical_id="etcd",
                    display_name="etcd",
                    category="Distributed Systems",
                    search_query="etcd raft consensus",
                    source_segment_id=new_segments[-1].segment_id,
                )
            ]
        return []


class _StubSummaryService:
    async def summarize(
        self, candidate: TopicCandidate, context_window: list[TranscriptSegment]
    ) -> list[str]:
        return [
            f"{candidate.display_name} is a strongly consistent distributed key-value store.",
            "Uses the Raft consensus algorithm for leader election and replication.",
        ]


def test_websocket_and_orchestrator_end_to_end(tmp_path: Path) -> None:
    settings = Settings(
        topic_debounce_seconds=0.05,
        persist_session=False,
        session_dir=str(tmp_path / "sessions"),
        enable_dev_routes=True,
    )
    controller = BackendController(
        settings=settings,
        topic_detector=_StubDetector(),
        summary_service=_StubSummaryService(),
    )
    app = create_app(settings=settings, controller=controller)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        with client.websocket_connect("/ws") as ws:
            initial = ws.receive_json()
            assert initial["type"] == "system.status"
            assert initial["listening"] is False

            # Toggle listening on
            ws.send_json({"type": "session.toggle", "listening": True})
            status_evt = ws.receive_json()
            assert status_evt["type"] == "system.status"
            assert status_evt["listening"] is True

            # Inject utterance via dev route
            resp = client.post(
                "/api/dev/utterance",
                json={"speaker": "others", "text": "We moved state to etcd.", "is_final": True},
            )
            assert resp.status_code == 200

            events = [ws.receive_json() for _ in range(3)]
            types = [e["type"] for e in events]
            assert types == ["transcript.final", "topic.detected", "topic.summary"]
            assert events[1]["topic_id"] == "etcd"
            assert len(events[2]["bullets"]) == 2

            # Manual lookup bypasses gatekeeper
            ws.send_json({"type": "topic.manual_lookup", "text": "Raft Consensus"})
            manual_detected = ws.receive_json()
            manual_summary = ws.receive_json()
            assert manual_detected["type"] == "topic.detected"
            assert manual_detected["topic_id"] == "raft-consensus"
            assert manual_summary["type"] == "topic.summary"

            # Clear session wipes in-memory state
            ws.send_json({"type": "session.clear"})
            cleared = ws.receive_json()
            assert cleared["type"] == "session.cleared"
            assert controller.store.transcripts == []
            assert controller.store.topics.list_cards() == []
