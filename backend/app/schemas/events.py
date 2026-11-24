"""Pydantic schemas for internal pipeline events and WebSocket contracts."""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

Speaker = Literal["you", "others"]
TopicStatus = Literal["researching", "complete", "error"]


# --- Internal Domain Models ---


class TranscriptSegment(BaseModel):
    segment_id: str = Field(default_factory=lambda: f"seg_{uuid.uuid4().hex[:10]}")
    speaker: Speaker
    text: str
    timestamp: float = Field(default_factory=time.time)
    is_final: bool = True


class TopicCandidate(BaseModel):
    canonical_id: str
    display_name: str
    category: str = "General"
    search_query: str = ""
    aliases: list[str] = Field(default_factory=list)
    source_segment_id: str | None = None


class TopicCard(BaseModel):
    topic_id: str
    display_name: str
    category: str
    status: TopicStatus = "researching"
    bullets: list[str] = Field(default_factory=list)
    source_segment_id: str | None = None
    latency_ms: int | None = None
    error_message: str | None = None
    created_at: float = Field(default_factory=time.time)


# --- Server -> Client WebSocket Events ---


class SystemStatusEvent(BaseModel):
    type: Literal["system.status"] = "system.status"
    listening: bool
    mic_device: str | None = None
    loopback_device: str | None = None
    gpu_active: bool = False
    persist_session: bool = False


class TranscriptPartialEvent(BaseModel):
    type: Literal["transcript.partial"] = "transcript.partial"
    segment_id: str
    speaker: Speaker
    text: str
    timestamp: float


class TranscriptFinalEvent(BaseModel):
    type: Literal["transcript.final"] = "transcript.final"
    segment_id: str
    speaker: Speaker
    text: str
    timestamp: float


class TopicDetectedEvent(BaseModel):
    type: Literal["topic.detected"] = "topic.detected"
    topic_id: str
    display_name: str
    category: str
    source_segment_id: str | None = None


class TopicSummaryEvent(BaseModel):
    type: Literal["topic.summary"] = "topic.summary"
    topic_id: str
    bullets: list[str]
    latency_ms: int


class TopicErrorEvent(BaseModel):
    type: Literal["topic.error"] = "topic.error"
    topic_id: str
    message: str


class SessionClearedEvent(BaseModel):
    type: Literal["session.cleared"] = "session.cleared"


ServerEvent = Union[
    SystemStatusEvent,
    TranscriptPartialEvent,
    TranscriptFinalEvent,
    TopicDetectedEvent,
    TopicSummaryEvent,
    TopicErrorEvent,
    SessionClearedEvent,
]


# --- Client -> Server WebSocket Commands ---


class SessionToggleCommand(BaseModel):
    type: Literal["session.toggle"]
    listening: bool


class SessionClearCommand(BaseModel):
    type: Literal["session.clear"]


class SettingsUpdateCommand(BaseModel):
    type: Literal["settings.update"]
    persist_session: bool | None = None


class TopicManualLookupCommand(BaseModel):
    type: Literal["topic.manual_lookup"]
    text: str = Field(min_length=1, max_length=200)
    context_segment_id: str | None = None


class TopicDismissCommand(BaseModel):
    type: Literal["topic.dismiss"]
    topic_id: str
    allow_redetect: bool = False


class DevicesSelectCommand(BaseModel):
    type: Literal["devices.select"]
    mic_id: str | None = None
    loopback_id: str | None = None


ClientCommand = Annotated[
    Union[
        SessionToggleCommand,
        SessionClearCommand,
        SettingsUpdateCommand,
        TopicManualLookupCommand,
        TopicDismissCommand,
        DevicesSelectCommand,
    ],
    Field(discriminator="type"),
]


class DevUtteranceRequest(BaseModel):
    speaker: Speaker = "others"
    text: str = Field(min_length=1)
    is_final: bool = True
