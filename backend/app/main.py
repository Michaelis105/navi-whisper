"""FastAPI entrypoint, WebSocket relay, and health/device/dev endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.audio.pipeline import AudioPipelineManager
from app.config import Settings, get_settings, register_cuda_dll_directories
from app.controller.orchestrator import BackendController
from app.llm.gemini_summary import GeminiSummaryService
from app.rag.retriever import NoOpRetriever
from app.schemas.events import DevUtteranceRequest, TranscriptSegment
from app.topics.gemini_detector import GeminiTopicDetector


def create_app(
    settings: Settings | None = None,
    controller: BackendController | None = None,
    enable_audio_pipeline: bool | None = None,
) -> FastAPI:
    """Create and configure the Navi-Whisper FastAPI application."""
    register_cuda_dll_directories()
    active_settings = settings or get_settings()
    audio_pipeline: AudioPipelineManager | None = None

    if controller is not None:
        active_controller = controller
    else:
        detector = GeminiTopicDetector(active_settings)
        summary_service = GeminiSummaryService(
            active_settings, retriever=NoOpRetriever()
        )
        active_controller = BackendController(
            active_settings,
            topic_detector=detector,
            summary_service=summary_service,
        )
        if enable_audio_pipeline is not False:
            audio_pipeline = AudioPipelineManager(active_settings, active_controller)
            active_controller.audio_control = audio_pipeline

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.controller = active_controller
        app.state.audio_pipeline = audio_pipeline
        await active_controller.start()
        if audio_pipeline is not None:
            await audio_pipeline.start()
        try:
            yield
        finally:
            if audio_pipeline is not None:
                await audio_pipeline.stop()
            await active_controller.stop()

    app = FastAPI(title="Navi-Whisper Backend", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.controller = active_controller
    app.state.audio_pipeline = audio_pipeline

    @app.get("/health")
    async def health() -> dict[str, object]:
        ctrl: BackendController = app.state.controller
        return {
            "status": "ok",
            "listening": ctrl.listening,
            "gpu_active": ctrl.gpu_active,
            "mic_device": ctrl.mic_device,
            "loopback_device": ctrl.loopback_device,
            "persist_session": ctrl.persist_session,
            "gemini_configured": active_settings.has_valid_gemini_key,
        }

    @app.get("/api/devices")
    async def get_devices() -> dict[str, object]:
        pipeline: AudioPipelineManager | None = app.state.audio_pipeline
        if pipeline is None:
            return {"mics": [], "loopbacks": []}
        return pipeline.get_available_devices()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        ctrl: BackendController = app.state.controller
        await ctrl.connect(websocket)
        try:
            while True:
                raw_text = await websocket.receive_text()
                await ctrl.handle_raw_command(raw_text)
        except WebSocketDisconnect:
            ctrl.disconnect(websocket)
        except Exception:
            ctrl.disconnect(websocket)

    if active_settings.enable_dev_routes:

        @app.post("/api/dev/utterance")
        async def inject_dev_utterance(body: DevUtteranceRequest) -> dict[str, str]:
            ctrl: BackendController = app.state.controller
            segment = TranscriptSegment(
                speaker=body.speaker,
                text=body.text,
                is_final=body.is_final,
            )
            await ctrl.ingest_transcript(segment)
            return {"status": "queued", "segment_id": segment.segment_id}

    return app


app = create_app()
