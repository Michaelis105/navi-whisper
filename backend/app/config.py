"""Application configuration and Windows CUDA DLL path registration."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


def register_cuda_dll_directories() -> list[Path]:
    """Register pip-installed NVIDIA CUDA DLL directories on Windows.

    ctranslate2 / faster-whisper look for cublas64_12.dll and cudnn64_9.dll.
    Registering the site-packages/nvidia/*/bin folders avoids requiring a
    system-wide CUDA Toolkit installation on PATH.
    """
    added: list[Path] = []
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return added

    for base in sys.path:
        nvidia_dir = Path(base) / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        for bin_dir in nvidia_dir.glob("*/bin"):
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))
                added.append(bin_dir)
    return added


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and backend/.env."""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = "YOUR_GEMINI_API_KEY_HERE"
    topic_model: str = "gemini-2.5-flash-lite"
    summary_model: str = "gemini-2.5-flash"

    whisper_model: str = "small.en"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"

    host: str = "127.0.0.1"
    port: int = 8765
    persist_session: bool = False
    session_dir: str = "sessions"
    enable_dev_routes: bool = True

    topic_debounce_seconds: float = 1.5
    context_window_seconds: float = 30.0
    max_concurrent_summaries: int = 2

    @property
    def has_valid_gemini_key(self) -> bool:
        key = self.gemini_api_key.strip()
        return bool(key) and key != "YOUR_GEMINI_API_KEY_HERE"


@lru_cache
def get_settings() -> Settings:
    return Settings()
