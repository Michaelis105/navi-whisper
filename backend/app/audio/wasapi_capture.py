"""Windows WASAPI dual-stream capture (Mic + System Loopback) and stateful 16kHz resampler."""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.schemas.events import Speaker

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class AudioDeviceInfo:
    """Metadata for a Windows WASAPI input or loopback audio endpoint."""

    device_id: str
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: int
    is_loopback: bool
    is_default: bool = False


@dataclass(frozen=True)
class RawAudioChunk:
    """Raw 16kHz mono float32 frame emitted from a capture stream."""

    speaker: Speaker
    samples_16k: np.ndarray
    timestamp: float


class StatefulResampler:
    """Downmixes multi-channel PCM and resamples cleanly to 16 kHz mono float32.

    Maintains fractional sample phase and boundary sample continuity across callbacks
    to prevent inter-chunk clicks that could trigger false VAD activations.
    """

    def __init__(
        self,
        source_rate: int,
        channels: int,
        smart_single_mic: bool = False,
        target_rate: int = TARGET_SAMPLE_RATE,
    ) -> None:
        if source_rate <= 0:
            raise ValueError(f"Invalid source_rate: {source_rate}")
        if channels <= 0:
            raise ValueError(f"Invalid channel count: {channels}")

        self.source_rate = int(source_rate)
        self.channels = int(channels)
        self.smart_single_mic = smart_single_mic
        self.target_rate = int(target_rate)
        self._step = self.source_rate / float(self.target_rate)

        # Stateful continuity across chunk boundaries
        self._phase: float = 0.0
        self._prev_sample: float | None = None

    def reset(self) -> None:
        self._phase = 0.0
        self._prev_sample = None

    def _downmix_to_mono(self, pcm: np.ndarray) -> np.ndarray:
        if self.channels == 1:
            return pcm.astype(np.float32, copy=False)

        frames = pcm.reshape(-1, self.channels).astype(np.float32, copy=False)
        if self.smart_single_mic and self.channels == 2 and len(frames) > 0:
            # USB XLR interfaces (e.g., UA Volt, Focusrite) often expose 2 channels
            # where Mic 1 is on Left and Right is near-zero silence. Avoid -6 dB loss.
            energy_l = float(np.dot(frames[:, 0], frames[:, 0]))
            energy_r = float(np.dot(frames[:, 1], frames[:, 1]))
            total = energy_l + energy_r
            if total > 1e-8:
                if energy_l > 0.92 * total:
                    return np.ascontiguousarray(frames[:, 0])
                if energy_r > 0.92 * total:
                    return np.ascontiguousarray(frames[:, 1])

        return np.mean(frames, axis=1, dtype=np.float32)

    def process(self, interleaved_float32: np.ndarray) -> np.ndarray:
        """Convert interleaved float32 samples into 16 kHz mono float32."""
        if interleaved_float32.size == 0:
            return np.empty(0, dtype=np.float32)

        mono = self._downmix_to_mono(interleaved_float32)
        n_in = len(mono)
        if n_in == 0:
            return np.empty(0, dtype=np.float32)

        if self.source_rate == self.target_rate:
            return np.clip(mono, -1.0, 1.0)

        # Fast exact 3:1 anti-aliased decimation path for 48 kHz -> 16 kHz
        if self.source_rate == 48000 and self.target_rate == 16000:
            return self._resample_fractional(mono, anti_alias_taps=3)

        # General fractional resampling (e.g., 44.1 kHz -> 16 kHz, 96 kHz -> 16 kHz)
        taps = max(1, int(round(self._step)))
        return self._resample_fractional(mono, anti_alias_taps=min(taps, 6))

    def _resample_fractional(
        self, mono: np.ndarray, anti_alias_taps: int = 1
    ) -> np.ndarray:
        # Prepend previous sample so both anti-aliasing and linear interpolation
        # remain continuous across chunk boundaries.
        if self._prev_sample is None:
            extended = np.concatenate(([mono[0]], mono))
        else:
            extended = np.concatenate(([self._prev_sample], mono))

        self._prev_sample = float(mono[-1])

        if anti_alias_taps >= 2 and len(extended) >= anti_alias_taps:
            kernel = np.ones(anti_alias_taps, dtype=np.float32) / float(anti_alias_taps)
            # Pad left edge so output length matches `extended`
            padded = np.pad(extended, (anti_alias_taps - 1, 0), mode="edge")
            filtered = np.convolve(padded, kernel, mode="valid").astype(np.float32, copy=False)
        else:
            filtered = extended

        # Valid input coordinates in `mono` space are [0, len(mono)).
        # In `filtered` (which has 1 prepended element at index 0), mono[0] is at index 1.0,
        # and _prev_sample is at index 0.0. Thus phase in [-1, len(mono)-1) maps to [0, len(mono)).
        max_pos = float(len(mono))
        if self._phase >= max_pos:
            self._phase -= max_pos
            return np.empty(0, dtype=np.float32)

        positions = np.arange(self._phase, max_pos, self._step, dtype=np.float64)
        if positions.size == 0:
            self._phase -= max_pos
            return np.empty(0, dtype=np.float32)

        # Advance phase relative to next chunk start
        self._phase = float(positions[-1] + self._step - max_pos)

        # Shift by +1.0 to index into `filtered` (where index 0 is _prev_sample)
        shifted = positions + 1.0
        idx_floor = np.floor(shifted).astype(np.int64)
        frac = (shifted - idx_floor).astype(np.float32)
        idx_ceil = np.minimum(idx_floor + 1, len(filtered) - 1)
        idx_floor = np.clip(idx_floor, 0, len(filtered) - 1)

        out = filtered[idx_floor] * (1.0 - frac) + filtered[idx_ceil] * frac
        return np.clip(out.astype(np.float32, copy=False), -1.0, 1.0)


def list_wasapi_devices() -> dict[str, list[AudioDeviceInfo]]:
    """Enumerate Windows WASAPI microphone inputs and render loopback devices."""
    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    try:
        wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        wasapi_index = int(wasapi_info["index"])
        default_in_idx = int(wasapi_info.get("defaultInputDevice", -1))

        default_loopback_idx = -1
        try:
            lb_info = pa.get_default_wasapi_loopback()
            default_loopback_idx = int(lb_info["index"])
        except Exception:
            pass

        mics: list[AudioDeviceInfo] = []
        loopbacks: list[AudioDeviceInfo] = []

        for idx in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(idx)
            if int(dev.get("hostApi", -1)) != wasapi_index:
                continue
            max_ch = int(dev.get("maxInputChannels", 0))
            if max_ch <= 0:
                continue
            is_lb = bool(dev.get("isLoopbackDevice", False))
            info = AudioDeviceInfo(
                device_id=str(idx),
                index=idx,
                name=str(dev.get("name", f"Device {idx}")),
                max_input_channels=max_ch,
                default_sample_rate=int(round(float(dev.get("defaultSampleRate", 48000)))),
                is_loopback=is_lb,
                is_default=(idx == default_loopback_idx if is_lb else idx == default_in_idx),
            )
            if is_lb:
                loopbacks.append(info)
            else:
                mics.append(info)

        return {"mics": mics, "loopbacks": loopbacks}
    finally:
        pa.terminate()


class WASAPIDualCapture:
    """Non-blocking dual-channel WASAPI audio capture for Mic ('you') + Loopback ('others')."""

    def __init__(self, chunk_Queue_max: int = 512) -> None:
        self._queue: queue.Queue[RawAudioChunk] = queue.Queue(maxsize=chunk_Queue_max)
        self._pa: Any = None
        self._streams: list[Any] = []
        self._running = False

        self._selected_mic_id: str | None = None
        self._selected_loopback_id: str | None = None
        self.mic_device_name: str | None = None
        self.loopback_device_name: str | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    def drain_chunks(self, max_items: int = 64) -> list[RawAudioChunk]:
        """Drain available 16kHz PCM chunks without blocking."""
        items: list[RawAudioChunk] = []
        for _ in range(max_items):
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return items

    def set_preferred_devices(
        self, mic_id: str | None = None, loopback_id: str | None = None
    ) -> None:
        if mic_id is not None:
            self._selected_mic_id = mic_id or None
        if loopback_id is not None:
            self._selected_loopback_id = loopback_id or None

    def start(self) -> None:
        """Open WASAPI Microphone and Loopback streams."""
        if self._running:
            return

        import pyaudiowpatch as pyaudio

        self._pa = pyaudio.PyAudio()
        try:
            mic_dev = self._resolve_mic_device(self._pa)
            lb_dev = self._resolve_loopback_device(self._pa)

            if mic_dev is not None:
                self._open_channel_stream(
                    pyaudio_mod=pyaudio,
                    dev_info=mic_dev,
                    speaker="you",
                    smart_single_mic=True,
                )
                self.mic_device_name = str(mic_dev.get("name"))
            else:
                self.mic_device_name = None
                logger.warning("No WASAPI microphone input device available.")

            if lb_dev is not None:
                self._open_channel_stream(
                    pyaudio_mod=pyaudio,
                    dev_info=lb_dev,
                    speaker="others",
                    smart_single_mic=False,
                )
                self.loopback_device_name = str(lb_dev.get("name"))
            else:
                self.loopback_device_name = None
                logger.warning("No WASAPI loopback render device available.")

            self._running = True
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        """Close active streams and release PortAudio resources cleanly."""
        self._running = False
        for stream in self._streams:
            try:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
            except Exception:
                logger.debug("Ignored error while closing WASAPI stream", exc_info=True)
        self._streams.clear()

        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                logger.debug("Ignored error terminating PyAudio", exc_info=True)
            self._pa = None

        # Clear stale queued audio on stop
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _resolve_mic_device(self, pa: Any) -> dict[str, Any] | None:
        import pyaudiowpatch as pyaudio

        if self._selected_mic_id is not None:
            try:
                return pa.get_device_info_by_index(int(self._selected_mic_id))
            except Exception:
                logger.warning("Requested mic_id=%s not found; using default.", self._selected_mic_id)

        try:
            wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            idx = int(wasapi.get("defaultInputDevice", -1))
            if idx >= 0:
                return pa.get_device_info_by_index(idx)
        except Exception:
            pass

        try:
            return pa.get_default_input_device_info()
        except Exception:
            return None

    def _resolve_loopback_device(self, pa: Any) -> dict[str, Any] | None:
        if self._selected_loopback_id is not None:
            try:
                return pa.get_device_info_by_index(int(self._selected_loopback_id))
            except Exception:
                logger.warning(
                    "Requested loopback_id=%s not found; using default.",
                    self._selected_loopback_id,
                )

        try:
            return pa.get_default_wasapi_loopback()
        except Exception:
            return None

    def _open_channel_stream(
        self,
        pyaudio_mod: Any,
        dev_info: dict[str, Any],
        speaker: Speaker,
        smart_single_mic: bool,
    ) -> None:
        idx = int(dev_info["index"])
        channels = max(1, min(int(dev_info.get("maxInputChannels", 1)), 2))
        sample_rate = int(round(float(dev_info.get("defaultSampleRate", 48000))))
        # ~32ms callback buffer at source rate
        frames_per_buffer = max(256, int(round(sample_rate * 0.032)))

        resampler = StatefulResampler(
            source_rate=sample_rate,
            channels=channels,
            smart_single_mic=smart_single_mic,
            target_rate=TARGET_SAMPLE_RATE,
        )

        def _callback(in_data: bytes | None, frame_count: int, time_info: Any, status: int):
            if in_data:
                pcm = np.frombuffer(in_data, dtype=np.float32)
                samples_16k = resampler.process(pcm)
                if samples_16k.size > 0:
                    chunk = RawAudioChunk(
                        speaker=speaker,
                        samples_16k=samples_16k,
                        timestamp=time.time(),
                    )
                    try:
                        self._queue.put_nowait(chunk)
                    except queue.Full:
                        try:
                            self._queue.get_nowait()
                            self._queue.put_nowait(chunk)
                        except queue.Empty:
                            pass
            return (None, pyaudio_mod.paContinue)

        stream = self._pa.open(
            format=pyaudio_mod.paFloat32,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=idx,
            frames_per_buffer=frames_per_buffer,
            stream_callback=_callback,
        )
        stream.start_stream()
        self._streams.append(stream)
