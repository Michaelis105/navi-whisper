import {
  Cpu,
  EyeOff,
  HardDrive,
  Mic,
  MicOff,
  Pin,
  Radio,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";
import { useState } from "react";
import type { ClientCommand, DevicesResponse, SystemStatusEvent } from "../types/events";

interface HeaderBarProps {
  connected: boolean;
  status: SystemStatusEvent;
  devices: DevicesResponse;
  isDesktop: boolean;
  alwaysOnTop: boolean;
  stealthMode: boolean;
  onToggleAlwaysOnTop: () => void;
  onToggleStealthMode: () => void;
  onSendCommand: (cmd: ClientCommand) => void;
  onRefreshDevices: () => void;
}

export function HeaderBar({
  connected,
  status,
  devices,
  isDesktop,
  alwaysOnTop,
  stealthMode,
  onToggleAlwaysOnTop,
  onToggleStealthMode,
  onSendCommand,
  onRefreshDevices,
}: HeaderBarProps) {
  const [showDevices, setShowDevices] = useState(false);

  const toggleDevicesPopover = () => {
    if (!showDevices) {
      onRefreshDevices();
    }
    setShowDevices((v) => !v);
  };

  return (
    <header className="header-bar">
      <div className="header-left">
        <div className="brand">
          <Radio size={17} className={status.listening ? "pulse-accent" : "muted-icon"} />
          <span className="brand-title">Navi Whisper</span>
        </div>

        <span className={`pill ${connected ? "pill-ok" : "pill-warn"}`}>
          <span className="dot" />
          {connected ? "Connected" : "Reconnecting"}
        </span>

        <span className={`pill ${status.gpu_active ? "pill-gpu" : "pill-muted"}`}>
          <Cpu size={12} />
          {status.gpu_active ? "CUDA Whisper" : "CPU Whisper"}
        </span>

        <div className="device-wrapper">
          <button
            type="button"
            className="btn-ghost btn-sm"
            onClick={toggleDevicesPopover}
            title="Select microphone and system loopback devices"
          >
            <SlidersHorizontal size={14} />
            <span>Audio Devices</span>
          </button>

          {showDevices && (
            <div className="device-popover">
              <div className="device-group">
                <label>Microphone (You)</label>
                <select
                  value=""
                  onChange={(e) => {
                    onSendCommand({ type: "devices.select", mic_id: e.target.value });
                    setShowDevices(false);
                  }}
                >
                  <option value="" disabled>
                    {status.mic_device || "Default Microphone"}
                  </option>
                  {devices.mics.map((m) => (
                    <option key={m.device_id} value={m.device_id}>
                      {m.name} {m.is_default ? "(Default)" : ""}
                    </option>
                  ))}
                </select>
              </div>

              <div className="device-group">
                <label>System Audio Loopback (Others)</label>
                <select
                  value=""
                  onChange={(e) => {
                    onSendCommand({ type: "devices.select", loopback_id: e.target.value });
                    setShowDevices(false);
                  }}
                >
                  <option value="" disabled>
                    {status.loopback_device || "Default Speakers Loopback"}
                  </option>
                  {devices.loopbacks.map((l) => (
                    <option key={l.device_id} value={l.device_id}>
                      {l.name} {l.is_default ? "(Default)" : ""}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="header-actions">
        <button
          type="button"
          className={`btn-primary ${status.listening ? "btn-listening" : ""}`}
          disabled={!connected}
          onClick={() =>
            onSendCommand({ type: "session.toggle", listening: !status.listening })
          }
        >
          {status.listening ? <MicOff size={15} /> : <Mic size={15} />}
          <span>{status.listening ? "Pause Listening" : "Start Listening"}</span>
        </button>

        {isDesktop && (
          <>
            <button
              type="button"
              className={`btn-ghost ${alwaysOnTop ? "btn-active" : ""}`}
              onClick={onToggleAlwaysOnTop}
              title="Keep window always on top of calls"
            >
              <Pin size={14} />
              <span>On Top</span>
            </button>

            <button
              type="button"
              className={`btn-ghost ${stealthMode ? "btn-active" : ""}`}
              onClick={onToggleStealthMode}
              title="Hide Navi Whisper from screen sharing (Content Protection)"
            >
              <EyeOff size={14} />
              <span>Stealth</span>
            </button>
          </>
        )}

        <button
          type="button"
          className={`btn-ghost ${status.persist_session ? "btn-active" : ""}`}
          disabled={!connected}
          onClick={() =>
            onSendCommand({
              type: "settings.update",
              persist_session: !status.persist_session,
            })
          }
          title={
            status.persist_session
              ? "Saving transcript & summaries to sessions/ on disk"
              : "In-memory only (click to enable disk logging)"
          }
        >
          <HardDrive size={14} />
          <span>{status.persist_session ? "Saving" : "Memory"}</span>
        </button>

        <button
          type="button"
          className="btn-ghost"
          disabled={!connected}
          onClick={() => onSendCommand({ type: "session.clear" })}
          title="Clear current session transcript and topic cards"
        >
          <Trash2 size={14} />
          <span>Clear</span>
        </button>
      </div>
    </header>
  );
}
