export type Speaker = "you" | "others";
export type TopicStatus = "researching" | "complete" | "error";

export interface SystemStatusEvent {
  type: "system.status";
  listening: boolean;
  mic_device: string | null;
  loopback_device: string | null;
  gpu_active: boolean;
  persist_session: boolean;
}

export interface TranscriptPartialEvent {
  type: "transcript.partial";
  segment_id: string;
  speaker: Speaker;
  text: string;
  timestamp: number;
}

export interface TranscriptFinalEvent {
  type: "transcript.final";
  segment_id: string;
  speaker: Speaker;
  text: string;
  timestamp: number;
}

export interface TopicDetectedEvent {
  type: "topic.detected";
  topic_id: string;
  display_name: string;
  category: string;
  source_segment_id: string | null;
}

export interface TopicSummaryEvent {
  type: "topic.summary";
  topic_id: string;
  bullets: string[];
  latency_ms: number;
}

export interface TopicErrorEvent {
  type: "topic.error";
  topic_id: string;
  message: string;
}

export interface SessionClearedEvent {
  type: "session.cleared";
}

export type ServerEvent =
  | SystemStatusEvent
  | TranscriptPartialEvent
  | TranscriptFinalEvent
  | TopicDetectedEvent
  | TopicSummaryEvent
  | TopicErrorEvent
  | SessionClearedEvent;

export type ClientCommand =
  | { type: "session.toggle"; listening: boolean }
  | { type: "session.clear" }
  | { type: "settings.update"; persist_session?: boolean }
  | { type: "topic.manual_lookup"; text: string; context_segment_id?: string | null }
  | { type: "topic.dismiss"; topic_id: string; allow_redetect?: boolean }
  | { type: "devices.select"; mic_id?: string | null; loopback_id?: string | null };

export interface TranscriptItem {
  segment_id: string;
  speaker: Speaker;
  text: string;
  timestamp: number;
  is_final: boolean;
}

export interface TopicCardItem {
  topic_id: string;
  display_name: string;
  category: string;
  status: TopicStatus;
  bullets: string[];
  source_segment_id: string | null;
  latency_ms: number | null;
  error_message: string | null;
  pinned: boolean;
  created_at: number;
}

export interface AudioDeviceInfo {
  device_id: string;
  index: number;
  name: string;
  max_input_channels: number;
  default_sample_rate: number;
  is_loopback: boolean;
  is_default: boolean;
}

export interface DevicesResponse {
  mics: AudioDeviceInfo[];
  loopbacks: AudioDeviceInfo[];
}

export interface DesktopBridge {
  isDesktop: boolean;
  setAlwaysOnTop: (enabled: boolean) => Promise<boolean>;
  setStealthMode: (enabled: boolean) => Promise<boolean>;
}

declare global {
  interface Window {
    desktopBridge?: DesktopBridge;
  }
}
