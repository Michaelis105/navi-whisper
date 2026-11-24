import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ClientCommand,
  DevicesResponse,
  ServerEvent,
  Speaker,
  SystemStatusEvent,
  TopicCardItem,
  TranscriptItem,
} from "../types/events";

function resolveBackendPort(): string {
  if (typeof window !== "undefined") {
    const params = new URLSearchParams(window.location.search);
    const qp = params.get("backendPort");
    if (qp) return qp;
  }
  return import.meta.env.VITE_BACKEND_PORT || "8765";
}

const DEFAULT_STATUS: SystemStatusEvent = {
  type: "system.status",
  listening: false,
  mic_device: null,
  loopback_device: null,
  gpu_active: false,
  persist_session: false,
};

export function useNaviSocket() {
  const port = resolveBackendPort();
  const httpBase = `http://127.0.0.1:${port}`;
  const wsUrl = `ws://127.0.0.1:${port}/ws`;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const retryMsRef = useRef(500);

  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<SystemStatusEvent>(DEFAULT_STATUS);
  const [transcripts, setTranscripts] = useState<TranscriptItem[]>([]);
  const [partials, setPartials] = useState<Partial<Record<Speaker, TranscriptItem>>>({});
  const [topics, setTopics] = useState<TopicCardItem[]>([]);
  const [devices, setDevices] = useState<DevicesResponse>({ mics: [], loopbacks: [] });

  const fetchDevices = useCallback(async () => {
    try {
      const res = await fetch(`${httpBase}/api/devices`);
      if (res.ok) {
        const data = (await res.json()) as DevicesResponse;
        setDevices(data);
      }
    } catch {
      // Ignore transient startup failures; WS reconnect refreshes
    }
  }, [httpBase]);

  const handleServerEvent = useCallback((event: ServerEvent) => {
    switch (event.type) {
      case "system.status":
        setStatus(event);
        break;

      case "transcript.partial":
        setPartials((prev) => ({
          ...prev,
          [event.speaker]: {
            segment_id: event.segment_id,
            speaker: event.speaker,
            text: event.text,
            timestamp: event.timestamp,
            is_final: false,
          },
        }));
        break;

      case "transcript.final":
        setPartials((prev) => {
          const existing = prev[event.speaker];
          if (existing && existing.segment_id === event.segment_id) {
            const next = { ...prev };
            delete next[event.speaker];
            return next;
          }
          return prev;
        });
        setTranscripts((prev) => {
          if (prev.some((t) => t.segment_id === event.segment_id)) {
            return prev;
          }
          return [
            ...prev,
            {
              segment_id: event.segment_id,
              speaker: event.speaker,
              text: event.text,
              timestamp: event.timestamp,
              is_final: true,
            },
          ];
        });
        break;

      case "topic.detected":
        setTopics((prev) => {
          const nextCard: TopicCardItem = {
            topic_id: event.topic_id,
            display_name: event.display_name,
            category: event.category,
            status: "researching",
            bullets: [],
            source_segment_id: event.source_segment_id,
            latency_ms: null,
            error_message: null,
            pinned: prev.find((c) => c.topic_id === event.topic_id)?.pinned ?? false,
            created_at: Date.now(),
          };
          const filtered = prev.filter((c) => c.topic_id !== event.topic_id);
          return [nextCard, ...filtered];
        });
        break;

      case "topic.summary":
        setTopics((prev) =>
          prev.map((card) =>
            card.topic_id === event.topic_id
              ? {
                  ...card,
                  status: "complete",
                  bullets: event.bullets,
                  latency_ms: event.latency_ms,
                  error_message: null,
                }
              : card
          )
        );
        break;

      case "topic.error":
        setTopics((prev) =>
          prev.map((card) =>
            card.topic_id === event.topic_id
              ? {
                  ...card,
                  status: "error",
                  error_message: event.message,
                }
              : card
          )
        );
        break;

      case "session.cleared":
        setTranscripts([]);
        setPartials({});
        setTopics([]);
        break;
    }
  }, []);

  useEffect(() => {
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (disposed) return;
        setConnected(true);
        retryMsRef.current = 500;
        void fetchDevices();
      };

      ws.onmessage = (msg) => {
        try {
          const parsed = JSON.parse(String(msg.data)) as ServerEvent;
          handleServerEvent(parsed);
        } catch {
          // Ignore malformed JSON frames
        }
      };

      ws.onclose = () => {
        if (disposed) return;
        setConnected(false);
        wsRef.current = null;
        const delay = retryMsRef.current;
        retryMsRef.current = Math.min(delay * 1.5, 4000);
        reconnectTimerRef.current = window.setTimeout(connect, delay);
      };

      ws.onerror = () => {
        ws.close();
      };
    };

    connect();

    return () => {
      disposed = true;
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      wsRef.current?.close();
    };
  }, [fetchDevices, handleServerEvent, wsUrl]);

  const sendCommand = useCallback((cmd: ClientCommand) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(cmd));
    }
  }, []);

  const togglePinTopic = useCallback((topicId: string) => {
    setTopics((prev) =>
      prev.map((c) => (c.topic_id === topicId ? { ...c, pinned: !c.pinned } : c))
    );
  }, []);

  const dismissTopic = useCallback(
    (topicId: string, allowRedetect = false) => {
      setTopics((prev) => prev.filter((c) => c.topic_id !== topicId));
      sendCommand({ type: "topic.dismiss", topic_id: topicId, allow_redetect: allowRedetect });
    },
    [sendCommand]
  );

  return {
    connected,
    status,
    transcripts,
    partials,
    topics,
    devices,
    fetchDevices,
    sendCommand,
    togglePinTopic,
    dismissTopic,
  };
}
