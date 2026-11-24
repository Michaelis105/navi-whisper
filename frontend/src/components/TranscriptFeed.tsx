import { ArrowDown, Search, Sparkles } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Speaker, TopicCardItem, TranscriptItem } from "../types/events";

interface TranscriptFeedProps {
  transcripts: TranscriptItem[];
  partials: Partial<Record<Speaker, TranscriptItem>>;
  topics: TopicCardItem[];
  listening: boolean;
  highlightedSegmentId: string | null;
  onSelectTopic: (topicId: string) => void;
  onManualLookup: (text: string, contextSegmentId?: string) => void;
}

function formatTime(epochSec: number): string {
  const ms = epochSec > 1e12 ? epochSec : epochSec * 1000;
  return new Date(ms).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function TranscriptFeed({
  transcripts,
  partials,
  topics,
  listening,
  highlightedSegmentId,
  onSelectTopic,
  onManualLookup,
}: TranscriptFeedProps) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [selectionState, setSelectionState] = useState<{
    text: string;
    segmentId?: string;
    top: number;
    left: number;
  } | null>(null);

  const topicsBySegment = useMemo(() => {
    const map = new Map<string, TopicCardItem[]>();
    for (const card of topics) {
      if (!card.source_segment_id) continue;
      const list = map.get(card.source_segment_id) ?? [];
      list.push(card);
      map.set(card.source_segment_id, list);
    }
    return map;
  }, [topics]);

  const activePartials = useMemo(
    () =>
      Object.values(partials).filter(
        (p): p is TranscriptItem => Boolean(p && p.text.trim())
      ),
    [partials]
  );

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcripts, activePartials, autoScroll]);

  useEffect(() => {
    if (!highlightedSegmentId || !scrollRef.current) return;
    const el = scrollRef.current.querySelector(
      `[data-segment-id="${CSS.escape(highlightedSegmentId)}"]`
    );
    if (el) {
      setAutoScroll(false);
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [highlightedSegmentId]);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 64;
    setAutoScroll(nearBottom);
  };

  const handleMouseUp = () => {
    const sel = window.getSelection();
    const text = sel?.toString().trim() ?? "";
    if (!text || text.length < 2 || text.length > 140 || !sel || sel.rangeCount === 0) {
      setSelectionState(null);
      return;
    }
    const range = sel.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    let node: Node | null = range.commonAncestorContainer;
    let segId: string | undefined;
    while (node) {
      if (node instanceof HTMLElement && node.dataset.segmentId) {
        segId = node.dataset.segmentId;
        break;
      }
      node = node.parentNode;
    }
    setSelectionState({
      text,
      segmentId: segId,
      top: Math.max(12, rect.top - 40),
      left: Math.max(16, rect.left),
    });
  };

  return (
    <section className="transcript-pane" onMouseUp={handleMouseUp}>
      <div className="pane-header">
        <h2>Live Conversation</h2>
        <span className="pane-subtitle">
          {transcripts.length} {transcripts.length === 1 ? "utterance" : "utterances"}
        </span>
      </div>

      {selectionState && (
        <button
          type="button"
          className="selection-lookup-btn"
          style={{ top: selectionState.top, left: selectionState.left }}
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => {
            onManualLookup(selectionState.text, selectionState.segmentId);
            setSelectionState(null);
            window.getSelection()?.removeAllRanges();
          }}
        >
          <Search size={13} />
          <span>Summarize &ldquo;{selectionState.text.slice(0, 28)}&rdquo;</span>
        </button>
      )}

      <div className="transcript-scroll" ref={scrollRef} onScroll={handleScroll}>
        {transcripts.length === 0 && activePartials.length === 0 ? (
          <div className="empty-state">
            <Sparkles size={26} className="muted-icon" />
            <p className="empty-title">
              {listening
                ? "Listening for voice activity..."
                : "Ready to listen to your call"}
            </p>
            <p className="empty-desc">
              Click <strong>Start Listening</strong> to capture your microphone and system
              audio. Specialized terms will be summarized automatically on the right.
            </p>
          </div>
        ) : (
          <div className="transcript-list">
            {transcripts.map((seg) => {
              const linkedTopics = topicsBySegment.get(seg.segment_id) ?? [];
              const isHighlighted = seg.segment_id === highlightedSegmentId;
              return (
                <article
                  key={seg.segment_id}
                  data-segment-id={seg.segment_id}
                  className={`utterance utterance-${seg.speaker} ${
                    isHighlighted ? "utterance-highlighted" : ""
                  }`}
                >
                  <div className="utterance-meta">
                    <span className={`speaker-badge speaker-${seg.speaker}`}>
                      {seg.speaker === "you" ? "You" : "Others"}
                    </span>
                    <span className="utterance-time">{formatTime(seg.timestamp)}</span>
                  </div>
                  <p className="utterance-text">{seg.text}</p>
                  {linkedTopics.length > 0 && (
                    <div className="inline-topics">
                      {linkedTopics.map((topic) => (
                        <button
                          key={topic.topic_id}
                          type="button"
                          className="inline-topic-pill"
                          onClick={() => onSelectTopic(topic.topic_id)}
                        >
                          {topic.display_name}
                        </button>
                      ))}
                    </div>
                  )}
                </article>
              );
            })}

            {activePartials.map((part) => (
              <article
                key={`partial-${part.speaker}-${part.segment_id}`}
                className={`utterance utterance-${part.speaker} utterance-partial`}
              >
                <div className="utterance-meta">
                  <span className={`speaker-badge speaker-${part.speaker}`}>
                    {part.speaker === "you" ? "You" : "Others"}
                  </span>
                  <span className="utterance-time">live</span>
                </div>
                <p className="utterance-text">{part.text}</p>
              </article>
            ))}
          </div>
        )}
      </div>

      {!autoScroll && (
        <button
          type="button"
          className="jump-bottom-btn"
          onClick={() => {
            setAutoScroll(true);
            if (scrollRef.current) {
              scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
            }
          }}
        >
          <ArrowDown size={14} />
          <span>Jump to live</span>
        </button>
      )}
    </section>
  );
}
