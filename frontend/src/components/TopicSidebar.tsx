import { BookOpen, Search } from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import type { TopicCardItem } from "../types/events";
import { TopicCard } from "./TopicCard";

interface TopicSidebarProps {
  topics: TopicCardItem[];
  highlightedTopicId: string | null;
  connected: boolean;
  onManualLookup: (text: string, contextSegmentId?: string) => void;
  onTogglePin: (topicId: string) => void;
  onDismiss: (topicId: string) => void;
  onJumpToSegment: (segmentId: string) => void;
}

export function TopicSidebar({
  topics,
  highlightedTopicId,
  connected,
  onManualLookup,
  onTogglePin,
  onDismiss,
  onJumpToSegment,
}: TopicSidebarProps) {
  const [query, setQuery] = useState("");
  const listRef = useRef<HTMLDivElement | null>(null);

  const orderedTopics = useMemo(() => {
    const pinned = topics.filter((t) => t.pinned);
    const unpinned = topics.filter((t) => !t.pinned);
    return [...pinned, ...unpinned];
  }, [topics]);

  useEffect(() => {
    if (!highlightedTopicId || !listRef.current) return;
    const el = listRef.current.querySelector(
      `[data-topic-id="${CSS.escape(highlightedTopicId)}"]`
    );
    el?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [highlightedTopicId]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;
    onManualLookup(trimmed);
    setQuery("");
  };

  return (
    <aside className="sidebar-pane">
      <div className="pane-header">
        <h2>Topic Summaries</h2>
        <span className="pane-subtitle">{topics.length} active</span>
      </div>

      <form className="lookup-form" onSubmit={handleSubmit}>
        <Search size={14} className="lookup-icon" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Look up any term or concept..."
          disabled={!connected}
        />
        <button type="submit" disabled={!connected || !query.trim()}>
          Go
        </button>
      </form>

      <div className="sidebar-scroll" ref={listRef}>
        {orderedTopics.length === 0 ? (
          <div className="empty-state">
            <BookOpen size={24} className="muted-icon" />
            <p className="empty-title">No topics detected yet</p>
            <p className="empty-desc">
              Specialized terms from your conversation will appear here with 1–3 concise
              bullets, or highlight any text in the transcript to look it up.
            </p>
          </div>
        ) : (
          <div className="topic-list">
            {orderedTopics.map((card) => (
              <TopicCard
                key={card.topic_id}
                card={card}
                isHighlighted={card.topic_id === highlightedTopicId}
                onTogglePin={onTogglePin}
                onRefresh={(c) =>
                  onManualLookup(c.display_name, c.source_segment_id ?? undefined)
                }
                onDismiss={onDismiss}
                onJumpToSegment={onJumpToSegment}
              />
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}
