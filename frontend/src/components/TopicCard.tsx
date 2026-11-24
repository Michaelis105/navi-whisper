import { MessageSquareQuote, Pin, RefreshCw, X } from "lucide-react";
import type { TopicCardItem } from "../types/events";

interface TopicCardProps {
  card: TopicCardItem;
  isHighlighted: boolean;
  onTogglePin: (topicId: string) => void;
  onRefresh: (card: TopicCardItem) => void;
  onDismiss: (topicId: string) => void;
  onJumpToSegment: (segmentId: string) => void;
}

export function TopicCard({
  card,
  isHighlighted,
  onTogglePin,
  onRefresh,
  onDismiss,
  onJumpToSegment,
}: TopicCardProps) {
  return (
    <article
      data-topic-id={card.topic_id}
      className={`topic-card ${card.pinned ? "topic-card-pinned" : ""} ${
        isHighlighted ? "topic-card-highlighted" : ""
      }`}
    >
      <div className="topic-card-header">
        <div className="topic-title-wrap">
          <h3 className="topic-title">{card.display_name}</h3>
          <span className="topic-category">{card.category}</span>
        </div>

        <div className="topic-actions">
          {card.source_segment_id && (
            <button
              type="button"
              className="icon-btn"
              onClick={() => onJumpToSegment(card.source_segment_id!)}
              title="Jump to transcript utterance"
            >
              <MessageSquareQuote size={13} />
            </button>
          )}
          <button
            type="button"
            className={`icon-btn ${card.pinned ? "icon-btn-active" : ""}`}
            onClick={() => onTogglePin(card.topic_id)}
            title={card.pinned ? "Unpin card" : "Pin card"}
          >
            <Pin size={13} />
          </button>
          <button
            type="button"
            className="icon-btn"
            onClick={() => onRefresh(card)}
            title="Re-summarize topic"
          >
            <RefreshCw size={13} />
          </button>
          <button
            type="button"
            className="icon-btn"
            onClick={() => onDismiss(card.topic_id)}
            title="Dismiss topic"
          >
            <X size={13} />
          </button>
        </div>
      </div>

      {card.status === "researching" && (
        <div className="topic-skeleton" aria-label="Researching topic">
          <div className="shimmer-line shimmer-long" />
          <div className="shimmer-line shimmer-medium" />
        </div>
      )}

      {card.status === "complete" && (
        <>
          <ul className="topic-bullets">
            {card.bullets.map((bullet, idx) => (
              <li key={idx}>{bullet}</li>
            ))}
          </ul>
          {card.latency_ms !== null && (
            <div className="topic-footer">{card.latency_ms} ms</div>
          )}
        </>
      )}

      {card.status === "error" && (
        <div className="topic-error">
          <span>{card.error_message || "Could not fetch summary."}</span>
          <button type="button" className="btn-link" onClick={() => onRefresh(card)}>
            Retry
          </button>
        </div>
      )}
    </article>
  );
}
