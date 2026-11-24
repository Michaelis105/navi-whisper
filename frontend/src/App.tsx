import { useCallback, useState } from "react";
import { HeaderBar } from "./components/HeaderBar";
import { TopicSidebar } from "./components/TopicSidebar";
import { TranscriptFeed } from "./components/TranscriptFeed";
import { useDesktopBridge } from "./hooks/useDesktopBridge";
import { useNaviSocket } from "./hooks/useNaviSocket";

export function App() {
  const {
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
  } = useNaviSocket();

  const {
    isDesktop,
    alwaysOnTop,
    stealthMode,
    toggleAlwaysOnTop,
    toggleStealthMode,
  } = useDesktopBridge();

  const [highlightedTopicId, setHighlightedTopicId] = useState<string | null>(null);
  const [highlightedSegmentId, setHighlightedSegmentId] = useState<string | null>(null);

  const handleSelectTopic = useCallback((topicId: string) => {
    setHighlightedTopicId(topicId);
    window.setTimeout(() => setHighlightedTopicId((cur) => (cur === topicId ? null : cur)), 1800);
  }, []);

  const handleJumpToSegment = useCallback((segmentId: string) => {
    setHighlightedSegmentId(segmentId);
    window.setTimeout(
      () => setHighlightedSegmentId((cur) => (cur === segmentId ? null : cur)),
      1800
    );
  }, []);

  const handleManualLookup = useCallback(
    (text: string, contextSegmentId?: string) => {
      sendCommand({
        type: "topic.manual_lookup",
        text,
        context_segment_id: contextSegmentId ?? null,
      });
    },
    [sendCommand]
  );

  return (
    <div className="app-shell">
      <HeaderBar
        connected={connected}
        status={status}
        devices={devices}
        isDesktop={isDesktop}
        alwaysOnTop={alwaysOnTop}
        stealthMode={stealthMode}
        onToggleAlwaysOnTop={toggleAlwaysOnTop}
        onToggleStealthMode={toggleStealthMode}
        onSendCommand={sendCommand}
        onRefreshDevices={fetchDevices}
      />
      <main className="workspace-grid">
        <TranscriptFeed
          transcripts={transcripts}
          partials={partials}
          topics={topics}
          listening={status.listening}
          highlightedSegmentId={highlightedSegmentId}
          onSelectTopic={handleSelectTopic}
          onManualLookup={handleManualLookup}
        />
        <TopicSidebar
          topics={topics}
          highlightedTopicId={highlightedTopicId}
          connected={connected}
          onManualLookup={handleManualLookup}
          onTogglePin={togglePinTopic}
          onDismiss={(id) => dismissTopic(id, false)}
          onJumpToSegment={handleJumpToSegment}
        />
      </main>
    </div>
  );
}

export default App;
