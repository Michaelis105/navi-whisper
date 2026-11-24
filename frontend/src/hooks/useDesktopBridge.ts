import { useCallback, useState } from "react";

export function useDesktopBridge() {
  const bridge = typeof window !== "undefined" ? window.desktopBridge : undefined;
  const isDesktop = Boolean(bridge?.isDesktop);

  const [alwaysOnTop, setAlwaysOnTopState] = useState(false);
  const [stealthMode, setStealthModeState] = useState(false);

  const toggleAlwaysOnTop = useCallback(async () => {
    if (!bridge) return;
    const next = !alwaysOnTop;
    const applied = await bridge.setAlwaysOnTop(next);
    setAlwaysOnTopState(applied);
  }, [alwaysOnTop, bridge]);

  const toggleStealthMode = useCallback(async () => {
    if (!bridge) return;
    const next = !stealthMode;
    const applied = await bridge.setStealthMode(next);
    setStealthModeState(applied);
  }, [stealthMode, bridge]);

  return {
    isDesktop,
    alwaysOnTop,
    stealthMode,
    toggleAlwaysOnTop,
    toggleStealthMode,
  };
}
