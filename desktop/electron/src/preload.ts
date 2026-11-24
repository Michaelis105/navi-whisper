import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("desktopBridge", {
  isDesktop: true,
  setAlwaysOnTop: (enabled: boolean): Promise<boolean> =>
    ipcRenderer.invoke("desktop:set-always-on-top", enabled),
  setStealthMode: (enabled: boolean): Promise<boolean> =>
    ipcRenderer.invoke("desktop:set-stealth-mode", enabled),
});
