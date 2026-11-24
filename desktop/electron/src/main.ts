import { app, BrowserWindow, dialog, ipcMain } from "electron";
import { ChildProcess, spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import net from "node:net";
import path from "node:path";

let mainWindow: BrowserWindow | null = null;
let backendProcess: ChildProcess | null = null;
let activePort = 8765;

function getRepoRoot(): string {
  // dist/main.js -> desktop/electron/dist -> repo root
  return path.resolve(__dirname, "..", "..", "..");
}

function isPortAvailable(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once("error", () => resolve(false));
    server.once("listening", () => {
      server.close(() => resolve(true));
    });
    server.listen(port, "127.0.0.1");
  });
}

async function pickBackendPort(): Promise<number> {
  if (await isPortAvailable(8765)) {
    return 8765;
  }
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address();
      const port = typeof addr === "object" && addr ? addr.port : 8765;
      server.close(() => resolve(port));
    });
  });
}

function resolveBackendPython(repoRoot: string): string {
  const winVenv = path.join(repoRoot, "backend", ".venv", "Scripts", "python.exe");
  if (fs.existsSync(winVenv)) return winVenv;
  const posixVenv = path.join(repoRoot, "backend", ".venv", "bin", "python");
  if (fs.existsSync(posixVenv)) return posixVenv;
  return "python";
}

function startBackend(repoRoot: string, port: number): ChildProcess {
  const pythonBin = resolveBackendPython(repoRoot);
  const backendDir = path.join(repoRoot, "backend");
  const proc = spawn(
    pythonBin,
    ["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", String(port)],
    {
      cwd: backendDir,
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    }
  );

  proc.stdout?.on("data", (chunk: Buffer) => {
    process.stdout.write(`[backend] ${chunk.toString()}`);
  });
  proc.stderr?.on("data", (chunk: Buffer) => {
    process.stderr.write(`[backend] ${chunk.toString()}`);
  });
  return proc;
}

function probeHealth(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${port}/health`, (res) => {
      res.resume();
      resolve(res.statusCode === 200);
    });
    req.on("error", () => resolve(false));
    req.setTimeout(1000, () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function waitForBackend(port: number, timeoutMs = 20000): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await probeHealth(port)) {
      return true;
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  return false;
}

function stopBackend(): void {
  if (!backendProcess || backendProcess.pid === undefined) return;
  const pid = backendProcess.pid;
  backendProcess = null;

  if (process.platform === "win32") {
    spawnSync("taskkill", ["/pid", String(pid), "/T", "/F"], {
      stdio: "ignore",
      windowsHide: true,
    });
  } else {
    try {
      process.kill(pid, "SIGTERM");
    } catch {
      // Ignore if already exited
    }
  }
}

async function createWindow(repoRoot: string, port: number): Promise<void> {
  mainWindow = new BrowserWindow({
    width: 1160,
    height: 760,
    minWidth: 820,
    minHeight: 560,
    backgroundColor: "#0b0f17",
    title: "Navi Whisper",
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  const devUrl = process.env.NAVI_DEV_URL;
  if (devUrl) {
    const sep = devUrl.includes("?") ? "&" : "?";
    await mainWindow.loadURL(`${devUrl}${sep}backendPort=${port}`);
  } else {
    const indexHtml = path.join(repoRoot, "frontend", "dist", "index.html");
    await mainWindow.loadFile(indexHtml, {
      query: { backendPort: String(port) },
    });
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

ipcMain.handle("desktop:set-always-on-top", (_event, enabled: boolean) => {
  if (!mainWindow) return false;
  mainWindow.setAlwaysOnTop(Boolean(enabled), "screen-saver");
  return mainWindow.isAlwaysOnTop();
});

ipcMain.handle("desktop:set-stealth-mode", (_event, enabled: boolean) => {
  if (!mainWindow) return false;
  const next = Boolean(enabled);
  mainWindow.setContentProtection(next);
  return next;
});

app.whenReady().then(async () => {
  const repoRoot = getRepoRoot();
  if (process.env.NAVI_EXTERNAL_BACKEND !== "1") {
    activePort = await pickBackendPort();
    backendProcess = startBackend(repoRoot, activePort);
    const ready = await waitForBackend(activePort);
    if (!ready) {
      dialog.showErrorBox(
        "Navi Whisper Backend Failed",
        `Timed out waiting for FastAPI backend on port ${activePort}. Check backend/.venv and backend/.env.`
      );
      stopBackend();
      app.quit();
      return;
    }
  }

  await createWindow(repoRoot, activePort);
});

app.on("before-quit", () => {
  stopBackend();
});

app.on("window-all-closed", () => {
  stopBackend();
  if (process.platform !== "darwin") {
    app.quit();
  }
});
