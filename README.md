# Navi-Whisper

Passively listens to PC voice/video calls (Microphone + Windows WASAPI Loopback), transcribes locally with CUDA `faster-whisper`, filters out non-technical chatter, and renders concise 1–3 bullet summaries of specialized topics in a companion sidebar.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full architecture details.

## Setup

### 1. Backend (`backend/`)
1. Configure your API key in `backend/.env` (copy from `backend/.env.example`):
   ```env
   GEMINI_API_KEY=your_actual_key_here
   ```
2. Dependencies live in `backend/.venv`:
   ```powershell
   cd backend
   .\.venv\Scripts\python.exe -m pytest -v
   ```

### 2. Frontend (`frontend/`)
Uses Node.js installed at `E:\Applications\Dependencies\nodejs`:
```powershell
$env:PATH = "E:\Applications\Dependencies\nodejs;$env:PATH"
cd frontend
npm install
npm run build
```

### 3. Desktop App (`desktop/electron/`)
Builds the frontend if needed, spawns the Python FastAPI sidecar automatically, and launches the Electron companion window (with **Always on Top** and **Stealth / Content Protection** support):
```powershell
$env:PATH = "E:\Applications\Dependencies\nodejs;$env:PATH"
cd desktop\electron
npm install
npm start
```
