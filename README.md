# 🎬 Intelligent Video Analyzer v3.0

Enhanced GPU‑accelerated video analysis with OCR, intelligent frame de‑duplication, and natural‑language summaries — wrapped in a clean Streamlit UI.

> **Status:** v3.0 (latest) · Works on Windows & Linux · CPU or CUDA GPU

---

## Table of Contents

- [Key Features](#key-features)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Run](#run)
- [Usage Tips](#usage-tips)
- [Troubleshooting](#troubleshooting)
- [Benchmarks & Performance](#benchmarks--performance)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

- **Intelligent key‑frame extraction** with visual‑similarity checks to avoid near‑duplicates
- **OCR on frames (EasyOCR)** with optional CUDA acceleration
- **Natural language, multi‑part responses** to user questions about the video
- **ChromaDB vector store** for fast semantic search over frame captions + OCR text
- **End‑to‑end Streamlit app** — upload → analyze → ask questions → export
- **Cache & artifacts** saved as images, OCR text, and captions per session
- **Configurable vision/chat models** via a local **Ollama**-compatible API

---

## Demo

- Launch the app (see [Run](#run)) and upload a small `.mp4`.
- The app will:
  1. Select unique frames (scene changes & periodic sampling)
  2. Run OCR (EasyOCR) per selected frame
  3. Describe frames via vision LLM
  4. Store combined text in ChromaDB
  5. Generate a comprehensive report and enable QA chat over the video

---

## Project Structure

```
videoreader/
├─ 1_version_video_analyser.py
├─ 2_version_video_analyser.py
├─ 3_version_video_analyser.py    # ← v3 app (latest)
├─ requirements.txt
├─ .env                            # local config (see Configuration)
├─ streamlit.log / streamlit_log.out
└─ cache/
   ├─ frames/      # extracted frames per session
   ├─ captions/    # per-frame caption text
   ├─ ocr/         # per-frame OCR text
   └─ chromadb/    # persistent ChromaDB index
```
> The app creates `cache/` on first run and groups artifacts by a random **session id**.

---

## How It Works

1. **Frame Selection**
   - Periodic sampling + scene change detection (grayscale absdiff)
   - Near‑duplicate filtering using 3‑channel histograms + correlation threshold
2. **OCR**
   - EasyOCR (GPU if available) extracts English text from each key frame
3. **Vision Captioning**
   - Each frame is sent to a **vision model** via a local Ollama‑compatible endpoint,
     with OCR text provided as context for better accuracy
4. **Similarity‑aware Captions**
   - Descriptions are compared to recent ones (Sentence‑Transformers) to avoid redundancy
5. **Vector Store**
   - Combined caption + OCR text is stored in **ChromaDB** with cosine similarity
6. **Natural QA & Report**
   - Queries retrieve the most relevant snippets → a **chat model** produces fluent answers
   - A comprehensive report is generated post‑analysis

---

## Requirements

- **Python** 3.10+ recommended
- **FFmpeg** available in `PATH` (for some video formats)
- **GPU (optional)** with recent **CUDA** drivers for acceleration (Torch & EasyOCR)
- Local **Ollama‑compatible** API (defaults in code):
  - `OLLAMA_API_URL = http://localhost:11436/api`
  - `VISION_MODEL = llama4:scout`
  - `CHAT_MODEL = llama4:scout`

> _CPU‑only works_, but OCR & generation will be slower.

### Python Dependencies

This repo includes `requirements.txt`. Typical contents:

```
# Core
streamlit
opencv-python
numpy
requests
python-dotenv

# Vector & NLP
chromadb
sentence-transformers
scikit-learn

# OCR & Imaging
easyocr
pillow
torch  # choose CUDA or CPU build as needed

# Utilities
jsonlines
psutil
gputil
beautifulsoup4
googlesearch-python

# Video I/O (optional helpers)
moviepy
ffmpeg-python
```

---

## Installation

> Windows PowerShell example assumes the project folder:
> `C:\Users\Abhishek Patel\OneDrive - Netweb Technologies India Pvt. Ltd\Work\VS Code_Test Area\videoreader`

1. **Create & activate a virtual environment**
   ```powershell
   py -3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   _Linux/macOS:_
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Python dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Install/verify FFmpeg**
   - Windows: use `choco install ffmpeg` (or add FFmpeg to PATH)
   - Linux: `sudo apt-get install ffmpeg`

4. **(Optional) Torch CUDA build**
   - If you have a CUDA GPU, install a matching Torch build from the official index.

5. **(Optional) EasyOCR language packs**
   - English (`en`) is enabled by default. Add more codes in `easyocr.Reader([...])` if needed.

---

## Configuration

Configure via environment variables (recommended) or edit defaults in `3_version_video_analyser.py`.

Create **.env** in the project root:

```env
# Ollama‑compatible API endpoint
OLLAMA_API_URL=http://localhost:11436/api

# Model ids (must exist on your local model server)
VISION_MODEL=llama4:scout
CHAT_MODEL=llama4:scout

# Optional GPU pinning (set only if you want a specific GPU)
CUDA_VISIBLE_DEVICES=0
```

> If `.env` is present, consider loading it at app start with `python-dotenv` (already included).

---

## Run

From the project root (virtualenv active):

```bash
streamlit run 3_version_video_analyser.py
```

Then open the URL Streamlit prints (usually <http://localhost:8501>).

**Workflow:** Upload a video → click **“Start Intelligent Analysis”** → review stats → ask questions in the chat → download artifacts from `cache/`.

---

## Usage Tips

- **Similarity Threshold:** Controls how aggressively we drop near‑duplicate frames.
  - Higher = fewer frames, faster, but risk of missing subtle changes.
- **Big Videos:** Prefer shorter clips for rapid iteration, then scale up.
- **OCR Languages:** Add more language codes to `easyocr.Reader([...])` when needed.
- **ChromaDB Reset:** Use the sidebar **“Clear Session Cache”** to delete the current session’s frames, OCR files, captions, and vector index.

---

## Troubleshooting

- **Torch/EasyOCR uses CPU only**
  - Ensure you installed a **CUDA** build of Torch and the right NVIDIA drivers.
  - Verify with:
    ```python
    import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
    ```
- **EasyOCR CUDA errors**
  - Reinstall Torch for the matching CUDA version.
  - Set `gpu=False` in `easyocr.Reader([...], gpu=False)` to force CPU.
- **FFmpeg not found**
  - Install FFmpeg and make sure it’s in your `PATH`.
- **Ollama / model endpoint unreachable**
  - Confirm `OLLAMA_API_URL` and that the vision/chat model ids exist and can run inference.
- **Chroma collection already exists**
  - The app handles it, but you can wipe the index via the sidebar cache controls.

---

## Benchmarks & Performance

- **Throughput** scales with fewer selected frames and GPU OCR.
- **Latency** driven by vision/chat model size and local model server.
- **Storage**: images + text per frame; see `cache/` for counts and sizes.

> Use the sidebar metrics to monitor frames analyzed and OCR coverage.

---

## Roadmap

- [ ] Multi‑language OCR presets (UI toggle)
- [ ] Export report as PDF/Markdown
- [ ] Pluggable embedding models for ChromaDB
- [ ] Batch processing CLI
- [ ] Optional cloud‑hosted inference backends

---

## Contributing

Contributions are welcome! Please open an issue or PR. For significant changes, discuss your plans first.

---

## License

This project is licensed under the **MIT License** (or your organization’s preferred license). Add the full text in `LICENSE`.
