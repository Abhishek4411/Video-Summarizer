#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU-accelerated video-analysis demo that talks to a local Ollama server.
Adds UI buttons to clear the current session’s cache **or** wipe every cached
frame / ChromaDB collection for all sessions.

Put the file wherever you like, then run:

    streamlit run video_analyzer_with_cache_clear.py

(Ensure the Ollama container from the command above is already running.)
"""

import os
import shutil
import tempfile
import base64
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import streamlit as st
import chromadb
from chromadb.config import Settings
import requests

# ────────────────────────────────────────────────────────────────────────────────
# GPU SELECTION – lock to GPU 0 (matches container)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ────────────────────────────────────────────────────────────────────────────────
# CONSTANTS & PATHS
CACHE_DIR   = Path("./cache")
FRAMES_DIR  = CACHE_DIR / "frames"
DB_DIR      = CACHE_DIR / "chromadb"

for d in (CACHE_DIR, FRAMES_DIR, DB_DIR):
    d.mkdir(parents=True, exist_ok=True)

OLLAMA_API_URL = "http://localhost:11435/api"
VISION_MODEL   = "llama3.2-vision:latest"
CHAT_MODEL     = "gemma3:27b"

# ────────────────────────────────────────────────────────────────────────────────
# VECTOR DATABASE
chroma_client = chromadb.PersistentClient(
    path=str(DB_DIR), settings=Settings(anonymized_telemetry=False)
)

# ────────────────────────────────────────────────────────────────────────────────
# SESSION UTILITIES
def get_session_id() -> str:
    """Return a stable random ID per browser session."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = hashlib.md5(
            f"{datetime.now().isoformat()}_{np.random.randint(1_000_000)}".encode()
        ).hexdigest()[:12]
    return st.session_state.session_id


# ────────────────────────────────────────────────────────────────────────────────
# FRAME EXTRACTION
def extract_key_frames(video_path: str, threshold: float = 30.0
                       ) -> List[Tuple[int, np.ndarray, float]]:
    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames, prev_frame, frame_count = [], None, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # first frame OR last frame OR significant diff
        if (
            prev_frame is None
            or frame_count == total_frames - 1
            or np.mean(cv2.absdiff(prev_frame, gray)) > threshold
        ):
            frames.append((frame_count, frame, frame_count / fps))

        # sample once each second for coverage
        if frame_count % int(fps) == 0:
            frames.append((frame_count, frame, frame_count / fps))

        prev_frame = gray
        frame_count += 1

    cap.release()

    # de-dup while preserving order
    seen, unique = set(), []
    for fid, frm, ts in frames:
        if fid not in seen:
            unique.append((fid, frm, ts))
            seen.add(fid)
    return unique


# ────────────────────────────────────────────────────────────────────────────────
# HELPERS
def save_frame(frame: np.ndarray, session_id: str, frame_idx: int) -> str:
    sess_dir = FRAMES_DIR / session_id
    sess_dir.mkdir(exist_ok=True, parents=True)
    path = sess_dir / f"frame_{frame_idx:06d}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)

def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def analyze_frame_with_vision(image_path: str, frame_idx: int, timestamp: float) -> Dict:
    prompt = f"""
Analyze this video frame captured at {timestamp:.2f}s (frame {frame_idx}).

1. Scene overview
2. Every object: position, colour, texture
3. Motion or blur indicators
4. Technical notes: lighting, camera angle, quality
5. Any text / numbers (meter readings etc.)
6. Notable changes from surrounding context

Be exhaustive and precise.
"""
    try:
        r = requests.post(
            f"{OLLAMA_API_URL}/generate",
            json={
                "model": VISION_MODEL,
                "prompt": prompt,
                "images": [_b64(image_path)],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 1000},
            },
            timeout=300,
        )
        r.raise_for_status()
        desc = r.json().get("response", "")
    except Exception as exc:
        desc = f"❌ Error analysing frame: {exc}"
    return {
        "frame_idx": frame_idx,
        "timestamp": timestamp,
        "description": desc,
        "image_path": image_path,
    }

def store_in_chromadb(session_id: str, analyses: List[Dict]):
    name = f"video_analysis_{session_id}"
    try:
        col = chroma_client.create_collection(name=name,
                                              metadata={"hnsw:space": "cosine"})
    except chromadb.errors.CollectionAlreadyExistsError:
        col = chroma_client.get_collection(name=name)

    docs, metas, ids = [], [], []
    for a in analyses:
        docs.append(f"Frame {a['frame_idx']} @ {a['timestamp']:.2f}s: {a['description']}")
        metas.append(
            {"frame_idx": a["frame_idx"], "timestamp": a["timestamp"],
             "image_path": a["image_path"]}
        )
        ids.append(f"frame_{a['frame_idx']}")
    col.add(documents=docs, metadatas=metas, ids=ids)
    return col

def generate_report(session_id: str, video_name: str) -> str:
    name = f"video_analysis_{session_id}"
    col  = chroma_client.get_collection(name=name)
    res  = col.get()
    rows = sorted(zip(res["documents"], res["metadatas"]),
                  key=lambda r: r[1]["frame_idx"])
    context = "\n\n".join(d for d, _ in rows)
    prompt  = f"""
You are a meticulous analyst. Using the frame-by-frame notes below,
create a full report.

FRAME NOTES
-----------
{context}

FORMAT
------
1. Executive Summary
2. Detailed Timeline
3. Key Observations
4. Technical Analysis
5. Conclusions & Insights
"""
    r = requests.post(
        f"{OLLAMA_API_URL}/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 2000},
        },
        timeout=300,
    )
    return r.json().get("response", "❌ Failed to build report")

def query_analysis(session_id: str, question: str) -> str:
    name = f"video_analysis_{session_id}"
    col  = chroma_client.get_collection(name=name)
    res  = col.query(query_texts=[question], n_results=5)
    if not res["documents"][0]:
        return "No matching frames found."

    ctx = "\n\n".join(
        f"Frame {m['frame_idx']} @ {m['timestamp']:.2f}s:\n{doc}"
        for doc, m in zip(res["documents"][0], res["metadatas"][0])
    )
    prompt = f"""
Answer the user's question using ONLY the context below.

CONTEXT
-------
{ctx}

QUESTION
--------
{question}
"""
    r = requests.post(
        f"{OLLAMA_API_URL}/generate",
        json={"model": CHAT_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    return r.json().get("response", "❌ LLM error")


# ────────────────────────────────────────────────────────────────────────────────
# CACHE-CLEARING HELPERS
def clear_session_cache(session_id: str):
    """Remove frames & ChromaDB collection for this session only."""
    shutil.rmtree(FRAMES_DIR / session_id, ignore_errors=True)
    try:
        chroma_client.delete_collection(name=f"video_analysis_{session_id}")
    except chromadb.errors.InvalidCollectionError:
        pass
    st.session_state.pop("video_analyzed", None)
    st.session_state.pop("analyses", None)
    st.session_state.pop("messages", None)

# def clear_all_caches():
#     """Dangerous: wipe every cached frame and the entire ChromaDB store."""
#     shutil.rmtree(CACHE_DIR, ignore_errors=True)
#     CACHE_DIR.mkdir(); (FRAMES_DIR).mkdir(); (DB_DIR).mkdir()
#     # reset chroma_client
#     global chroma_client
#     chroma_client = chromadb.PersistentClient(
#         path=str(DB_DIR), settings=Settings(anonymized_telemetry=False)
#     )
#     st.session_state.clear()


# ────────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
st.set_page_config(page_title="GPU Video Analyzer", layout="wide")
session_id = get_session_id()

st.title("🎥 GPU Video Analysis System")
st.caption(f"Session ID • {session_id}")

uploaded_file = st.file_uploader(
    "Upload a video file (max 200 MB)", type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    col1, col2 = st.columns(2)
    with col1:
        st.video(video_path)
    with col2:
        if st.button("🚀 Analyze video", type="primary"):
            with st.spinner("Extracting key frames…"):
                frames = extract_key_frames(video_path)
                st.info(f"{len(frames)} key frames selected.")
            analyses, pb, txt = [], st.progress(0), st.empty()
            for i, (fid, frm, ts) in enumerate(frames, 1):
                txt.text(f"Analysing frame {i}/{len(frames)} …")
                pb.progress(i / len(frames))
                img_pth = save_frame(frm, session_id, fid)
                analyses.append(analyze_frame_with_vision(img_pth, fid, ts))
            store_in_chromadb(session_id, analyses)
            report = generate_report(session_id, uploaded_file.name)
            st.session_state.video_analyzed = True
            st.session_state.analyses = analyses
            st.session_state.messages = [{"role": "assistant", "content": report}]
            txt.empty(); pb.empty()
            os.unlink(video_path)

# ────────────────────────────────────────────────────────────────────────────────
# CHAT
if st.session_state.get("video_analyzed"):
    st.divider()
    st.subheader("💬 Chat with the analysis")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask about the video"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = query_analysis(session_id, prompt)
                st.write(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

# ────────────────────────────────────────────────────────────────────────────────
# SIDEBAR
with st.sidebar:
    st.subheader("🖼️ Analyzed frames")
    for a in st.session_state.get("analyses", []):
        with st.expander(f"Frame {a['frame_idx']} @ {a['timestamp']:.2f}s"):
            st.image(a["image_path"], use_container_width=True)
            st.caption(a["description"][:200] + "…")

    st.subheader("⚙️ Cache management")
    if st.button("Clear **this session** cache"):
        clear_session_cache(session_id)
        st.success("Session cache cleared – reload page to start fresh.")
    # if st.button("Clear **ALL** caches (irreversible)", type="secondary"):
    #     clear_all_caches()
    #     st.success("All caches cleared – reload page to start over.")
