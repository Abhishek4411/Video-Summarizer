#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced GPU-accelerated video-analysis with OCR, intelligent frame deduplication,
and natural language responses. Version 3.0

Features:
- OCR text extraction from frames
- Intelligent frame deduplication
- Natural, human-like responses
- Automatic multi-part responses for long content
- Structured caption storage
"""

import os
import shutil
import tempfile
import base64
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

import cv2
import numpy as np
import streamlit as st
st.set_page_config(page_title="Intelligent Video Analyzer v3", layout="wide")
import chromadb
from chromadb.config import Settings
import requests
import easyocr
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ────────────────────────────────────────────────────────────────────────────────
# GPU SELECTION
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ────────────────────────────────────────────────────────────────────────────────
# CONSTANTS & PATHS
CACHE_DIR   = Path("./cache")
FRAMES_DIR  = CACHE_DIR / "frames"
DB_DIR      = CACHE_DIR / "chromadb"
CAPTIONS_DIR = CACHE_DIR / "captions"
OCR_DIR     = CACHE_DIR / "ocr"

for d in (CACHE_DIR, FRAMES_DIR, DB_DIR, CAPTIONS_DIR, OCR_DIR):
    d.mkdir(parents=True, exist_ok=True)

OLLAMA_API_URL = "http://localhost:11436/api"
VISION_MODEL   = "llama4:scout"
CHAT_MODEL     = "llama4:scout"


# Initialize OCR reader (GPU-accelerated if available)
ocr_reader = easyocr.Reader(['en'], gpu=True)

# Initialize sentence transformer for similarity checking
@st.cache_resource
def get_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

sentence_model = get_sentence_model()

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

# st.set_page_config(page_title="Intelligent Video Analyzer v3", layout="wide")
session_id = get_session_id()

# ────────────────────────────────────────────────────────────────────────────────
# ENHANCED FRAME EXTRACTION WITH SIMILARITY CHECK
def compute_frame_hash(frame: np.ndarray) -> str:
    """Compute perceptual hash of frame for similarity detection."""
    resized = cv2.resize(frame, (64, 64))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return hashlib.md5(gray.tobytes()).hexdigest()

def are_frames_similar(frame1: np.ndarray, frame2: np.ndarray, threshold: float = 0.95) -> bool:
    """Check if two frames are visually similar using histogram comparison."""
    hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    
    hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return correlation > threshold

def extract_intelligent_frames(video_path: str, similarity_threshold: float = 0.85) -> List[Tuple[int, np.ndarray, float]]:
    """Extract key frames with intelligent deduplication."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    prev_frame = None
    frame_count = 0
    last_added_frame = None
    min_frame_interval = int(fps * 0.5)  # At least 0.5 seconds between similar frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        should_add = False
        
        # Always add first and last frames
        if frame_count == 0 or frame_count == total_frames - 1:
            should_add = True
        # Check for scene changes
        elif prev_frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            diff = np.mean(cv2.absdiff(prev_gray, gray))
            
            # Significant change detected
            if diff > 30.0:
                # Check if not too similar to last added frame
                if last_added_frame is None or not are_frames_similar(frame, last_added_frame, similarity_threshold):
                    should_add = True
        
        # Periodic sampling (every 2 seconds) if no recent frame
        if not should_add and frame_count % int(fps * 2) == 0:
            if last_added_frame is None or not are_frames_similar(frame, last_added_frame, similarity_threshold):
                should_add = True
        
        if should_add:
            frames.append((frame_count, frame, frame_count / fps))
            last_added_frame = frame.copy()
        
        prev_frame = frame
        frame_count += 1
    
    cap.release()
    return frames

# ────────────────────────────────────────────────────────────────────────────────
# OCR FUNCTIONALITY
def perform_ocr(frame: np.ndarray) -> List[str]:
    """Extract text from frame using OCR."""
    try:
        results = ocr_reader.readtext(frame, detail=0)
        return [text.strip() for text in results if text.strip()]
    except Exception as e:
        return []

def save_ocr_results(session_id: str, frame_idx: int, ocr_texts: List[str]) -> str:
    """Save OCR results to text file."""
    sess_ocr_dir = OCR_DIR / session_id
    sess_ocr_dir.mkdir(exist_ok=True, parents=True)
    
    ocr_path = sess_ocr_dir / f"frame_{frame_idx:06d}_ocr.txt"
    with open(ocr_path, 'w', encoding='utf-8') as f:
        if ocr_texts:
            f.write("Detected Text:\n")
            for text in ocr_texts:
                f.write(f"- {text}\n")
        else:
            f.write("No text detected in this frame.\n")
    
    return str(ocr_path)

# ────────────────────────────────────────────────────────────────────────────────
# ENHANCED FRAME ANALYSIS
def save_frame(frame: np.ndarray, session_id: str, frame_idx: int) -> str:
    sess_dir = FRAMES_DIR / session_id
    sess_dir.mkdir(exist_ok=True, parents=True)
    path = sess_dir / f"frame_{frame_idx:06d}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)

def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def save_caption(session_id: str, frame_idx: int, timestamp: float, caption: str) -> str:
    """Save frame caption to text file."""
    sess_caption_dir = CAPTIONS_DIR / session_id
    sess_caption_dir.mkdir(exist_ok=True, parents=True)
    
    caption_path = sess_caption_dir / f"frame_{frame_idx:06d}_caption.txt"
    with open(caption_path, 'w', encoding='utf-8') as f:
        f.write(f"Frame {frame_idx} at {timestamp:.2f} seconds:\n\n")
        f.write(caption)
    
    return str(caption_path)

def check_content_similarity(new_content: str, existing_contents: List[str], threshold: float = 0.85) -> bool:
    """Check if new content is similar to existing contents."""
    if not existing_contents:
        return False
    
    new_embedding = sentence_model.encode([new_content])
    existing_embeddings = sentence_model.encode(existing_contents)
    
    similarities = cosine_similarity(new_embedding, existing_embeddings)[0]
    return np.max(similarities) > threshold

def analyze_frame_comprehensive(frame: np.ndarray, image_path: str, frame_idx: int, 
                               timestamp: float, session_id: str,
                               previous_analyses: List[Dict]) -> Optional[Dict]:
    """Comprehensive frame analysis with OCR and similarity checking."""
    
    # Perform OCR
    ocr_texts = perform_ocr(frame)
    ocr_path = save_ocr_results(session_id, frame_idx, ocr_texts)
    
    # Create enhanced prompt including OCR results
    ocr_context = ""
    if ocr_texts:
        ocr_context = f"\n\nText visible in frame: {', '.join(ocr_texts)}"
    
    prompt = f"""
Describe what you observe in this frame captured at {timestamp:.2f} seconds.{ocr_context}

Provide a natural, comprehensive description covering:
- The overall scene and environment
- People, objects, and their positions
- Actions or movements occurring
- Notable details about lighting, colors, and composition
- Any visible text or signage (confirm OCR accuracy)
- Significant changes from previous observations

Write as if describing the scene to someone who cannot see it, using natural language without technical jargon.
"""
    
    try:
        # Get frame description from vision model
        r = requests.post(
            f"{OLLAMA_API_URL}/generate",
            json={
                "model": VISION_MODEL,
                "prompt": prompt,
                "images": [_b64(image_path)],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 1500},
            },
            timeout=300,
        )
        r.raise_for_status()
        description = r.json().get("response", "")
        
        # Check if this description is too similar to previous ones
        if previous_analyses:
            previous_descriptions = [a["description"] for a in previous_analyses[-5:]]  # Check last 5
            if check_content_similarity(description, previous_descriptions):
                # Ask for differences only
                diff_prompt = f"""
This frame at {timestamp:.2f}s appears similar to recent frames. 
What specific changes or new details can you identify that weren't present before?
If there are no significant changes, briefly confirm this.
"""
                r = requests.post(
                    f"{OLLAMA_API_URL}/generate",
                    json={
                        "model": VISION_MODEL,
                        "prompt": diff_prompt,
                        "images": [_b64(image_path)],
                        "stream": False,
                        "options": {"temperature": 0.3, "num_predict": 500},
                    },
                    timeout=300,
                )
                diff_description = r.json().get("response", "")
                
                # If no significant changes, skip this frame
                if "no significant changes" in diff_description.lower():
                    return None
                
                description = diff_description
        
        # Save caption
        caption_path = save_caption(session_id, frame_idx, timestamp, description)
        
        return {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "description": description,
            "image_path": image_path,
            "ocr_texts": ocr_texts,
            "ocr_path": ocr_path,
            "caption_path": caption_path
        }
        
    except Exception as exc:
        return {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "description": f"Unable to analyze frame: {str(exc)}",
            "image_path": image_path,
            "ocr_texts": ocr_texts,
            "ocr_path": ocr_path,
            "caption_path": ""
        }

# ────────────────────────────────────────────────────────────────────────────────
# ENHANCED DATABASE STORAGE
def store_in_chromadb_enhanced(session_id: str, analyses: List[Dict]):
    """Store enhanced analysis data including OCR results."""
    name = f"video_analysis_{session_id}"
    try:
        col = chroma_client.create_collection(name=name,
                                              metadata={"hnsw:space": "cosine"})
    except chromadb.errors.CollectionAlreadyExistsError:
        col = chroma_client.get_collection(name=name)
    
    docs, metas, ids = [], [], []
    for a in analyses:
        # Combine description with OCR text for better search
        ocr_context = ""
        if a.get("ocr_texts"):
            ocr_context = f" Text visible: {', '.join(a['ocr_texts'])}"
        
        combined_text = f"At {a['timestamp']:.2f} seconds: {a['description']}{ocr_context}"
        
        docs.append(combined_text)
        metas.append({
            "frame_idx": a["frame_idx"],
            "timestamp": a["timestamp"],
            "image_path": a["image_path"],
            "ocr_path": a.get("ocr_path", ""),
            "caption_path": a.get("caption_path", ""),
            "has_text": len(a.get("ocr_texts", [])) > 0
        })
        ids.append(f"frame_{a['frame_idx']}")
    
    if docs:  # Only add if there are documents
        col.add(documents=docs, metadatas=metas, ids=ids)
    return col

# ────────────────────────────────────────────────────────────────────────────────
# NATURAL LANGUAGE GENERATION
def generate_natural_response(context: str, question: str, session_id: str) -> List[str]:
    """Generate natural, human-like responses, splitting if needed."""
    
    # First, get a complete answer
    prompt = f"""
You are describing a video to someone who hasn't seen it. Use natural, conversational language.

Based on these observations:
{context}

Answer this question: {question}

Guidelines:
- Speak naturally as if describing what you saw to a friend
- Be specific and detailed where relevant
- Don't mention technical terms like "frames", "OCR", "captions"
- If describing text you saw, just say "I saw" or "there was text that said"
- Focus on answering exactly what was asked
- Use descriptive, engaging language
"""
    
    r = requests.post(
        f"{OLLAMA_API_URL}/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.4, "num_predict": 4000},
        },
        timeout=600,
    )
    
    full_response = r.json().get("response", "I couldn't analyze that aspect of the video.")
    
    # Split response if it's too long
    max_chunk_size = 2000
    if len(full_response) > max_chunk_size:
        # Split into logical chunks at paragraph boundaries
        paragraphs = full_response.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 < max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Add continuation indicators
        for i in range(len(chunks)):
            if i < len(chunks) - 1:
                chunks[i] += "\n\n*[Continuing...]*"
            if i > 0:
                chunks[i] = "*[Continued from above]*\n\n" + chunks[i]
        
        return chunks
    else:
        return [full_response]

def query_analysis_enhanced(session_id: str, question: str) -> List[str]:
    """Enhanced query with natural language responses."""
    name = f"video_analysis_{session_id}"
    col = chroma_client.get_collection(name=name)
    
    # Get more results for better context
    res = col.query(query_texts=[question], n_results=10)
    
    if not res["documents"][0]:
        return ["I couldn't find any relevant information about that in the video."]
    
    # Build comprehensive context
    seen_timestamps = set()
    context_parts = []
    
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        timestamp = meta["timestamp"]
        if timestamp not in seen_timestamps:
            seen_timestamps.add(timestamp)
            context_parts.append(doc)
    
    context = "\n\n".join(context_parts)
    
    return generate_natural_response(context, question, session_id)

def generate_comprehensive_report(session_id: str, video_name: str) -> str:
    """Generate a natural, comprehensive report."""
    name = f"video_analysis_{session_id}"
    col = chroma_client.get_collection(name=name)
    res = col.get()
    
    if not res["documents"]:
        return "No analysis data available for this video."
    
    # Sort by timestamp
    sorted_data = sorted(zip(res["documents"], res["metadatas"]), 
                        key=lambda x: x[1]["timestamp"])
    
    # Group similar content
    timeline_events = []
    for doc, meta in sorted_data:
        timeline_events.append({
            "time": meta["timestamp"],
            "description": doc,
            "has_text": meta.get("has_text", False)
        })
    
    # Create context for report
    context = "\n\n".join([f"At {e['time']:.1f}s: {e['description']}" for e in timeline_events])
    
    prompt = f"""
Create a comprehensive, natural report about this video. Write as if you're describing the video to someone who needs to understand what it contains.

Video name: {video_name}

Observations:
{context}

Structure your report with:
1. An engaging overview of what the video shows
2. A narrative description of how the video unfolds over time
3. Key moments and important details
4. Any text or information that appeared in the video
5. Overall insights and conclusions

Write in a professional but accessible style, as if preparing a report for a colleague.
"""
    
    r = requests.post(
        f"{OLLAMA_API_URL}/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.4, "num_predict": 3000},
        },
        timeout=600,
    )
    
    return r.json().get("response", "Failed to generate report.")

# ────────────────────────────────────────────────────────────────────────────────
# CACHE MANAGEMENT
def clear_session_cache(session_id: str):
    """Remove all cached data for this session."""
    for dir_path in [FRAMES_DIR / session_id, CAPTIONS_DIR / session_id, OCR_DIR / session_id]:
        shutil.rmtree(dir_path, ignore_errors=True)
    
    try:
        chroma_client.delete_collection(name=f"video_analysis_{session_id}")
    except chromadb.errors.InvalidCollectionError:
        pass
    
    st.session_state.pop("video_analyzed", None)
    st.session_state.pop("analyses", None)
    st.session_state.pop("messages", None)

# ────────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
st.title("🎬 Intelligent Video Analysis System v3.0")
st.caption(f"Enhanced with OCR and Natural Language Processing • Session: {session_id}")

uploaded_file = st.file_uploader(
    "Upload your video file (max 200 MB)", 
    type=["mp4", "avi", "mov", "mkv", "webm"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name
    
    col1, col2 = st.columns(2)
    with col1:
        st.video(video_path)
    
    with col2:
        st.markdown("### Analysis Options")
        similarity_threshold = st.slider(
            "Frame similarity threshold", 
            0.7, 0.95, 0.85,
            help="Higher values = fewer similar frames"
        )
        
        if st.button("🚀 Start Intelligent Analysis", type="primary"):
            start_time = time.time()
            
            with st.spinner("📹 Extracting key frames intelligently..."):
                frames = extract_intelligent_frames(video_path, similarity_threshold)
                st.success(f"✅ Selected {len(frames)} unique key frames")
            
            analyses = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (fid, frm, ts) in enumerate(frames, 1):
                status_text.text(f"🔍 Analyzing frame {i}/{len(frames)} (OCR + Vision)...")
                progress_bar.progress(i / len(frames))
                
                img_path = save_frame(frm, session_id, fid)
                analysis = analyze_frame_comprehensive(
                    frm, img_path, fid, ts, session_id, analyses
                )
                
                if analysis:  # Only add if not similar to previous
                    analyses.append(analysis)
            
            status_text.text("💾 Storing in vector database...")
            store_in_chromadb_enhanced(session_id, analyses)
            
            status_text.text("📝 Generating comprehensive report...")
            report = generate_comprehensive_report(session_id, uploaded_file.name)
            
            # Store state
            st.session_state.video_analyzed = True
            st.session_state.analyses = analyses
            st.session_state.messages = [{"role": "assistant", "content": report}]
            
            # Clean up UI
            status_text.empty()
            progress_bar.empty()
            
            # Show stats
            elapsed_time = time.time() - start_time
            st.success(f"""
            ✅ Analysis Complete!
            - Processed {len(analyses)} unique frames
            - Extracted text from {sum(1 for a in analyses if a.get('ocr_texts'))} frames
            - Time taken: {elapsed_time:.1f} seconds
            """)
            
            os.unlink(video_path)

# ────────────────────────────────────────────────────────────────────────────────
# CHAT INTERFACE
if st.session_state.get("video_analyzed"):
    st.divider()
    st.subheader("💬 Ask Questions About the Video")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], list):
                for part in msg["content"]:
                    st.write(part)
                    if part != msg["content"][-1]:
                        st.divider()
            else:
                st.write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know about the video?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_parts = query_analysis_enhanced(session_id, prompt)
                
                # Display multi-part responses
                for i, part in enumerate(response_parts):
                    st.write(part)
                    if i < len(response_parts) - 1:
                        st.divider()
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_parts
                })

# ────────────────────────────────────────────────────────────────────────────────
# SIDEBAR
with st.sidebar:
    st.subheader("📊 Analysis Details")
    
    if st.session_state.get("analyses"):
        st.metric("Frames Analyzed", len(st.session_state.get("analyses", [])))
        
        frames_with_text = sum(1 for a in st.session_state.analyses if a.get("ocr_texts"))
        st.metric("Frames with Text", frames_with_text)
        
        with st.expander("🖼️ View Analyzed Frames"):
            for a in st.session_state.get("analyses", []):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(a["image_path"], use_container_width=True)
                with col2:
                    st.caption(f"**Time:** {a['timestamp']:.1f}s")
                    if a.get("ocr_texts"):
                        st.caption(f"**Text found:** {', '.join(a['ocr_texts'][:3])}...")
                st.divider()
    
    st.subheader("⚙️ Cache Management")
    if st.button("🗑️ Clear Session Cache"):
        clear_session_cache(session_id)
        st.success("✅ Session cache cleared!")
        st.info("Refresh the page to start a new analysis.")
    
    with st.expander("📁 Cache Statistics"):
        if session_id:
            frame_count = len(list((FRAMES_DIR / session_id).glob("*.jpg"))) if (FRAMES_DIR / session_id).exists() else 0
            caption_count = len(list((CAPTIONS_DIR / session_id).glob("*.txt"))) if (CAPTIONS_DIR / session_id).exists() else 0
            ocr_count = len(list((OCR_DIR / session_id).glob("*.txt"))) if (OCR_DIR / session_id).exists() else 0
            
            st.write(f"**Frames cached:** {frame_count}")
            st.write(f"**Captions saved:** {caption_count}")
            st.write(f"**OCR results:** {ocr_count}")