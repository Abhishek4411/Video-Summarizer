import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import streamlit as st
import cv2
import numpy as np
import tempfile
import shutil
from pathlib import Path
import base64
import hashlib
import json
from datetime import datetime
import chromadb
from chromadb.config import Settings
import requests
from PIL import Image
import io
from typing import List, Dict, Tuple
import time

# Initialize ChromaDB client with local storage
CACHE_DIR = Path("./cache")
FRAMES_DIR = CACHE_DIR / "frames"
DB_DIR = CACHE_DIR / "chromadb"

# Create directories
CACHE_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

# Ollama API configuration
OLLAMA_API_URL = "http://localhost:11435/api"
VISION_MODEL = "llama3.2-vision:latest"  # Using the most recent vision model
CHAT_MODEL = "llama3.3:latest"  # For final analysis

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(
    path=str(DB_DIR),
    settings=Settings(anonymized_telemetry=False)
)

def get_session_id():
    """Generate unique session ID for each user"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(
            f"{datetime.now().isoformat()}_{np.random.randint(1000000)}".encode()
        ).hexdigest()[:12]
    return st.session_state.session_id

def extract_key_frames(video_path: str, threshold: float = 30.0) -> List[Tuple[int, np.ndarray, float]]:
    """Extract key frames based on frame difference"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    prev_frame = None
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale for comparison
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(prev_frame, gray)
            mean_diff = np.mean(diff)
            
            # If significant change or first/last frame
            if mean_diff > threshold or frame_count == 0 or frame_count == total_frames - 1:
                timestamp = frame_count / fps
                frames.append((frame_count, frame, timestamp))
        else:
            # Always include first frame
            frames.append((0, frame, 0.0))
        
        prev_frame = gray
        frame_count += 1
        
        # Also sample every second to ensure coverage
        if frame_count % int(fps) == 0:
            timestamp = frame_count / fps
            frames.append((frame_count, frame, timestamp))
    
    cap.release()
    
    # Remove duplicates while preserving order
    unique_frames = []
    seen = set()
    for f in frames:
        if f[0] not in seen:
            unique_frames.append(f)
            seen.add(f[0])
    
    return sorted(unique_frames, key=lambda x: x[0])

def save_frame(frame: np.ndarray, session_id: str, frame_idx: int) -> str:
    """Save frame to disk and return path"""
    session_dir = FRAMES_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    
    frame_path = session_dir / f"frame_{frame_idx:06d}.jpg"
    cv2.imwrite(str(frame_path), frame)
    return str(frame_path)

def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 for Ollama API"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def analyze_frame_with_vision(image_path: str, frame_idx: int, timestamp: float) -> Dict:
    """Analyze single frame with vision model"""
    try:
        # Prepare detailed prompt for comprehensive analysis
        prompt = f"""Analyze this video frame captured at {timestamp:.2f} seconds (frame {frame_idx}).

Provide an extremely detailed and comprehensive analysis including:

1. **Scene Overview**: Describe the entire scene, environment, and setting in detail.

2. **Objects and Elements**: List and describe EVERY visible object, their positions, colors, textures, and relationships to each other.

3. **Actions and Movement**: Describe any actions, movements, or dynamic elements visible in the frame. Note any motion blur or indicators of movement.

4. **Technical Details**: 
   - Lighting conditions and shadows
   - Camera angle and perspective
   - Image quality and clarity
   - Any text, numbers, or readings visible (especially important for meter readings)

5. **Changes from Context**: Based on this being frame {frame_idx}, describe what might have changed or what this frame represents in the video sequence.

6. **Specific Observations**: Pay special attention to:
   - Any numerical displays or meter readings
   - Text or labels
   - Warning signs or indicators
   - Unusual or noteworthy details

Be extremely descriptive and thorough. This analysis will be used to understand the video content in detail."""

        # Call Ollama vision API
        response = requests.post(
            f"{OLLAMA_API_URL}/generate",
            json={
                "model": VISION_MODEL,
                "prompt": prompt,
                "images": [encode_image_base64(image_path)],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1000  # Allow for long responses
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "description": result.get("response", ""),
                "image_path": image_path
            }
        else:
            return {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "description": f"Error analyzing frame: {response.status_code}",
                "image_path": image_path
            }
            
    except Exception as e:
        return {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "description": f"Error: {str(e)}",
            "image_path": image_path
        }

def store_in_chromadb(session_id: str, analyses: List[Dict]):
    """Store frame analyses in ChromaDB"""
    try:
        collection_name = f"video_analysis_{session_id}"
        
        # Create or get collection
        try:
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except:
            collection = chroma_client.get_collection(name=collection_name)
        
        # Prepare data for storage
        documents = []
        metadatas = []
        ids = []
        
        for analysis in analyses:
            doc = f"Frame {analysis['frame_idx']} at {analysis['timestamp']:.2f}s: {analysis['description']}"
            documents.append(doc)
            metadatas.append({
                "frame_idx": analysis['frame_idx'],
                "timestamp": analysis['timestamp'],
                "image_path": analysis['image_path']
            })
            ids.append(f"frame_{analysis['frame_idx']}")
        
        # Add to collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return collection
        
    except Exception as e:
        st.error(f"ChromaDB error: {str(e)}")
        return None

def generate_comprehensive_analysis(session_id: str, video_name: str) -> str:
    """Generate comprehensive video analysis using all frame descriptions"""
    try:
        collection_name = f"video_analysis_{session_id}"
        collection = chroma_client.get_collection(name=collection_name)
        
        # Get all documents
        results = collection.get()
        
        # Sort by frame index
        frame_data = []
        for i in range(len(results['documents'])):
            frame_data.append({
                'document': results['documents'][i],
                'metadata': results['metadatas'][i]
            })
        
        frame_data.sort(key=lambda x: x['metadata']['frame_idx'])
        
        # Prepare context for LLM
        context = f"Video Analysis for: {video_name}\n\n"
        context += "Frame-by-frame descriptions:\n\n"
        
        for data in frame_data:
            context += data['document'] + "\n\n"
        
        # Generate comprehensive analysis
        prompt = f"""Based on the following frame-by-frame analysis of a video, provide a comprehensive summary and analysis:

{context}

Please provide:

1. **Executive Summary**: A concise overview of what happens in the video.

2. **Detailed Timeline**: A chronological breakdown of key events and changes throughout the video.

3. **Key Observations**: 
   - Important patterns or trends observed
   - Significant changes or transitions
   - Any readings, measurements, or data captured

4. **Technical Analysis**:
   - Quality and consistency of the video
   - Any technical issues or noteworthy aspects

5. **Conclusions and Insights**: What can be concluded from this video analysis?

Be thorough and connect the individual frame descriptions into a coherent narrative."""

        response = requests.post(
            f"{OLLAMA_API_URL}/generate",
            json={
                "model": CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 2000
                }
            }
        )
        
        if response.status_code == 200:
            return response.json().get("response", "Analysis failed")
        else:
            return f"Error generating analysis: {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"

def query_video_analysis(session_id: str, query: str) -> str:
    """Query the video analysis with specific questions"""
    try:
        collection_name = f"video_analysis_{session_id}"
        collection = chroma_client.get_collection(name=collection_name)
        
        # Search for relevant frames
        results = collection.query(
            query_texts=[query],
            n_results=5
        )
        
        if not results['documents'][0]:
            return "No relevant information found for your query."
        
        # Prepare context
        context = "Relevant frame descriptions:\n\n"
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            context += f"Frame {metadata['frame_idx']} at {metadata['timestamp']:.2f}s:\n"
            context += doc + "\n\n"
        
        # Generate answer
        prompt = f"""Based on the following video frame descriptions, answer the user's question:

{context}

User Question: {query}

Provide a detailed and accurate answer based on the video content."""

        response = requests.post(
            f"{OLLAMA_API_URL}/generate",
            json={
                "model": CHAT_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json().get("response", "Failed to generate answer")
        else:
            return f"Error: {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Video Analyzer", layout="wide")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'video_analyzed' not in st.session_state:
    st.session_state.video_analyzed = False
if 'analyses' not in st.session_state:
    st.session_state.analyses = []

session_id = get_session_id()

st.title("Video Analysis System")

# Video upload
uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file is not None:
    # Save uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.video(video_path)
    
    with col2:
        if st.button("Analyze Video", type="primary"):
            with st.spinner("Extracting key frames..."):
                frames = extract_key_frames(video_path)
                st.info(f"Extracted {len(frames)} key frames")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            analyses = []
            for i, (frame_idx, frame, timestamp) in enumerate(frames):
                status_text.text(f"Analyzing frame {i+1}/{len(frames)}...")
                progress_bar.progress((i + 1) / len(frames))
                
                # Save frame
                frame_path = save_frame(frame, session_id, frame_idx)
                
                # Analyze frame
                analysis = analyze_frame_with_vision(frame_path, frame_idx, timestamp)
                analyses.append(analysis)
                
                # Show preview
                with st.expander(f"Frame {frame_idx} at {timestamp:.2f}s"):
                    st.image(frame_path, width=300)
                    st.write(analysis['description'])
            
            # Store in ChromaDB
            status_text.text("Storing in vector database...")
            collection = store_in_chromadb(session_id, analyses)
            
            # Generate comprehensive analysis
            status_text.text("Generating comprehensive analysis...")
            comprehensive_analysis = generate_comprehensive_analysis(session_id, uploaded_file.name)
            
            st.session_state.video_analyzed = True
            st.session_state.analyses = analyses
            st.session_state.messages.append({
                "role": "assistant",
                "content": comprehensive_analysis
            })
            
            status_text.text("Analysis complete!")
            progress_bar.progress(1.0)
            
            # Clean up temp file
            os.unlink(video_path)

# Chat interface
if st.session_state.video_analyzed:
    st.divider()
    st.subheader("Chat with Video Analysis")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about the video..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_video_analysis(session_id, prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with frame gallery
if st.session_state.analyses:
    with st.sidebar:
        st.subheader("Analyzed Frames")
        for analysis in st.session_state.analyses:
            with st.expander(f"Frame {analysis['frame_idx']}"):
                st.image(analysis['image_path'], use_container_width=True)
                st.caption(f"Time: {analysis['timestamp']:.2f}s")
                st.text(analysis['description'][:200] + "...")