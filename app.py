import os
import io
import queue
import threading
import tempfile
from datetime import datetime
import time

import av
import numpy as np
import streamlit as st
from scipy.io.wavfile import write
from streamlit_webrtc import WebRtcMode, webrtc_streamer, RTCConfiguration
import whisper

st.set_page_config(page_title="Whisper Streaming STT", layout="centered")
st.title("🎤 Whisper Streaming STT (Browser Mic → Live Captions)")

# ---------- Whisper 모델 로딩 (캐시) ----------
@st.cache_resource
def load_model(model_name="base"):
    return whisper.load_model(model_name)

# 메모리 문제로 base만 권장
model_name = st.selectbox("Model", ["tiny", "base"], index=1, 
                          help="Streamlit Cloud에서는 tiny/base만 권장")
model = load_model(model_name)

# ---------- 파라미터 ----------
SAMPLE_RATE = 16000        # Whisper는 16kHz 사용
CHANNELS = 1
CHUNK_SECONDS = st.slider("Chunk seconds", 2, 10, 5)
OVERLAP_SECONDS = st.slider("Overlap seconds", 0, 3, 1)
LANG = st.selectbox("Language", ["auto", "ko", "en"], index=1)

# ---------- 상태 초기화 ----------
if "audio_q" not in st.session_state:
    st.session_state.audio_q = queue.Queue()
if "transcript_q" not in st.session_state:
    st.session_state.transcript_q = queue.Queue()
if "caption" not in st.session_state:
    st.session_state.caption = ""
if "running" not in st.session_state:
    st.session_state.running = False
if "worker_thread" not in st.session_state:
    st.session_state.worker_thread = None

# ---------- 오디오 프레임 콜백 ----------
def audio_frame_callback(frame: av.AudioFrame):
    sound = frame.to_ndarray()
    
    # 리샘플링 (48000 → 16000)
    if frame.sample_rate != SAMPLE_RATE:
        # 간단한 decimation (실제로는 librosa 사용 권장)
        ratio = frame.sample_rate // SAMPLE_RATE
        sound = sound[::ratio]
    
    # 모노로 변환
    if sound.ndim == 2:
        sound = sound.mean(axis=1)
    
    sound = sound.astype(np.float32)
    st.session_state.audio_q.put(sound)
    
    return frame

# ---------- 전사 워커 스레드 ----------
def transcribe_worker():
    sr = SAMPLE_RATE
    chunk_len = int(CHUNK_SECONDS * sr)
    overlap_len = int(OVERLAP_SECONDS * sr)
    
    ring = np.zeros(0, dtype=np.float32)
    last_tail = np.zeros(0, dtype=np.float32)
    
    while st.session_state.running:
        # 큐에서 오디오 수집
        collected = []
        try:
            while len(collected) < 10:  # 최대 10개 배치
                part = st.session_state.audio_q.get(timeout=0.1)
                collected.append(part)
        except queue.Empty:
            if not collected:
                continue
        
        if collected:
            ring = np.concatenate([ring] + collected)
        
        # 충분히 모이면 전사
        while len(ring) >= chunk_len:
            seg = ring[:chunk_len]
            ring = ring[chunk_len:]
            
            # Overlap 처리
            if overlap_len > 0 and len(last_tail) > 0:
                seg_for_stt = np.concatenate([last_tail, seg])
                last_tail = seg[-overlap_len:].copy()
            else:
                seg_for_stt = seg
                if overlap_len > 0:
                    last_tail = seg[-overlap_len:].copy()
            
            # 전사 실행
            try:
                # Whisper는 float32 [-1, 1] 범위 필요
                seg_for_stt = seg_for_stt.clip(-1.0, 1.0)
                
                kwargs = {"fp16": False}  # CPU 사용
                if LANG != "auto":
                    kwargs["language"] = LANG
                
                result = model.transcribe(seg_for_stt, **kwargs)
                text = result.get("text", "").strip()
                
                if text:
                    # 결과를 큐에 넣기 (메인 스레드에서 표시)
                    st.session_state.transcript_q.put(text)
                    
            except Exception as e:
                st.session_state.transcript_q.put(f"[Error: {str(e)}]")

# ---------- WebRTC 설정 ----------
st.info("🎤 아래 'START' 버튼을 클릭하여 마이크 권한을 허용하세요.")

rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_ctx = webrtc_streamer(
    key="whisper-streaming",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_config,
    media_stream_constraints={
        "audio": {
            "sampleRate": 48000,
            "channelCount": 1,
            "echoCancellation": True,
            "noiseSuppression": True,
        },
        "video": False
    },
    audio_frame_callback=audio_frame_callback,
    async_processing=True,
)

# ---------- 컨트롤 버튼 ----------
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎙️ Start Transcription"):
        if not st.session_state.running:
            st.session_state.running = True
            st.session_state.caption = ""
            
            # 워커 스레드 시작
            if st.session_state.worker_thread is None or not st.session_state.worker_thread.is_alive():
                st.session_state.worker_thread = threading.Thread(
                    target=transcribe_worker, 
                    daemon=True
                )
                st.session_state.worker_thread.start()
            
            st.success("✅ Transcription started!")

with col2:
    if st.button("⏹️ Stop"):
        st.session_state.running = False
        st.info("⏸️ Transcription stopped.")

with col3:
    if st.button("🗑️ Clear"):
        st.session_state.caption = ""
        # 큐 비우기
        while not st.session_state.transcript_q.empty():
            try:
                st.session_state.transcript_q.get_nowait()
            except queue.Empty:
                break

# ---------- 자막 표시 (메인 스레드) ----------
# 전사 큐에서 결과 가져오기
try:
    while not st.session_state.transcript_q.empty():
        text = st.session_state.transcript_q.get_nowait()
        st.session_state.caption += " " + text
        
        # 너무 길어지면 앞부분 제거
        words = st.session_state.caption.split()
        if len(words) > 500:
            st.session_state.caption = " ".join(words[-500:])
except queue.Empty:
    pass

# 자막 표시
st.markdown("---")
st.subheader("📝 Live Captions")
caption_container = st.container()
with caption_container:
    if st.session_state.caption:
        st.markdown(f"**{st.session_state.caption}**")
    else:
        st.info("음성 인식 결과가 여기에 표시됩니다...")

# ---------- 상태 표시 ----------
if webrtc_ctx.state.playing:
    st.success("🔴 Recording...")
else:
    st.warning("⚪ Not recording")

# 자동 새로고침 (전사 결과 업데이트용)
if st.session_state.running:
    time.sleep(0.5)
    st.rerun()

# ---------- 디버그 정보 ----------
with st.expander("🔧 Debug Info"):
    st.write(f"Audio queue size: {st.session_state.audio_q.qsize()}")
    st.write(f"Transcript queue size: {st.session_state.transcript_q.qsize()}")
    st.write(f"Worker thread alive: {st.session_state.worker_thread.is_alive() if st.session_state.worker_thread else False}")
