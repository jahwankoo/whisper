import os
import io
import queue
import threading
import tempfile
from datetime import datetime, timedelta

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
    # base/medium/large-v3 등 선택 가능
    return whisper.load_model(model_name)

model_name = st.selectbox("Model", ["base", "small", "medium", "large-v3"], index=0)
model = load_model(model_name)

# ---------- 파라미터 ----------
SAMPLE_RATE = 48000        # WebRTC 기본
CHANNELS = 1               # 단일 채널로 다운믹스
CHUNK_SECONDS = st.slider("Chunk seconds", 2, 10, 5)   # 몇 초 단위로 잘라서 전사할지
OVERLAP_SECONDS = st.slider("Overlap seconds", 0, 3, 1)  # 경계 단어 보완용
LANG = st.selectbox("Language (optional)", ["auto", "ko", "en"], index=0)
AUTO_LANG = (LANG == "auto")

# ---------- 상태 ----------
if "audio_q" not in st.session_state:
    st.session_state.audio_q = queue.Queue()
if "caption" not in st.session_state:
    st.session_state.caption = ""
if "running" not in st.session_state:
    st.session_state.running = False
if "last_saved_wav" not in st.session_state:
    st.session_state.last_saved_wav = None

caption_box = st.empty()
status_box = st.empty()

# ---------- 오디오 수신 처리 ----------
class AudioProcessor:
    def __init__(self):
        self.buffer = []

    def recv(self, frame: av.AudioFrame):
        # float32 ndarray [channels, samples]
        pcm = frame.to_ndarray().astype(np.float32)
        # 다운믹스 → mono
        if pcm.ndim == 2 and pcm.shape[0] > 1:
            pcm = np.mean(pcm, axis=0, keepdims=True)
        st.session_state.audio_q.put(pcm.squeeze())

# ---------- 전사용 워커 스레드 ----------
def transcribe_worker():
    """
    오디오 큐에서 샘플을 읽어 일정 길이(CHUNK_SECONDS)마다 WAV로 저장 후 Whisper 전사,
    결과를 자막창에 누적 표시. 경계 품질을 위해 OVERLAP_SECONDS 만큼 앞부분을 합침.
    """
    sr = SAMPLE_RATE
    chunk_len = int(CHUNK_SECONDS * sr)
    overlap_len = int(OVERLAP_SECONDS * sr)

    ring = np.zeros(0, dtype=np.float32)
    last_tail = np.zeros(0, dtype=np.float32)

    status_box.info("Listening… streaming transcription in progress.")

    while st.session_state.running:
        # 큐에서 가용 샘플 최대한 모으기 (non-block)
        got_any = False
        while True:
            try:
                part = st.session_state.audio_q.get(timeout=0.1)
                ring = np.concatenate([ring, part])
                got_any = True
            except queue.Empty:
                break

        if not got_any:
            continue

        # 충분히 모이면 전사 실행
        while len(ring) >= chunk_len:
            # overlap을 앞에 붙여서 자연스럽게
            start_idx = 0
            seg = ring[start_idx:start_idx+chunk_len]

            # 다음 라운드를 위해 ring 줄이기
            ring = ring[chunk_len:]

            # 경계 보완: 이전 tail + 현재 chunk 결합
            if overlap_len > 0:
                seg_for_stt = np.concatenate([last_tail, seg])
                last_tail = seg[-overlap_len:].copy()
            else:
                seg_for_stt = seg

            # WAV로 저장 후 전사
            with tempfile.TemporaryDirectory() as td:
                wav_path = os.path.join(td, "seg.wav")
                write(wav_path, sr, seg_for_stt)
                # 언어 옵션
                kwargs = {}
                if not AUTO_LANG:
                    kwargs["language"] = LANG
                # 전사
                try:
                    result = model.transcribe(wav_path, **kwargs)
                    text = result.get("text", "").strip()
                except Exception as e:
                    text = f"[ERR:{e}]"

            if text:
                # 자막 누적(최근 10줄만 유지)
                new_caption = (st.session_state.caption + " " + text).strip()
                lines = new_caption.split()
                if len(lines) > 500:  # 너무 길어지면 앞부분 절단
                    new_caption = " ".join(lines[-500:])
                st.session_state.caption = new_caption
                caption_box.markdown(f"**Live captions:**\n\n{st.session_state.caption}")

# ---------- WebRTC ----------
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
webrtc_ctx = webrtc_streamer(
    key="whisper-streaming",
    mode=WebRtcMode.RECVONLY,
    rtc_configuration=rtc_config,
    media_stream_constraints={"audio": True, "video": False},
)

# ---------- 컨트롤 ----------
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Start"):
        if not st.session_state.running:
            st.session_state.caption = ""
            st.session_state.running = True
            # 워커 스레드 기동
            t = threading.Thread(target=transcribe_worker, daemon=True)
            t.start()
with col2:
    if st.button("Stop"):
        st.session_state.running = False
        status_box.info("Stopped.")
with col3:
    if st.button("Clear Captions"):
        st.session_state.caption = ""
        caption_box.empty()

# ---------- 오디오 프레임 수신 루프 ----------
if webrtc_ctx and webrtc_ctx.state.playing:
    # WebRTC에서 오는 오디오 프레임을 지속 수신
    try:
        while True:
            frame = webrtc_ctx.audio_receiver.get_frame(timeout=0.01)
            AudioProcessor().recv(frame)
    except queue.Empty:
        pass
    except Exception:
        pass

# ---------- 최종 녹음 파일 저장(선택) ----------
st.markdown("---")
st.subheader("Save last N seconds as WAV")
save_sec = st.slider("Seconds to save", 3, 60, 10)
if st.button("Save Snippet"):
    # 큐에 남아있는 것들을 가능한 만큼 모아 WAV로 저장
    samples = []
    try:
        while True:
            samples.append(st.session_state.audio_q.get_nowait())
    except queue.Empty:
        pass
    if samples:
        buf = np.concatenate(samples).astype(np.float32)
        target_len = int(save_sec * SAMPLE_RATE)
        buf = buf[-target_len:] if len(buf) > target_len else buf
        td = tempfile.mkdtemp()
        path = os.path.join(td, f"snippet_{datetime.now().strftime('%H%M%S')}.wav")
        write(path, SAMPLE_RATE, buf)
        st.session_state.last_saved_wav = path
        st.success(f"Saved: {path}")
        st.audio(path)
        with open(path, "rb") as f:
            st.download_button("⬇️ Download WAV", f, file_name=os.path.basename(path))
    else:
        st.warning("No audio buffered yet.")
