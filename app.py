import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import whisper
import numpy as np
import tempfile
import os
from scipy.io.wavfile import write

st.set_page_config(page_title="Whisper AI Voice Recorder", layout="centered")
st.title("🎤 Whisper AI Voice Recorder & Transcriber")

# Whisper 모델 로드
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Audio Processor 정의
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        # 오디오 프레임 누적
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

# WebRTC 녹음 위젯
webrtc_ctx = webrtc_streamer(
    key="whisper-audio",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# Whisper 변환 버튼
if webrtc_ctx and webrtc_ctx.audio_receiver:
    if st.button("🎧 Stop & Transcribe"):
        with st.spinner("Processing..."):
            # 오디오 데이터 수집
            audio_frames = []
            while True:
                try:
                    frame = webrtc_ctx.audio_receiver.get_frame(timeout=1)
                except:
                    break
                audio_frames.append(frame.to_ndarray())

            if audio_frames:
                # 오디오 합치기
                audio = np.concatenate(audio_frames, axis=1).T.astype(np.float32)
                fs = 48000

                # 임시 파일 저장
                tmp_dir = tempfile.mkdtemp()
                wav_path = os.path.join(tmp_dir, "recorded.wav")
                write(wav_path, fs, audio)
                st.success(f"Saved: {wav_path}")
                st.audio(wav_path)

                # Whisper로 텍스트 변환
                result = model.transcribe(wav_path, language="ko")
                st.subheader("📝 Transcribed Text:")
                st.write(result["text"])

                # 다운로드 버튼
                with open(wav_path, "rb") as f:
                    st.download_button("⬇️ Download Recording", f, file_name="recorded.wav")
            else:
                st.warning("No audio data captured.")
