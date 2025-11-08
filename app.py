import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import tempfile
import os

# Whisper 모델 로드 (최초 1회만 로드됨)
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

st.title("🎙 Whisper AI Voice Recorder & Transcriber")

# 녹음 설정
fs = 16000  # 샘플링 주파수
duration = st.slider("Recording duration (seconds)", 3, 30, 10)

if st.button("Start Recording"):
    st.info("Recording... Speak now.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    # 임시 파일 저장
    tmp_dir = tempfile.mkdtemp()
    wav_path = os.path.join(tmp_dir, "recorded.wav")
    write(wav_path, fs, audio)
    st.success(f"Saved: {wav_path}")
    st.audio(wav_path)

    # Whisper로 음성 텍스트 변환
    with st.spinner("Transcribing..."):
        result = model.transcribe(wav_path, language="ko")
    st.subheader("📝 Transcribed Text:")
    st.write(result["text"])

    # 다운로드 버튼
    with open(wav_path, "rb") as f:
        st.download_button("⬇️ Download Recording", f, file_name="recorded.wav")
