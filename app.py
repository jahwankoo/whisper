import streamlit as st
import whisper
import tempfile
import os

st.set_page_config(page_title="Whisper STT", layout="centered")
st.title("🎤 Whisper Speech-to-Text")

@st.cache_resource
def load_model(model_name="base"):
    return whisper.load_model(model_name)

model_name = st.selectbox("Model", ["tiny", "base"], index=0)
model = load_model(model_name)

st.markdown("### 🎵 Upload Audio File")
uploaded_file = st.file_uploader(
    "Choose an audio file (MP3, WAV, M4A, etc.)", 
    type=["mp3", "wav", "m4a", "ogg", "flac"]
)

lang = st.selectbox("Language", ["auto", "ko", "en"], index=1)

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("🚀 Transcribe"):
        with st.spinner("Transcribing..."):
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            try:
                # 전사 실행
                kwargs = {"fp16": False}
                if lang != "auto":
                    kwargs["language"] = lang
                
                result = model.transcribe(tmp_path, **kwargs)
                
                st.success("✅ Transcription Complete!")
                st.markdown("### 📝 Result:")
                st.markdown(f"**{result['text']}**")
                
                # 상세 정보
                with st.expander("📊 Details"):
                    st.json({
                        "language": result.get("language"),
                        "duration": f"{result.get('segments', [{}])[-1].get('end', 0):.2f}s" if result.get('segments') else "N/A"
                    })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
            finally:
                # 임시 파일 삭제
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
