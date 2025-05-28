import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import time
from logic.speech_to_text import transcribe
from logic.text_generation import generate_response
from logic.text_to_speech import speak

# Page setup
st.set_page_config(page_title="TinyLlama Voice Assistant", layout="centered")
st.title("🎙️ TinyLlama Smart Voice Assistant (Continuous Mode)")

# Make audio folder
os.makedirs("audio", exist_ok=True)

# Record function
def record_audio(duration=5, filename="audio/input.wav"):
    fs = 16000
    st.info("🎧 Listening... Ask your question.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(filename, fs, np.int16(recording * 32767))
    return filename

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "listening" not in st.session_state:
    st.session_state.listening = False

# Start button
if st.button("▶️ Start Conversation"):
    st.session_state.listening = True

# Stop button
if st.session_state.listening:
    if st.button("🛑 Stop Listening"):
        st.session_state.listening = False

# Conversation loop
while st.session_state.get("listening", False):
    try:
        audio_file = record_audio()

        # Transcribe user input
        user_input = transcribe(audio_file)
        st.write("**You asked:**", user_input)
        st.session_state.chat_history.append(("You", user_input))

        # Generate response
        with st.spinner("🤖 Thinking..."):
            assistant_response = generate_response(user_input)
        st.write("**Assistant:**", assistant_response)
        st.session_state.chat_history.append(("Assistant", assistant_response))

        # Speak response
        speak(assistant_response)

        # Wait briefly before next loop
        time.sleep(1)

    except Exception as e:
        st.error(f"Error: {str(e)}")
        break

# Show full chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 🗣️ Conversation History")
    for speaker, message in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {message}")
