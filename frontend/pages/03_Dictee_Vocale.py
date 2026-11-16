"""Page 3 : Dictee vocale et transcription de notes d'audit (Whisper Tiny)."""

import os
from datetime import datetime
import requests
import streamlit as st

FASTAPI_URL = os.getenv("FASTAPI_BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Dictee Vocale | Brief 2",
    page_icon="🎙️",
    layout="wide",
)

st.title("Page 3 : Dictee Vocale des Notes d'Audit")
st.markdown("""
Enregistrez votre note au microphone : elle est transcrite par le modele **Whisper Tiny** (`openai/whisper-tiny`).
""")

audio = st.audio_input("Enregistrer une note vocale")

if audio:
    st.audio(audio)

    if st.button("Transcrire la note"):
        files = {"file": ("note_vocale.wav", audio.getvalue(), "audio/wav")}
        with st.spinner("Transcription en cours..."):
            response = requests.post(f"{FASTAPI_URL}/audio/transcribe", files=files, timeout=60)

        if response.status_code == 200:
            data = response.json()
            c1, c2 = st.columns(2)
            c1.metric("Temps d'inférence (s)", data["execution_time_seconds"])
            c2.metric("Langue", data["language"])

            texte = st.text_area("Texte transcrit", value=data["transcription"], height=150)
            compte_rendu = f"Note d'audit du {datetime.now():%d/%m/%Y %H:%M}\n\n{texte}"
            st.download_button("Exporter le compte-rendu", compte_rendu, file_name="note_audit.txt")
        else:
            st.error(f"Erreur {response.status_code} : {response.json().get('detail')}")
