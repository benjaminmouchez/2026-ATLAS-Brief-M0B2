"""Module de transcription vocale base sur Whisper Tiny (Hugging Face)."""

import time
from transformers import pipeline
from iatos_core.logger import logger


class AudioTranscriber:
    """Transcripteur de notes vocales utilisant openai/whisper-tiny."""

    def __init__(self):
        self.asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", chunk_length_s=30)
        logger.info("Pipeline Whisper Tiny initialise.")

    def transcribe(self, audio_bytes: bytes) -> dict:
        """Transcrit un fichier audio en texte."""
        if not audio_bytes:
            raise ValueError("Le fichier audio est vide.")

        debut = time.perf_counter()
        sortie = self.asr(audio_bytes, generate_kwargs={"language": "french"})
        duree = time.perf_counter() - debut

        texte = sortie["text"].strip()
        logger.info("Transcription en {t:.2f}s : '{txt}'", t=duree, txt=texte[:40])
        return {"transcription": texte, "language": "fr", "execution_time_seconds": round(duree, 3)}
