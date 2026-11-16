"""Module de transcription vocale base sur Whisper Tiny (Hugging Face)."""

import io
import time
import wave
import numpy as np
from transformers import pipeline
from iatos_core.logger import logger


class AudioTranscriber:
    """Transcripteur de notes vocales utilisant openai/whisper-tiny."""

    def __init__(self):
        self.asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", chunk_length_s=30)
        logger.info("Pipeline Whisper Tiny initialise.")

    def transcribe(self, audio_bytes: bytes) -> dict:
        """Transcrit un fichier WAV 16 bits en texte."""
        try:
            with wave.open(io.BytesIO(audio_bytes)) as wav:
                if wav.getsampwidth() != 2:
                    raise ValueError("Le fichier WAV doit etre en 16 bits.")
                rate = wav.getframerate()
                channels = wav.getnchannels()
                frames = wav.readframes(wav.getnframes())
        except (wave.Error, EOFError):
            raise ValueError("Fichier audio invalide, un WAV est attendu.")

        signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            signal = signal.reshape(-1, channels).mean(axis=1)
        if signal.size == 0:
            raise ValueError("Le fichier audio est vide.")

        # Whisper attend du 16 kHz
        positions = np.linspace(0, signal.size - 1, int(signal.size * 16000 / rate))
        signal = np.interp(positions, np.arange(signal.size), signal).astype(np.float32)

        debut = time.perf_counter()
        sortie = self.asr(
            {"raw": signal, "sampling_rate": 16000},
            generate_kwargs={"language": "french", "task": "transcribe"},
        )
        duree = time.perf_counter() - debut

        texte = sortie["text"].strip()
        logger.info("Transcription en {t:.2f}s : '{txt}'", t=duree, txt=texte[:40])
        return {"transcription": texte, "language": "fr", "execution_time_seconds": round(duree, 3)}
