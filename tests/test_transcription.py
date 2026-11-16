"""Tests unitaires du transcripteur Whisper."""

from unittest.mock import MagicMock, patch
import pytest
from iatos_core.ml.transcription import AudioTranscriber


def test_transcribe(wav_bytes):
    fake_asr = MagicMock(return_value={"text": " Note d'audit client 42. "})
    with patch("iatos_core.ml.transcription.pipeline", return_value=fake_asr):
        res = AudioTranscriber().transcribe(wav_bytes())
    assert res["transcription"] == "Note d'audit client 42."
    assert res["language"] == "fr"
    assert res["execution_time_seconds"] >= 0


def test_transcribe_audio_vide():
    with patch("iatos_core.ml.transcription.pipeline"):
        with pytest.raises(ValueError):
            AudioTranscriber().transcribe(b"")
