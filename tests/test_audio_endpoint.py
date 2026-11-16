"""Tests d'integration de la route /audio/transcribe."""

from unittest.mock import MagicMock, patch


def test_transcribe_success(client, wav_bytes):
    faux_modele = MagicMock()
    faux_modele.transcribe.return_value = {
        "transcription": "Note d'audit client 42.",
        "language": "fr",
        "execution_time_seconds": 0.82,
    }
    with patch("backend.main.audio_transcriber", faux_modele):
        response = client.post("/audio/transcribe", files={"file": ("note.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "note.wav"
    assert data["transcription"] == "Note d'audit client 42."


def test_transcribe_sans_fichier(client):
    response = client.post("/audio/transcribe")
    assert response.status_code == 422


def test_transcribe_fichier_invalide(client):
    response = client.post("/audio/transcribe", files={"file": ("note.txt", b"du texte", "text/plain")})
    assert response.status_code == 422


def test_transcribe_erreur_interne(client, wav_bytes):
    faux_modele = MagicMock()
    faux_modele.transcribe.side_effect = RuntimeError("boom")
    with patch("backend.main.audio_transcriber", faux_modele):
        response = client.post("/audio/transcribe", files={"file": ("note.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 500
