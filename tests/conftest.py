"""Configuration et fixtures pytest."""

import io
import sys
import wave
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Injecter la racine du projet et le dossier src dans sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

for path in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.main import app


@pytest.fixture(scope="session")
def client():
    """Fixture retournant un client de test FastAPI."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def wav_bytes():
    """Fabrique un petit fichier WAV mono 16 bits en memoire."""
    def _creer():
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
        return buffer.getvalue()

    return _creer
