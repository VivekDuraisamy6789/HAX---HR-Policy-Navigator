"""Voice I/O module for HR Policy Navigator (Sprint 3 — Amit-Enhanced).

Provides:
  1. text_to_speech(text)  -> bytes   — offline TTS via pyttsx3 (WAV bytes)
  2. transcribe(audio_bytes) -> str   — STT via SpeechRecognition or Whisper

Design constraints (Track B — no GPU, no frontier APIs):
  - pyttsx3: system TTS engine, zero network, runs on Windows/macOS/Linux
  - SpeechRecognition: lightweight, handles browser-recorded audio
  - Both degrade gracefully: ImportError -> empty string / silent, never crash the UI
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread lock — pyttsx3 engine is not thread-safe; guard with a mutex
# ---------------------------------------------------------------------------
_tts_lock = threading.Lock()
_tts_engine = None   # lazy-init


def _get_tts_engine():
    """Lazy-init pyttsx3 engine; returns None if not installed."""
    global _tts_engine
    if _tts_engine is not None:
        return _tts_engine
    try:
        import pyttsx3  # noqa: PLC0415
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)    # words/min — slightly slower for clarity
        engine.setProperty("volume", 0.95)
        voices = engine.getProperty("voices")
        if voices:
            en_voices = [v for v in voices if "english" in v.name.lower() or "en" in (v.id or "").lower()]
            if en_voices:
                engine.setProperty("voice", en_voices[min(1, len(en_voices) - 1)].id)
        _tts_engine = engine
        return engine
    except Exception as exc:
        logger.warning("pyttsx3 not available: %s", exc)
        return None


def text_to_speech(text: str, max_chars: int = 1500) -> Optional[bytes]:
    """
    Convert text to audio bytes (WAV format) using pyttsx3.

    Returns WAV bytes on success, None if pyttsx3 is unavailable or fails.
    Truncates very long answers to max_chars to keep latency under ~5s.
    """
    if not text or not text.strip():
        return None

    # Sanitise: strip markdown-ish syntax that sounds bad when read aloud
    import re
    clean = re.sub(r"\*\*|__|`{1,3}|\#{1,6}\s?|\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    clean = re.sub(r"\n{2,}", ". ", clean).replace("\n", " ").strip()
    clean = clean[:max_chars]

    with _tts_lock:
        engine = _get_tts_engine()
        if engine is None:
            return None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            engine.save_to_file(clean, tmp_path)
            engine.runAndWait()
            with open(tmp_path, "rb") as f:
                data = f.read()
            return data if data else None
        except Exception as exc:
            logger.warning("TTS synthesis failed: %s", exc)
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Speech-to-Text: SpeechRecognition + Whisper fallback
# ---------------------------------------------------------------------------
def stt_available() -> bool:
    """Return True if any speech-to-text engine is importable."""
    try:
        import speech_recognition  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import whisper  # noqa: F401
        return True
    except ImportError:
        pass
    return False


def whisper_available() -> bool:
    """Alias for backward compatibility."""
    return stt_available()


def tts_available() -> bool:
    """Return True if pyttsx3 is importable."""
    try:
        import pyttsx3  # noqa: F401
        return True
    except ImportError:
        return False


def transcribe(audio_bytes: bytes, language: str = "en") -> str:
    """
    Transcribe audio bytes to text using SpeechRecognition or Whisper.
    Returns the transcription string, or '' if unavailable or failed.
    """
    if not audio_bytes:
        return ""

    # Strategy 1: SpeechRecognition (fast, lightweight)
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with io.BytesIO(audio_bytes) as audio_file:
            with sr.AudioFile(audio_file) as source:
                audio_data = r.record(source)
        text = r.recognize_google(audio_data, language=language)
        if text:
            return text.strip()
    except Exception as exc:
        logger.debug("SpeechRecognition pass: %s", exc)

    # Strategy 2: Whisper (if installed)
    try:
        import whisper
        import numpy as np
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            model = whisper.load_model("tiny")
            result = model.transcribe(tmp_path, language=language)
            return result.get("text", "").strip()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as exc:
        logger.debug("Whisper pass: %s", exc)

    return ""
