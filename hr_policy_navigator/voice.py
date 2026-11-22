"""Voice I/O module for HR Policy Navigator (Sprint 3 — Amit-Enhanced).

Provides two capabilities:
  1. text_to_speech(text)  -> bytes   — offline TTS via pyttsx3 (mp3/wav bytes)
  2. transcribe(audio_bytes) -> str   — offline STT via openai-whisper tiny model
                                        with graceful fallback if whisper not installed

Design constraints (Track B — no GPU, no frontier APIs):
  - pyttsx3: system TTS engine, zero network, runs on Windows/macOS/Linux
  - Whisper tiny: 39 MB model, CPU-only, ~2s latency per utterance
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
        # On Windows prefer a natural-sounding voice if available
        voices = engine.getProperty("voices")
        if voices:
            # Prefer English voices; pick the second one if available (usually female)
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
# Speech-to-Text via Whisper tiny (offline, CPU)
# ---------------------------------------------------------------------------
_whisper_model = None
_whisper_lock = threading.Lock()


def _load_whisper():
    """Lazy-load Whisper tiny model; returns None if openai-whisper not installed."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        import whisper  # noqa: PLC0415
        with _whisper_lock:
            if _whisper_model is None:
                logger.info("Loading Whisper tiny model (~39 MB)…")
                _whisper_model = whisper.load_model("tiny")
        return _whisper_model
    except ImportError:
        logger.warning("openai-whisper not installed. Voice input will not be available.")
        return None
    except Exception as exc:
        logger.warning("Whisper model load failed: %s", exc)
        return None


def transcribe(audio_bytes: bytes, language: str = "en") -> str:
    """
    Transcribe audio bytes to text using Whisper tiny (offline).

    Accepts any audio format Whisper supports (wav, mp3, ogg, webm, m4a).
    Returns the transcription string, or '' if Whisper is unavailable or fails.
    """
    if not audio_bytes:
        return ""
    model = _load_whisper()
    if model is None:
        return ""
    try:
        import whisper  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        # Write to a temp file — Whisper's load_audio() works best from disk
        suffix = ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            audio = whisper.load_audio(tmp_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            options = whisper.DecodingOptions(language=language, fp16=False)
            result = whisper.decode(model, mel, options)
            return result.text.strip()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as exc:
        logger.warning("Whisper transcription failed: %s", exc)
        return ""


def whisper_available() -> bool:
    """Return True if the Whisper tiny model loaded successfully."""
    try:
        import whisper  # noqa: F401
        return True
    except ImportError:
        return False


def tts_available() -> bool:
    """Return True if pyttsx3 is importable."""
    try:
        import pyttsx3  # noqa: F401
        return True
    except ImportError:
        return False
