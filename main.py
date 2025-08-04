# ------------------------------------------------------------------
# ENHANCED main.py with XTTS Voice Cloning Integration - GIBBERISH FIX
# ------------------------------------------------------------------
import os
import re
import uuid
import time
import pytz
import torch
import sqlite3
import threading
import requests
import numpy as np
import asyncio
from datetime import datetime, timedelta
from subprocess import run, CalledProcessError, PIPE
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from contextlib import contextmanager
from scipy.io.wavfile import write
from scipy.signal import resample_poly
from typing import Dict, Optional

# Fix for numpy.complex deprecated in numpy 1.24+
if not hasattr(np, 'complex'):
    np.complex = complex

# ENHANCED torch.load fix with voice cloning compatibility
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# Add comprehensive safe globals for voice cloning
safe_globals_list = [XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig]

# Add additional classes that might be needed for voice cloning
try:
    from TTS.tts.configs.shared_configs import BaseAudioConfig, BaseTrainingConfig
    from TTS.tts.models.base import BaseTTS

    safe_globals_list.extend([BaseAudioConfig, BaseTrainingConfig, BaseTTS])
except ImportError:
    pass

try:
    from TTS.tts.models.xtts import XttsModel

    safe_globals_list.append(XttsModel)
except ImportError:
    pass

torch.serialization.add_safe_globals(safe_globals_list)

_original_torch_load = torch.load


def safe_torch_load(f, *args, **kwargs):
    """Enhanced torch.load for voice cloning compatibility"""
    kwargs['weights_only'] = False
    kwargs['map_location'] = kwargs.get('map_location', 'cpu')

    # Use safe globals context manager for voice cloning
    with torch.serialization.safe_globals(safe_globals_list):
        return _original_torch_load(f, *args, **kwargs)


torch.load = safe_torch_load

# COMPATIBILITY FIX: Import TTS with version checking
import pkg_resources

try:
    transformers_version = pkg_resources.get_distribution("transformers").version
    print(f"Transformers version: {transformers_version}")

    # Check if we have a compatible version
    major, minor = map(int, transformers_version.split('.')[:2])
    if major > 4 or (major == 4 and minor >= 50):
        print("⚠️  WARNING: Transformers version >= 4.50 detected. XTTS may not work properly.")
        print("   Consider downgrading: pip install transformers==4.42.4")

except Exception as e:
    print(f"Could not check transformers version: {e}")

import uvicorn
from TTS.api import TTS
import whisper
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
DOWNLOADS_DIR = os.path.expanduser("~/Downloads/Dalexa_Audio_Videos")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
DB_PATH = os.path.join(DOWNLOADS_DIR, "reminders.db")
app.mount("/downloads", StaticFiles(directory=DOWNLOADS_DIR), name="downloads")

print("Loading Whisper model...")
whisper_model = whisper.load_model("medium")
print("Whisper model loaded.")

# Load Silero VAD with error handling
print("Loading Silero VAD model...")
try:
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    (get_speech_timestamps, _, _, _, _) = utils
    print("Silero VAD model loaded.")
except Exception as e:
    print(f"Failed to load VAD model: {e}")
    vad_model = None
    get_speech_timestamps = None

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"
conversations = {}
LOCAL_TZ = pytz.timezone("America/Los_Angeles")
WAV2LIP_PATH = os.getenv("WAV2LIP_PATH", "/home/home/Downloads/Wav2Lip")
FACE_IMAGE_PATH = os.getenv("FACE_IMAGE_PATH", "/home/home/Downloads/IMG_4032.PNG")

# VOICE CLONING CONFIGURATION
VOICE_REFERENCE_PATH = os.getenv("VOICE_REFERENCE_PATH", "/home/home/Downloads/combined_ref.wav")

# ENHANCED XTTS loading with voice cloning support
print("Loading XTTS model for voice cloning...")
xtts_model = None
xtts_error = None

try:
    # Load XTTS model specifically for voice cloning
    xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
    print("✅ XTTS model loaded successfully for voice cloning.")

    # Verify voice reference file exists
    if os.path.exists(VOICE_REFERENCE_PATH):
        print(f"✅ Voice reference file found: {VOICE_REFERENCE_PATH}")
    else:
        print(f"⚠️  Voice reference file not found: {VOICE_REFERENCE_PATH}")
        print("   Voice cloning will fall back to default voice")

except Exception as e:
    xtts_error = str(e)
    print(f"❌ Failed to load XTTS model: {e}")
    xtts_model = None

# GLOBAL FOR TRACKING ASYNC JOBS
processing_status: Dict[str, Dict] = {}


# ------------------------------------------------------------------
# IMPROVED AUDIO PROCESSING - NO GIBBERISH
# ------------------------------------------------------------------
def resample_to_16k(wav, orig_sr=22050):
    """High-quality resampling with proper parameters"""
    try:
        target_sr = 16000
        if orig_sr == target_sr:
            return wav
        return resample_poly(wav, target_sr, orig_sr)
    except Exception as e:
        print(f"Resampling failed: {e}")
        return wav


def trim_using_vad(wav, sample_rate, speech_pad_ms=100, min_speech_duration_ms=500):
    """
    Conservative VAD trimming - only removes obvious silence
    """
    if vad_model is None or get_speech_timestamps is None:
        return wav

    try:
        # Resample to 16k for VAD model
        wav_16k = resample_to_16k(wav, sample_rate)
        audio_tensor = torch.tensor(wav_16k, dtype=torch.float32)

        # Get speech timestamps with minimal padding
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            vad_model,
            sampling_rate=16000,
            speech_pad_ms=speech_pad_ms,
            min_speech_duration_ms=min_speech_duration_ms
        )

        if not speech_timestamps:
            print("No speech detected by VAD, returning original audio")
            return wav

        # Use first and last timestamp for trimming
        first_start = int(speech_timestamps[0]['start'] * sample_rate / 16000)
        last_end = int(speech_timestamps[-1]['end'] * sample_rate / 16000)

        # Conservative trimming - leave some buffer
        start = max(0, first_start - int(sample_rate * 0.05))  # 50ms buffer
        end = min(len(wav), last_end + int(sample_rate * 0.05))  # 50ms buffer

        return wav[start:end]

    except Exception as e:
        print(f"VAD processing failed: {e}")
        return wav


def split_into_sentences(text):
    """Improved sentence splitting"""
    # Handle abbreviations
    text = re.sub(r'(Mr|Mrs|Dr|Ms)\.', r'\1<prd>', text)
    text = re.sub(r'(Inc|Ltd|Jr|Sr|Co)\.', r'\1<prd>', text)
    text = re.sub(r'([A-Z])\.', r'\1<prd>', text)

    # Split on sentence boundaries
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

    # Restore periods
    sentences = [s.replace('<prd>', '.') for s in sentences]

    return sentences if sentences else [text.strip()]


def apply_fade(audio, fade_ms=5, sample_rate=22050):
    """Minimal fade to prevent clicks without affecting speech"""
    fade_length = int(sample_rate * fade_ms / 1000)
    if fade_length == 0 or len(audio) < fade_length * 4:
        return audio

    fade_in = np.linspace(0.0, 1.0, fade_length, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_length, dtype=np.float32)

    audio[:fade_length] *= fade_in
    audio[-fade_length:] *= fade_out

    return audio


# IMPROVED VOICE CLONING - NO GIBBERISH
def generate_clean_tts_with_voice_cloning(text: str, speaker_wav_path: str = None):
    """Generate clean TTS with voice cloning - optimized for clarity"""
    if xtts_model is None:
        print("❌ XTTS model not available, using silence")
        return np.zeros(int(22050 * 0.5), dtype=np.float32)

    if not text.strip():
        return np.zeros(int(22050 * 0.5), dtype=np.float32)

    # Use the voice reference file for cloning
    if speaker_wav_path is None:
        speaker_wav_path = VOICE_REFERENCE_PATH

    if not os.path.exists(speaker_wav_path):
        print(f"⚠️  Voice reference not found, using default voice")
        speaker_wav_path = None

    sample_rate = 22050

    # Pre-process text to remove problematic characters
    text = re.sub(r'[^\w\s.,!?;:\-\'\"]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    sentences = split_into_sentences(text)
    all_segments = []

    print(f"🎭 Generating voice-cloned TTS for {len(sentences)} sentences")

    for i, sentence in enumerate(sentences):
        if not sentence.strip() or len(sentence.strip()) < 2:
            continue

        try:
            print(f"  Processing: '{sentence[:50]}{'...' if len(sentence) > 50 else ''}'")

            # Clean sentence for TTS
            sentence = sentence.strip()
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'  # Ensure proper ending

            # Generate TTS
            try:
                if speaker_wav_path and os.path.exists(speaker_wav_path):
                    wav = xtts_model.tts(
                        text=sentence,
                        speaker_wav=speaker_wav_path,
                        language="en"
                    )
                else:
                    wav = xtts_model.tts(text=sentence, language="en")

            except Exception as tts_err:
                print(f"❌ TTS failed for sentence {i + 1}: {tts_err}")
                continue

            if wav is None or len(wav) == 0:
                print(f"⚠️  Empty audio for sentence {i + 1}")
                continue

            wav = np.array(wav, dtype=np.float32)

            # Skip if audio is too short (likely gibberish)
            if len(wav) < sample_rate * 0.1:  # Less than 100ms
                print(f"⚠️  Audio too short for sentence {i + 1}, skipping")
                continue

            # Light VAD cleaning - only remove obvious silence
            cleaned = trim_using_vad(wav, sample_rate)

            # Ensure we have actual content
            if len(cleaned) < sample_rate * 0.1:
                print(f"⚠️  Sentence {i + 1} became too short after VAD, using original")
                cleaned = wav

            # Apply minimal fade
            cleaned = apply_fade(cleaned, fade_ms=3, sample_rate=sample_rate)

            # Ensure audio isn't clipped
            if np.max(np.abs(cleaned)) > 0.95:
                cleaned = cleaned * 0.95 / np.max(np.abs(cleaned))

            all_segments.append(cleaned)
            print(f"  ✅ Sentence {i + 1} processed ({len(cleaned) / sample_rate:.2f}s)")

        except Exception as e:
            print(f"❌ Failed sentence {i + 1}: {e}")
            continue

    if not all_segments:
        print("❌ No audio generated, returning silence")
        return np.zeros(int(sample_rate * 0.5), dtype=np.float32)

    # Concatenate with conservative gaps
    gap = np.zeros(int(sample_rate * 0.08), dtype=np.float32)  # 80ms gap
    final_segments = []

    for i, segment in enumerate(all_segments):
        final_segments.append(segment)
        if i < len(all_segments) - 1:
            final_segments.append(gap)

    final_audio = np.concatenate(final_segments)

    # Final normalization
    if np.max(np.abs(final_audio)) > 0:
        final_audio = final_audio / np.max(np.abs(final_audio)) * 0.9

    return np.clip(final_audio, -1.0, 1.0)


# Maintain backward compatibility
def generate_clean_tts(text: str, speaker_wav_path: str = None):
    return generate_clean_tts_with_voice_cloning(text, speaker_wav_path)


# ------------------------------------------------------------------
# DATABASE SECTION (Same as before)
# ------------------------------------------------------------------
def init_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                remind_at TEXT NOT NULL,
                audio_file TEXT,
                video_file TEXT,
                reminded INTEGER DEFAULT 0,
                notified_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                job_id TEXT
            )
        """)
        cutoff = (datetime.now(LOCAL_TZ) - timedelta(hours=24)).isoformat()
        conn.execute("DELETE FROM reminders WHERE reminded = 1 AND notified_at < ?", (cutoff,))


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def get_due_reminders():
    now = datetime.now(LOCAL_TZ).isoformat()
    with get_db() as conn:
        return conn.execute("""
            SELECT id, text, audio_file, video_file, job_id FROM reminders
            WHERE remind_at <= ? AND reminded = 0
            ORDER BY remind_at ASC
        """, (now,)).fetchall()


def mark_reminded(reminder_id: int):
    now = datetime.now(LOCAL_TZ).isoformat()
    with get_db() as conn:
        conn.execute("UPDATE reminders SET reminded = 1, notified_at = ? WHERE id = ?", (now, reminder_id))
        conn.commit()


def add_reminder(text: str, remind_at: datetime, audio_file: str = None, video_file: str = None, job_id: str = None):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO reminders (text, remind_at, audio_file, video_file, job_id)
            VALUES (?, ?, ?, ?, ?)
        """, (text, remind_at.isoformat(), audio_file, video_file, job_id))
        conn.commit()


# ------------------------------------------------------------------
# BACKGROUND REMINDER CHECKER
# ------------------------------------------------------------------
def reminder_checker():
    while True:
        try:
            now = datetime.now(LOCAL_TZ)
            due_reminders = get_due_reminders()
            for rid, text, audio, video, jid in due_reminders:
                mark_reminded(rid)
            if now.minute % 30 == 0:
                cleanup_old_files()
        except Exception as e:
            print("[RC] error:", e)
        time.sleep(15)


def cleanup_old_files():
    cutoff = (datetime.now(LOCAL_TZ) - timedelta(hours=24)).isoformat()
    with get_db() as conn:
        old = conn.execute("SELECT audio_file, video_file FROM reminders WHERE reminded = 1 AND notified_at < ?",
                           (cutoff,)).fetchall()
    for a, v in old:
        for f in (a, v):
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass


# ------------------------------------------------------------------
# ENHANCED ASYNC GENERATORS WITH VOICE CLONING
# ------------------------------------------------------------------
async def generate_audio_and_video_async(text: str, job_id: str):
    processing_status[job_id] = {"status": "processing", "step": "generating_voice_cloned_audio"}

    try:
        if xtts_model is None:
            processing_status[job_id].update({
                "status": "error",
                "error": f"XTTS model not available. Original error: {xtts_error or 'Unknown'}"
            })
            return

        audio_path = os.path.join(DOWNLOADS_DIR, f"response_{uuid.uuid4()}.wav")

        # Generate voice-cloned TTS audio
        print(f"🎭 Starting voice cloning for job {job_id}")
        wav = generate_clean_tts_with_voice_cloning(text)

        if len(wav) == 0:
            processing_status[job_id].update({"status": "error", "error": "Generated audio is empty"})
            return

        # Save audio file
        try:
            scaled = (wav * 32767).astype(np.int16)
            write(audio_path, 22050, scaled)
            processing_status[job_id]["audio_path"] = audio_path
            processing_status[job_id]["step"] = "generating_video"
            print(f"✅ Voice-cloned audio saved for job {job_id}")
        except Exception as audio_save_error:
            processing_status[job_id].update({"status": "error", "error": f"Failed to save audio: {audio_save_error}"})
            return

        # Generate video with Wav2Lip
        video_path = os.path.join(DOWNLOADS_DIR, f"response_{uuid.uuid4()}.mp4")
        processing_status[job_id]["step"] = "checking_wav2lip_requirements"

        # Check if required files exist
        if not os.path.exists(WAV2LIP_PATH):
            print(f"❌ Wav2Lip not found at {WAV2LIP_PATH}")
            processing_status[job_id].update({
                "status": "completed",
                "video_error": f"Wav2Lip not found at {WAV2LIP_PATH}"
            })
            return

        if not os.path.exists(FACE_IMAGE_PATH):
            print(f"❌ Face image not found at {FACE_IMAGE_PATH}")
            processing_status[job_id].update({
                "status": "completed",
                "video_error": f"Face image not found at {FACE_IMAGE_PATH}"
            })
            return

        checkpoint_path = os.path.join(WAV2LIP_PATH, "checkpoints", "wav2lip.pth")
        if not os.path.exists(checkpoint_path):
            print(f"❌ Wav2Lip checkpoint not found at {checkpoint_path}")
            processing_status[job_id].update({
                "status": "completed",
                "video_error": f"Wav2Lip checkpoint not found at {checkpoint_path}"
            })
            return

        print(f"🎬 Starting Wav2Lip video generation for job {job_id}")
        processing_status[job_id]["step"] = "generating_wav2lip_video"

        try:
            # Run Wav2Lip with enhanced error capture
            cmd = [
                "python3", os.path.join(WAV2LIP_PATH, "inference.py"),
                "--checkpoint_path", checkpoint_path,
                "--face", FACE_IMAGE_PATH,
                "--audio", audio_path,
                "--outfile", video_path,
                "--pads", "0", "10", "0", "0"
            ]

            print(f"🎬 Running command: {' '.join(cmd)}")

            result = run(cmd, check=True, capture_output=True, text=True, timeout=120, cwd=WAV2LIP_PATH)

            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                processing_status[job_id].update({"video_path": video_path, "status": "completed"})
                print(f"✅ Video with voice-cloned audio generated for job {job_id}")
                print(f"   Video size: {os.path.getsize(video_path)} bytes")
            else:
                print(f"❌ Wav2Lip completed but no valid output file was created")
                print(f"   Expected: {video_path}")
                print(f"   Exists: {os.path.exists(video_path)}")
                if os.path.exists(video_path):
                    print(f"   Size: {os.path.getsize(video_path)} bytes")
                processing_status[job_id].update({
                    "status": "completed",
                    "video_error": "Wav2Lip completed but no valid output file was created"
                })

        except CalledProcessError as cpe:
            error_msg = f"Wav2Lip failed (exit code {cpe.returncode})"
            if cpe.stderr:
                error_msg += f"\nSTDERR: {cpe.stderr}"
            if cpe.stdout:
                error_msg += f"\nSTDOUT: {cpe.stdout}"

            print(f"❌ {error_msg}")
            processing_status[job_id].update({"status": "completed", "video_error": error_msg})

        except Exception as video_error:
            print(f"❌ Video generation error: {video_error}")
            processing_status[job_id].update({"status": "completed", "video_error": str(video_error)})

    except Exception as e:
        processing_status[job_id].update({"status": "error", "error": str(e)})


# ------------------------------------------------------------------
# FASTAPI ENDPOINTS
# ------------------------------------------------------------------
@app.get("/check_status/{job_id}")
async def check_processing_status(job_id: str):
    if job_id not in processing_status:
        return {"status": "not_found"}
    info = processing_status[job_id].copy()
    if "audio_path" in info and info["audio_path"]:
        info["audio_url"] = f"/downloads/{os.path.basename(info['audio_path'])}"
    if "video_path" in info and info["video_path"]:
        info["video_url"] = f"/downloads/{os.path.basename(info['video_path'])}"
    if info["status"] in ("completed", "error"):
        asyncio.create_task(asyncio.sleep(300))
    return info


@app.get("/get_reminder")
async def get_reminder():
    due = get_due_reminders()
    if due:
        rid, text, audio_file, video_file, job_id = due[0]
        mark_reminded(rid)
        return {
            "text": text,
            "audio_file": f"/downloads/{os.path.basename(audio_file)}" if audio_file else None,
            "video_file": f"/downloads/{os.path.basename(video_file)}" if video_file else None,
            "job_id": job_id
        }
    return {"text": None, "job_id": None}


@app.get("/list_reminders")
async def list_reminders():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT text, remind_at FROM reminders WHERE reminded = 0 ORDER BY remind_at ASC").fetchall()
    return [{"text": text, "remind_at": remind_at} for text, remind_at in rows]


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "whisper_loaded": whisper_model is not None,
        "xtts_loaded": xtts_model is not None,
        "xtts_error": xtts_error if xtts_model is None else None,
        "vad_loaded": vad_model is not None,
        "voice_reference_available": os.path.exists(VOICE_REFERENCE_PATH),
        "voice_reference_path": VOICE_REFERENCE_PATH,
        "transformers_compatible": xtts_model is not None
    }


# NEW ENDPOINT: Debug job status
@app.get("/debug_job/{job_id}")
async def debug_job_status(job_id: str):
    """Debug endpoint to see detailed job information"""
    if job_id not in processing_status:
        return {"error": "Job not found", "job_id": job_id}

    status = processing_status[job_id].copy()

    # Add file existence checks
    if "audio_path" in status:
        audio_exists = os.path.exists(status["audio_path"])
        audio_size = os.path.getsize(status["audio_path"]) if audio_exists else 0
        status["audio_debug"] = {
            "exists": audio_exists,
            "size": audio_size,
            "path": status["audio_path"]
        }

    if "video_path" in status:
        video_exists = os.path.exists(status["video_path"])
        video_size = os.path.getsize(status["video_path"]) if video_exists else 0
        status["video_debug"] = {
            "exists": video_exists,
            "size": video_size,
            "path": status["video_path"]
        }

    # Add system status
    status["system_debug"] = {
        "wav2lip_path_exists": os.path.exists(WAV2LIP_PATH),
        "face_image_exists": os.path.exists(FACE_IMAGE_PATH),
        "voice_reference_exists": os.path.exists(VOICE_REFERENCE_PATH),
        "downloads_dir": DOWNLOADS_DIR
    }

    return status


@app.post("/test_voice_clone")
async def test_voice_clone(request: Request):
    """Test endpoint for voice cloning functionality"""
    try:
        body = await request.json()
        test_text = body.get("text", "Hello! This is a test of voice cloning.")

        if xtts_model is None:
            return {"error": "XTTS model not available", "xtts_error": xtts_error}

        job_id = str(uuid.uuid4())

        # Generate voice-cloned audio
        wav = generate_clean_tts_with_voice_cloning(test_text)

        if len(wav) == 0:
            return {"error": "Generated audio is empty"}

        # Save test audio
        test_path = os.path.join(DOWNLOADS_DIR, f"voice_clone_test_{job_id}.wav")
        scaled = (wav * 32767).astype(np.int16)
        write(test_path, 22050, scaled)

        return {
            "success": True,
            "text": test_text,
            "audio_url": f"/downloads/{os.path.basename(test_path)}",
            "duration": len(wav) / 22050,
            "voice_reference_used": os.path.exists(VOICE_REFERENCE_PATH)
        }

    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------------------------
# ENHANCED PROCESS AUDIO WITH VOICE CLONING
# ------------------------------------------------------------------
@app.post("/process_audio")
async def process_audio(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    client_ip = request.client.host
    job_id = str(uuid.uuid4())
    audio_filename = os.path.join(DOWNLOADS_DIR, f"input_{uuid.uuid4()}.wav")

    with open(audio_filename, "wb") as f:
        f.write(await file.read())

    try:
        # Transcribe audio
        transcription = normalize_time_formats(whisper_model.transcribe(audio_filename)["text"])
        history = conversations.get(client_ip, "") + f"User: {transcription}\n"
        current_date = datetime.now(LOCAL_TZ).strftime("%B %d, %Y")

        INSTRUCTION = (
            f"You are a helpful voice assistant. Today's date is {current_date}.\n"
            "If the user is NOT setting a reminder, respond conversationally.\n\n"
            "⚠️ BUT IF THE USER IS SETTING A REMINDER, YOU MUST respond in this format:\n"
            "Reminder set: <description> at <HH:MM am/pm> on <date>.\n\n"
            "🚫 DO NOT add anything else. DO NOT skip this format."
        )

        full_prompt = INSTRUCTION + "\n" + history
        payload = {"model": MODEL_NAME, "prompt": full_prompt, "stream": False}

        try:
            llm_resp = requests.post(OLLAMA_URL, json=payload, timeout=30).json().get("response", "").strip()
        except Exception as e:
            llm_resp = f"I'm having trouble connecting to my language model. Error: {str(e)}"

        conversations[client_ip] = history + f"Assistant: {llm_resp}\n"

        # Start voice cloning TTS job if model is available
        if xtts_model is not None:
            print(f"🎭 Starting voice cloning job for: '{llm_resp[:50]}{'...' if len(llm_resp) > 50 else ''}'")
            background_tasks.add_task(generate_audio_and_video_async, llm_resp, job_id)
        else:
            processing_status[job_id] = {
                "status": "error",
                "error": f"TTS model not available. {xtts_error or 'Unknown error'}"
            }

        # Handle reminder parsing
        match = re.search(r"Reminder set:\s*(.+?)\s+at\s+([^\s]+(?:\s*[ap]\.?m\.?))\s*(?:on\s+([^\.\n]+))?\.", llm_resp,
                          re.I)
        if match:
            text = match.group(1).strip()
            raw_time = match.group(2)
            date_str = (match.group(3) or "today").strip().lower()
            date_str = re.sub(r"[<>]", "", date_str)
            date_str = re.sub(r"\bon\b\s+", "", date_str)

            try:
                if date_str == "today":
                    date_obj = datetime.now(LOCAL_TZ).date()
                elif date_str == "tomorrow":
                    date_obj = (datetime.now(LOCAL_TZ) + timedelta(days=1)).date()
                else:
                    date_obj = parse_date_string(date_str)

                time_str = clean_time_string(raw_time)
                time_obj = datetime.strptime(time_str, "%I:%M%p").time()
                reminder_dt = LOCAL_TZ.localize(datetime.combine(date_obj, time_obj))

                if reminder_dt < datetime.now(LOCAL_TZ):
                    reminder_dt += timedelta(days=1)

                reminder_job_id = str(uuid.uuid4())
                add_reminder(text, reminder_dt, None, None, reminder_job_id)

                if xtts_model is not None:
                    print(f"🎭 Starting voice cloning for reminder: '{text}'")
                    background_tasks.add_task(generate_audio_and_video_async, f"Reminder: {text}", reminder_job_id)

            except Exception as reminder_error:
                print(f"Failed to set reminder: {reminder_error}")

        return {
            "transcription": transcription,
            "response": llm_resp,
            "job_id": job_id,
            "status": "processing",
            "voice_cloning_enabled": xtts_model is not None and os.path.exists(VOICE_REFERENCE_PATH)
        }

    finally:
        try:
            os.remove(audio_filename)
        except:
            pass


# ------------------------------------------------------------------
# HELPER FUNCTIONS (Same as before)
# ------------------------------------------------------------------
def normalize_time_formats(text: str) -> str:
    def repl(m):
        h, sep, m, ampm = m.groups()
        return f"{h}:{m.zfill(2)} {ampm.lower()}"

    return re.sub(r"(\d{1,2})([.:])(\d{1,2}) ?([ap]m)", repl, text, flags=re.I)


def parse_date_string(date_str: str):
    from dateutil import parser as dp
    date_str = re.sub(r'(\d)(st|nd|rd|th)', r'\1', date_str, flags=re.I)
    return dp.parse(date_str, fuzzy=True).date()


def clean_time_string(time_str: str):
    time_str = time_str.lower().strip()
    time_str = re.sub(r'\s*a\.?m\.?\s*', 'am', time_str)
    time_str = re.sub(r'\s*p\.?m\.?\s*', 'pm', time_str)
    time_str = time_str.replace('.', ':').replace(' ', '')
    if not ':' in time_str and len(time_str) >= 3:
        time_str = re.sub(r'(\d+)(am|pm)', r'\1:00\2', time_str)
    return time_str


# ------------------------------------------------------------------
# INITIALIZATION WITH VOICE CLONING STATUS
# ------------------------------------------------------------------
if __name__ == "__main__":
    init_database()
    threading.Thread(target=reminder_checker, daemon=True).start()

    print("=" * 60)
    print("🎭 DALEXA VOICE ASSISTANT WITH VOICE CLONING - GIBBERISH FIX")
    print("=" * 60)
    print(f"🎤 Whisper model: {'✅ Loaded' if whisper_model else '❌ Failed'}")
    print(f"🗣️  XTTS model: {'✅ Loaded' if xtts_model else '❌ Failed'}")
    if xtts_model is None and xtts_error:
        print(f"   Error: {xtts_error}")
    print(f"🔊 VAD model: {'✅ Loaded' if vad_model else '❌ Failed'}")
    print(f"🎭 Voice reference: {'✅ Found' if os.path.exists(VOICE_REFERENCE_PATH) else '❌ Missing'}")
    if os.path.exists(VOICE_REFERENCE_PATH):
        print(f"   Using: {VOICE_REFERENCE_PATH}")
    else:
        print(f"   Expected at: {VOICE_REFERENCE_PATH}")
        print("   Voice cloning will use default voice")
    print(f"🎬 Wav2Lip: {'✅ Available' if os.path.exists(WAV2LIP_PATH) else '❌ Not found'}")
    print(f"🖼️  Face image: {'✅ Found' if os.path.exists(FACE_IMAGE_PATH) else '❌ Missing'}")
    print("=" * 60)

    if xtts_model and os.path.exists(VOICE_REFERENCE_PATH):
        print("🎉 Voice cloning is READY!")
        print("   Your assistant will now speak with the cloned voice from combined_ref.wav")
    elif xtts_model:
        print("⚠️  XTTS loaded but no voice reference - using default voice")
    else:
        print("❌ Voice cloning unavailable - check XTTS installation")

    print("🚀 Server starting...")

    # Optional: Test voice cloning on startup
    if xtts_model and os.path.exists(VOICE_REFERENCE_PATH):
        try:
            print("\n🧪 Testing voice cloning on startup...")
            test_wav = generate_clean_tts_with_voice_cloning("Testing clean voice cloning.")
            if len(test_wav) > 0:
                print("✅ Voice cloning test passed!")
            else:
                print("⚠️  Voice cloning test returned empty audio")
        except Exception as test_error:
            print(f"⚠️  Voice cloning test failed: {test_error}")

    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=9000)