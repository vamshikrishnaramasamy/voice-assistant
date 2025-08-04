#!/usr/bin/env python3
"""
Standalone Reminder Manager with Enhanced Video/Audio Playback
Can be run independently to handle reminders from the database
"""

import asyncio
import sqlite3
import os
import time
import pytz
import torch
import numpy as np
import uuid
import platform
from datetime import datetime, timedelta
from subprocess import run, CalledProcessError, TimeoutExpired, PIPE
from contextlib import contextmanager
from scipy.io.wavfile import write
from scipy.signal import resample_poly
import argparse
import sys

# Fix for numpy.complex deprecated in numpy 1.24+
if not hasattr(np, 'complex'):
    np.complex = complex

# Enhanced torch.load fix with voice cloning compatibility
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

# Now import TTS
from TTS.api import TTS

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

# Configuration - Updated paths to match main.py
DOWNLOADS_DIR = os.path.expanduser("~/Downloads/Dalexa_Audio_Videos")
DB_PATH = os.path.join(DOWNLOADS_DIR, "reminders.db")
LOCAL_TZ = pytz.timezone("America/Los_Angeles")
CHECK_INTERVAL = 15  # seconds
CLEANUP_INTERVAL = 1800  # 30 minutes

# Voice cloning paths - match main.py
VOICE_REFERENCE_PATH = os.getenv("VOICE_REFERENCE_PATH", "/home/home/Downloads/combined_ref.wav")
WAV2LIP_PATH = os.getenv("WAV2LIP_PATH", "/home/home/Downloads/Wav2Lip")
FACE_IMAGE_PATH = os.getenv("FACE_IMAGE_PATH", "/home/home/Downloads/IMG_4032.PNG")

# Detect system for appropriate media players
SYSTEM = platform.system().lower()

# Initialize XTTS model with same logic as main.py
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


# VAD processing functions from main.py
def resample_to_16k(wav, orig_sr=22050):
    try:
        target_sr = 16000
        up = target_sr
        down = orig_sr
        return resample_poly(wav, up, down)
    except Exception as e:
        print(f"Resampling failed: {e}")
        if orig_sr > 16000:
            decimation_factor = int(orig_sr // 16000)
            return wav[::decimation_factor]
        return wav


def trim_using_vad(wav, sample_rate, speech_pad_ms=200, min_speech_duration_ms=250):
    """Trims silence from audio using Silero VAD, adding padding and filtering short noises."""
    if vad_model is None or get_speech_timestamps is None:
        print("VAD model not available, returning original audio")
        return wav

    try:
        # Resample to 16k for VAD model
        wav_16k = resample_to_16k(wav, sample_rate)
        audio_tensor = torch.tensor(wav_16k, dtype=torch.float32)

        # Get speech timestamps with padding and filtering
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            vad_model,
            sampling_rate=16000,
            speech_pad_ms=speech_pad_ms,
            min_speech_duration_ms=min_speech_duration_ms
        )

        if not speech_timestamps:
            print("No speech detected by VAD, returning silence")
            return np.zeros(1, dtype=np.float32)

        # Concatenate voiced segments
        voiced_segments = []
        for seg in speech_timestamps:
            start = int(seg['start'] * sample_rate / 16000)
            end = int(seg['end'] * sample_rate / 16000)

            start = max(0, min(start, len(wav)))
            end = max(start, min(end, len(wav)))

            if end > start:
                voiced_segments.append(wav[start:end])

        if voiced_segments:
            return np.concatenate(voiced_segments)
        else:
            print("VAD processing resulted in empty audio, returning silence")
            return np.zeros(1, dtype=np.float32)

    except Exception as e:
        print(f"VAD processing failed: {e}")
        return wav


def split_into_sentences(text):
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    return sentences if sentences else [text.strip()]


def apply_fade(audio, fade_ms, sample_rate):
    """Applies a fade-in and fade-out to an audio segment to avoid clicks."""
    fade_length = int(sample_rate * fade_ms / 1000)
    if fade_length == 0:
        return audio

    if fade_length > len(audio) // 2:
        fade_length = len(audio) // 2

    if fade_length > 0:
        fade_in = np.linspace(0.0, 1.0, fade_length, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_length, dtype=np.float32)

        audio[:fade_length] *= fade_in
        audio[-fade_length:] *= fade_out

    return audio


def generate_clean_tts_with_voice_cloning(text: str, speaker_wav_path: str = None):
    """Generate TTS using voice cloning with the reference audio - same as main.py"""
    if xtts_model is None:
        print("❌ XTTS model not available, generating silence")
        return np.zeros(int(22050 * 1.0), dtype=np.float32)

    if not text.strip():
        print("Empty text provided, returning silence")
        return np.zeros(int(22050 * 0.5), dtype=np.float32)

    # Use the voice reference file for cloning
    if speaker_wav_path is None:
        speaker_wav_path = VOICE_REFERENCE_PATH

    # Check if reference file exists
    if not os.path.exists(speaker_wav_path):
        print(f"⚠️  Voice reference file not found: {speaker_wav_path}")
        print("   Attempting to generate without speaker reference...")
        speaker_wav_path = None

    sample_rate = 22050
    sentences = split_into_sentences(text)
    all_cleaned = []

    print(f"🎭 Generating voice-cloned TTS for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    if speaker_wav_path:
        print(f"🎤 Using voice reference: {os.path.basename(speaker_wav_path)}")

    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue

        try:
            print(f"  Processing sentence {i + 1}/{len(sentences)}: {sentence[:30]}...")

            # Generate TTS with voice cloning
            try:
                if speaker_wav_path and os.path.exists(speaker_wav_path):
                    # Voice cloning with reference audio
                    wav = xtts_model.tts(
                        text=sentence,
                        speaker_wav=speaker_wav_path,
                        language="en"
                    )
                    print(f"  ✅ Voice cloned for sentence {i + 1}")
                else:
                    # Fallback without speaker reference
                    wav = xtts_model.tts(text=sentence, language="en")
                    print(f"  ⚠️  Generated without voice cloning for sentence {i + 1}")

            except AttributeError as attr_err:
                if "'GPT2InferenceModel' object has no attribute 'generate'" in str(attr_err):
                    print(f"⚠️  XTTS compatibility issue detected: {attr_err}")
                    print("   This is likely due to transformers version >= 4.50")
                    print("   Consider downgrading: pip install transformers==4.42.4")
                    break
                else:
                    raise attr_err
            except Exception as tts_err:
                print(f"❌ TTS generation failed for sentence {i + 1}: {tts_err}")
                continue

            if wav is None or len(wav) == 0:
                print(f"⚠️  Empty audio generated for sentence {i + 1}")
                continue

            wav = np.array(wav, dtype=np.float32)

            # Apply VAD cleaning if available
            cleaned = trim_using_vad(wav, sample_rate)

            # Apply fade-in and fade-out to prevent clicks
            if len(cleaned) > 0:
                cleaned = apply_fade(cleaned, fade_ms=10, sample_rate=sample_rate)

            if len(cleaned) > 0 and np.max(np.abs(cleaned)) > 0:
                all_cleaned.append(cleaned)
                print(f"  ✅ Successfully processed sentence {i + 1}")
            else:
                print(f"  ⚠️  Sentence {i + 1} resulted in silence after VAD")

        except Exception as e:
            print(f"❌ Failed to process sentence {i + 1}: {e}")
            continue

    if not all_cleaned:
        print("❌ No sentences were successfully processed, returning silence")
        return np.zeros(int(sample_rate * 1.0), dtype=np.float32)

    # Concatenate with slightly larger gaps for better pacing
    gap = np.zeros(int(sample_rate * 0.15), dtype=np.float32)  # 150ms gap
    final_segments = []
    for i, segment in enumerate(all_cleaned):
        final_segments.append(segment)
        if i < len(all_cleaned) - 1:
            final_segments.append(gap)

    final_audio = np.concatenate(final_segments)

    # Perform a single, final normalization for consistent volume
    if np.max(np.abs(final_audio)) > 0:
        final_audio = final_audio / np.max(np.abs(final_audio)) * 0.95

    final_clipped = np.clip(final_audio, -1.0, 1.0)

    print(f"🎉 Voice cloning complete! Generated {len(final_clipped) / sample_rate:.2f} seconds of audio")
    return final_clipped


@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    """Initialize the SQLite database for reminders"""
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
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
        # Clean up old reminded reminders (older than 24 hours)
        cutoff = (datetime.now(LOCAL_TZ) - timedelta(hours=24)).isoformat()
        conn.execute("DELETE FROM reminders WHERE reminded = 1 AND notified_at < ?", (cutoff,))
        conn.commit()


def get_due_reminders():
    """Get reminders that are due to be triggered"""
    now = datetime.now(LOCAL_TZ).isoformat()
    with get_db() as conn:
        return conn.execute("""
            SELECT id, text, audio_file, video_file, job_id FROM reminders
            WHERE remind_at <= ? AND reminded = 0
            ORDER BY remind_at ASC
        """, (now,)).fetchall()


def mark_reminded(reminder_id: int):
    """Mark a reminder as having been triggered"""
    now = datetime.now(LOCAL_TZ).isoformat()
    with get_db() as conn:
        conn.execute("""
            UPDATE reminders SET reminded = 1, notified_at = ? WHERE id = ?
        """, (now, reminder_id))
        conn.commit()


def list_pending_reminders():
    """List all pending reminders"""
    with get_db() as conn:
        reminders = conn.execute("""
            SELECT id, text, remind_at FROM reminders
            WHERE reminded = 0
            ORDER BY remind_at ASC
        """).fetchall()
    return reminders


def cleanup_old_files():
    """Clean up old audio/video files from completed reminders"""
    cutoff = (datetime.now(LOCAL_TZ) - timedelta(hours=2)).isoformat()
    with get_db() as conn:
        old_files = conn.execute("""
            SELECT audio_file, video_file FROM reminders
            WHERE reminded = 1 AND notified_at < ?
        """, (cutoff,)).fetchall()

        cleaned_count = 0
        for audio_file, video_file in old_files:
            for file_path in [audio_file, video_file]:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                        print(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        print(f"Failed to remove {file_path}: {e}")

        if cleaned_count > 0:
            print(f"Cleaned up {cleaned_count} old files")


def generate_wav2lip_video(audio_path: str, reminder_text: str) -> str:
    """Generate Wav2Lip video for reminder - same logic as main.py"""
    if not os.path.exists(WAV2LIP_PATH):
        print(f"❌ Wav2Lip not found at {WAV2LIP_PATH}")
        return None

    if not os.path.exists(FACE_IMAGE_PATH):
        print(f"❌ Face image not found at {FACE_IMAGE_PATH}")
        return None

    checkpoint_path = os.path.join(WAV2LIP_PATH, "checkpoints", "wav2lip.pth")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Wav2Lip checkpoint not found at {checkpoint_path}")
        return None

    video_path = os.path.join(DOWNLOADS_DIR, f"reminder_video_{uuid.uuid4()}.mp4")

    print(f"🎬 Generating Wav2Lip video for reminder: {reminder_text[:50]}...")

    try:
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
            print(f"✅ Wav2Lip video generated: {video_path}")
            print(f"   Video size: {os.path.getsize(video_path)} bytes")
            return video_path
        else:
            print(f"❌ Wav2Lip completed but no valid output file was created")
            return None

    except CalledProcessError as cpe:
        error_msg = f"Wav2Lip failed (exit code {cpe.returncode})"
        if cpe.stderr:
            error_msg += f"\nSTDERR: {cpe.stderr}"
        if cpe.stdout:
            error_msg += f"\nSTDOUT: {cpe.stdout}"
        print(f"❌ {error_msg}")
        return None

    except Exception as video_error:
        print(f"❌ Video generation error: {video_error}")
        return None


def get_media_players():
    """Get appropriate media players for the current system"""
    if SYSTEM == "darwin":  # macOS
        return {
            'video': ['open', '-a', 'QuickTime Player'],
            'audio': ['afplay'],
            'fallback_video': ['ffplay', '-autoexit', '-loglevel', 'quiet'],
            'fallback_audio': ['ffplay', '-autoexit', '-nodisp', '-loglevel', 'quiet']
        }
    elif SYSTEM == "linux":
        return {
            'video': ['mpv', '--really-quiet'],
            'audio': ['aplay'],
            'fallback_video': ['vlc', '--intf', 'dummy', '--play-and-exit'],
            'fallback_audio': ['paplay']
        }
    else:  # Windows or others
        return {
            'video': ['ffplay', '-autoexit', '-loglevel', 'quiet'],
            'audio': ['ffplay', '-autoexit', '-nodisp', '-loglevel', 'quiet'],
            'fallback_video': ['vlc', '--intf', 'dummy', '--play-and-exit'],
            'fallback_audio': ['powershell', '-c', '(New-Object Media.SoundPlayer']
        }


def play_media_file(file_path: str, media_type: str = 'auto'):
    """Play a media file with appropriate player"""
    if not os.path.exists(file_path):
        print(f"❌ Media file not found: {file_path}")
        return False

    players = get_media_players()

    # Determine media type if auto
    if media_type == 'auto':
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            media_type = 'video'
        elif ext in ['.wav', '.mp3', '.aac', '.ogg']:
            media_type = 'audio'
        else:
            media_type = 'video'  # default

    # Try primary players first, then fallbacks
    player_lists = []
    if media_type == 'video':
        player_lists = [players.get('video', []), players.get('fallback_video', [])]
    else:
        player_lists = [players.get('audio', []), players.get('fallback_audio', [])]

    for player_list in player_lists:
        if not player_list:
            continue

        try:
            cmd = player_list + [file_path]
            print(f"🎵 Playing {media_type}: {' '.join(cmd)}")

            result = run(cmd, timeout=60, check=True, capture_output=True, text=True)
            print(f"✅ Successfully played {media_type} file")
            return True

        except FileNotFoundError:
            print(f"Player not found: {player_list[0]}")
            continue
        except (CalledProcessError, TimeoutExpired) as e:
            print(f"Player failed: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error with player: {e}")
            continue

    print(f"❌ Failed to play {media_type} file with any available player")
    return False


def play_reminder(reminder_id: int, text: str, audio_file: str = None, video_file: str = None, job_id: str = None):
    """Enhanced reminder playback with voice cloning and video generation"""
    print(f"\n{'=' * 60}")
    print(f"🔔 REMINDER TRIGGERED: {text}")
    print(f"{'=' * 60}")

    played_video = False
    played_audio = False

    # First, try to play existing video (highest priority - contains synced lip movement)
    if video_file and os.path.exists(video_file):
        print(f"🎬 Playing existing reminder video...")
        played_video = play_media_file(video_file, 'video')
        if played_video:
            print(f"✅ Video playback completed")
            return True

    # If no video or video failed, try audio
    if not played_video and audio_file and os.path.exists(audio_file):
        print(f"🎵 Playing existing reminder audio...")
        played_audio = play_media_file(audio_file, 'audio')
        if played_audio:
            print(f"✅ Audio playback completed")
            return True

    # If no existing media, generate new voice-cloned content
    if not played_video and not played_audio:
        print(f"🎭 No existing media found, generating voice-cloned reminder...")

        # Generate voice-cloned audio
        if xtts_model is not None:
            reminder_audio_path = None
            try:
                # Generate TTS audio
                wav = generate_clean_tts_with_voice_cloning(f"Reminder: {text}")

                if len(wav) > 0:
                    # Save audio file
                    reminder_audio_path = os.path.join(DOWNLOADS_DIR, f"reminder_audio_{uuid.uuid4()}.wav")
                    scaled = (wav * 32767).astype(np.int16)
                    write(reminder_audio_path, 22050, scaled)
                    print(f"✅ Voice-cloned audio generated: {reminder_audio_path}")

                    # Try to generate video with Wav2Lip
                    reminder_video_path = generate_wav2lip_video(reminder_audio_path, text)

                    # Play video if generated successfully
                    if reminder_video_path and os.path.exists(reminder_video_path):
                        print(f"🎬 Playing newly generated video...")
                        played_video = play_media_file(reminder_video_path, 'video')

                        # Clean up temporary video after playing
                        if played_video:
                            try:
                                time.sleep(2)  # Wait for playback to finish
                                os.remove(reminder_video_path)
                                print(f"🗑️  Cleaned up temporary video file")
                            except:
                                pass

                    # If video didn't work, play audio
                    if not played_video:
                        print(f"🎵 Playing newly generated audio...")
                        played_audio = play_media_file(reminder_audio_path, 'audio')

                    # Clean up temporary audio file
                    if reminder_audio_path and os.path.exists(reminder_audio_path):
                        try:
                            time.sleep(2)  # Wait for playback to finish
                            os.remove(reminder_audio_path)
                            print(f"🗑️  Cleaned up temporary audio file")
                        except:
                            pass

                    if played_video or played_audio:
                        print(f"✅ Voice-cloned reminder played successfully")
                        return True

            except Exception as e:
                print(f"❌ Voice cloning failed: {e}")
                # Clean up any temporary files
                if reminder_audio_path and os.path.exists(reminder_audio_path):
                    try:
                        os.remove(reminder_audio_path)
                    except:
                        pass

    # Final fallback: system TTS or notification
    if not played_video and not played_audio:
        print(f"⚠️  Falling back to system notification...")

        try:
            # Try system TTS first
            if SYSTEM == "darwin":  # macOS
                run(["say", "-v", "Alex", f"Reminder: {text}"], timeout=30, check=True)
                print(f"✅ System TTS played reminder")
                return True
            elif SYSTEM == "linux":
                for tts_cmd in [["espeak", "-v", "en", f"Reminder: {text}"],
                                ["spd-say", "-l", "en", f"Reminder: {text}"]]:
                    try:
                        run(tts_cmd, timeout=30, check=True)
                        print(f"✅ System TTS played reminder")
                        return True
                    except FileNotFoundError:
                        continue
        except Exception as e:
            print(f"System TTS failed: {e}")

        # Last resort: system notification
        try:
            if SYSTEM == "darwin":
                run(["osascript", "-e",
                     f'display notification "Reminder: {text}" with title "Voice Assistant Reminder"'],
                    timeout=5, check=False)
                print(f"✅ System notification displayed")
                return True
        except:
            pass

        print(f"❌ All playback methods failed - displaying text only:")
        print(f"   📝 {text}")
        return False


async def reminder_loop():
    """Main reminder checking loop"""
    last_cleanup = time.time()

    while True:
        try:
            now = datetime.now(LOCAL_TZ)

            # Check for due reminders
            due_reminders = get_due_reminders()
            for reminder_id, text, audio_file, video_file, job_id in due_reminders:
                timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n[{timestamp}] 🔔 Processing Due Reminder...")

                success = play_reminder(reminder_id, text, audio_file, video_file, job_id)

                if success:
                    mark_reminded(reminder_id)
                    print(f"✅ Reminder {reminder_id} completed and marked as reminded")
                else:
                    print(f"⚠️  Reminder {reminder_id} had playback issues but marking as reminded")
                    mark_reminded(reminder_id)  # Mark anyway to avoid repeating

            # Periodic cleanup
            if time.time() - last_cleanup > CLEANUP_INTERVAL:
                cleanup_old_files()
                last_cleanup = time.time()

        except Exception as e:
            print(f"[Reminder Manager] Error: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(CHECK_INTERVAL)


def main():
    """Main entry point"""
    global CHECK_INTERVAL

    parser = argparse.ArgumentParser(description="Voice Assistant Reminder Manager with Voice Cloning")
    parser.add_argument("--list", action="store_true", help="List pending reminders and exit")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old files and exit")
    parser.add_argument("--check-interval", type=int, default=CHECK_INTERVAL,
                        help=f"Check interval in seconds (default: {CHECK_INTERVAL})")
    parser.add_argument("--test", type=str, help="Test reminder playback with given text")

    args = parser.parse_args()

    # Initialize database
    init_database()

    if args.list:
        # List pending reminders
        reminders = list_pending_reminders()
        if reminders:
            print("Pending reminders:")
            for reminder_id, text, remind_at in reminders:
                # Convert ISO string back to readable format
                dt = datetime.fromisoformat(remind_at.replace('Z', '+00:00'))
                if dt.tzinfo is None:
                    dt = LOCAL_TZ.localize(dt)
                else:
                    dt = dt.astimezone(LOCAL_TZ)
                print(f"  {reminder_id}: {text} at {dt.strftime('%Y-%m-%d %I:%M %p')}")
        else:
            print("No pending reminders.")
        return

    if args.cleanup:
        # Clean up old files
        cleanup_old_files()
        return

    if args.test:
        # Test reminder playback
        print(f"🧪 Testing reminder playback...")
        success = play_reminder(0, args.test, None, None, "test")  # Use dummy ID for test
        if success:
            print("✅ Test completed successfully")
        else:
            print("❌ Test had issues")
        return

    # Update check interval if specified
    CHECK_INTERVAL = args.check_interval

    print("=" * 60)
    print("🎭 VOICE ASSISTANT REMINDER MANAGER WITH VOICE CLONING")
    print("=" * 60)
    print(f"🎤 Whisper model: N/A (not needed for reminders)")
    print(f"🗣️  XTTS model: {'✅ Loaded' if xtts_model else '❌ Failed'}")
    if xtts_model is None and xtts_error:
        print(f"   Error: {xtts_error}")
    print(f"🎬 Wav2Lip path: {WAV2LIP_PATH}")
    print(f"🎭 Voice reference: {VOICE_REFERENCE_PATH}")
    print(f"🖼️  Face image: {FACE_IMAGE_PATH}")
    print(f"🔍 VAD model: {'✅ Loaded' if vad_model else '❌ Failed'}")
    print(f"📁 Downloads dir: {DOWNLOADS_DIR}")
    print(f"🗃️  Database: {DB_PATH}")
    print(f"⏰ Check interval: {CHECK_INTERVAL} seconds")
    print(f"🧹 Cleanup interval: {CLEANUP_INTERVAL} seconds")
    print(f"🖥️  System: {SYSTEM}")
    print("=" * 60)

    # Show current status
    try:
        pending_count = len(list_pending_reminders())
        print(f"📋 Pending reminders: {pending_count}")
        if pending_count > 0:
            print("   Use --list to see details")
    except Exception as e:
        print(f"⚠️  Could not check pending reminders: {e}")

    print("\n🔄 Starting reminder monitoring loop...")
    print("   Press Ctrl+C to stop")

    try:
        # Run the async reminder loop
        asyncio.run(reminder_loop())
    except KeyboardInterrupt:
        print("\n\n🛑 Reminder manager stopped by user")
        print("👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()