#!/usr/bin/env python3
"""
voice_client.py - Simplified for LOCAL execution with Picovoice Cobra VAD
Wake-word → record with Cobra VAD → async server → download → LOCAL playback
Video window always 1000×1000
NOTE: Reminders are handled by standalone reminder_manager.py
"""
import pvporcupine
import pvcobra
import sounddevice as sd
import soundfile as sf
import numpy as np
import requests
import subprocess
import uuid
import os
import sys
import time

# CONFIGURATION - UPDATED WITH YOUR PICOVOICE API KEY
API_KEY = "eBSmaIsfy6xy+eUhSrIQAZ0Mmfn2GI5g579OebQcjRsBm32omu8ggA=="  # Replace with your complete key
SERVER_IP = "localhost"
SERVER_URL = f"http://{SERVER_IP}:9000/process_audio"
STATUS_ENDPOINT = f"http://{SERVER_IP}:9000/check_status"
HEALTH_ENDPOINT = f"http://{SERVER_IP}:9000/health"
POLL_TIMEOUT = 180

SAMPLE_RATE = 16000
CHANNELS = 1
# Cobra VAD settings
VOICE_PROBABILITY_THRESHOLD = 0.5  # Threshold for considering speech (0.0-1.0)
SILENCE_FRAMES_THRESHOLD = 150      # Number of consecutive non-speech frames before stopping
CHUNK_DURATION = 0.3

VIDEO_OPTS = ["-x", "1000", "-y", "1000"]


# ------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------
def record_with_cobra_vad(cobra) -> str:
    """Record audio using Picovoice Cobra VAD for more accurate speech detection"""
    print("🎙️  Recording with Cobra VAD… Speak now!")
    recorded = []
    silence_frame_count = 0
    speech_detected = False
    
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE,
                            channels=CHANNELS,
                            dtype='int16',
                            blocksize=cobra.frame_length) as stream:
            while True:
                # Read audio data in chunks that match Cobra's frame length
                pcm_data = stream.read(cobra.frame_length)[0]
                pcm = np.frombuffer(pcm_data, dtype=np.int16)
                recorded.append(pcm)
                
                # Process with Cobra VAD
                voice_probability = cobra.process(pcm)
                
                if voice_probability >= VOICE_PROBABILITY_THRESHOLD:
                    # Speech detected
                    speech_detected = True
                    silence_frame_count = 0
                    print(f"🎤 Speech detected (probability: {voice_probability:.2f})")
                else:
                    # No speech detected
                    if speech_detected:  # Only count silence after we've detected speech
                        silence_frame_count += 1
                        print(f"⏸️  Silence frame {silence_frame_count}/{SILENCE_FRAMES_THRESHOLD}")
                        
                        if silence_frame_count >= SILENCE_FRAMES_THRESHOLD:
                            print("✅ End of speech detected")
                            break
                
    except Exception as e:
        print(f"❌ Recording error: {e}")
        return None

    if not speech_detected:
        print("❌ No speech detected")
        return None
        
    audio = np.concatenate(recorded)
    path = f"/tmp/{uuid.uuid4().hex}.wav"
    sf.write(path, audio, SAMPLE_RATE)
    print(f"✅ Saved recording to {path} ({len(audio)/SAMPLE_RATE:.2f} seconds)")
    return path


def play_media(path: str, is_video: bool = False):
    """Play media with Ubuntu-compatible players"""
    try:
        if is_video:
            # Try different video players available on Ubuntu
            for player in ["ffplay", "vlc", "mpv"]:
                try:
                    if player == "ffplay":
                        subprocess.run([player, "-autoexit", "-loglevel", "quiet"] + VIDEO_OPTS + [path],
                                       check=True, timeout=180)
                    elif player == "vlc":
                        subprocess.run([player, "--intf", "dummy", "--play-and-exit", path],
                                       check=True, timeout=180)
                    elif player == "mpv":
                        subprocess.run([player, "--really-quiet", "--geometry=1000x1000", path],
                                       check=True, timeout=180)
                    return
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            print("❌ No video player found (tried ffplay, vlc, mpv)")
        else:
            # Try different audio players available on Ubuntu
            for player in ["aplay", "paplay", "ffplay"]:
                try:
                    if player == "ffplay":
                        subprocess.run([player, "-autoexit", "-nodisp", "-loglevel", "quiet", path],
                                       check=True, timeout=180)
                    else:
                        subprocess.run([player, "-q", path], check=True, timeout=180)
                    return
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            print("❌ No audio player found (tried aplay, paplay, ffplay)")
    except Exception as e:
        print(f"❌ Playback error: {e}")


def download(url: str, dest: str):
    try:
        r = requests.get(url, stream=True, timeout=180)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"❌ Download error: {e}")
        raise


def wait_for_job(job_id: str) -> dict:
    url = f"{STATUS_ENDPOINT}/{job_id}"
    for _ in range(POLL_TIMEOUT):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data["status"] in ("completed", "error"):
                    return data
        except requests.RequestException:
            pass
        time.sleep(1)
    return {"status": "timeout"}


def send_audio(file_path: str):
    if not file_path:
        return

    print("🤖 Uploading audio …")
    try:
        with open(file_path, "rb") as f:
            r = requests.post(SERVER_URL, files={"file": f}, timeout=180)
        if r.status_code != 200:
            print("❌ Server error:", r.status_code, r.text)
            return
        data = r.json()
        job_id = data.get("job_id")
        if not job_id:
            print("❌ No job_id received")
            return
        print("⏳ Waiting for spoken-reply media …")
        status = wait_for_job(job_id)
        if status["status"] != "completed":
            print("❌ Generation failed:", status.get("error"))
            return

        # Play response media
        if status.get("video_url"):
            v_url = f"http://{SERVER_IP}:9000{status['video_url']}"
            v_path = f"/tmp/resp_{uuid.uuid4().hex}.mp4"
            download(v_url, v_path)
            play_media(v_path, is_video=True)
            # Clean up
            try:
                os.remove(v_path)
            except:
                pass
        elif status.get("audio_url"):
            a_url = f"http://{SERVER_IP}:9000{status['audio_url']}"
            a_path = f"/tmp/resp_{uuid.uuid4().hex}.wav"
            download(a_url, a_path)
            play_media(a_path)
            # Clean up
            try:
                os.remove(a_path)
            except:
                pass
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
    finally:
        # Clean up input file
        try:
            os.remove(file_path)
        except:
            pass


# ------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------
def main():
    # Check if server is running
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=5)
        if r.status_code != 200:
            print("❌ Server not accessible. Make sure your FastAPI server is running on localhost:9000")
            return
    except Exception as e:
        print("❌ Cannot connect to server. Make sure your FastAPI server is running on localhost:9000")
        print(f"Error: {e}")
        return

    print("📝 NOTE: Reminders are handled by standalone reminder_manager.py")
    print("🎧 Make sure to run 'python reminder_manager.py' separately for reminders")

    # Initialize Picovoice components
    porcupine = None
    cobra = None
    
    try:
        print("🔧 Initializing Picovoice components...")
        
        # Initialize Porcupine for wake-word detection
        porcupine = pvporcupine.create(access_key=API_KEY, keywords=["picovoice"])
        print(f"✅ Porcupine initialized (Sample rate: {porcupine.sample_rate})")
        
        # Initialize Cobra for VAD
        cobra = pvcobra.create(access_key=API_KEY)
        print(f"✅ Cobra VAD initialized (Sample rate: {cobra.sample_rate}, Frame length: {cobra.frame_length})")
        
        # Verify sample rates match
        if porcupine.sample_rate != cobra.sample_rate:
            print(f"⚠️  Warning: Sample rate mismatch - Porcupine: {porcupine.sample_rate}, Cobra: {cobra.sample_rate}")
        
        print(f"🎧 Listening for wake-word 'picovoice' with Cobra VAD...")
        print(f"🔊 Voice threshold: {VOICE_PROBABILITY_THRESHOLD}, Silence frames: {SILENCE_FRAMES_THRESHOLD}")

        with sd.InputStream(samplerate=porcupine.sample_rate,
                            channels=1,
                            dtype='int16',
                            blocksize=porcupine.frame_length) as stream:
            while True:
                pcm = stream.read(porcupine.frame_length)[0]
                pcm = np.frombuffer(pcm, dtype=np.int16)
                if porcupine.process(pcm) >= 0:
                    print("🎉 Wake-word detected!")
                    audio = record_with_cobra_vad(cobra)
                    if audio:
                        send_audio(audio)
                        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have:")
        print("1. Valid Picovoice API key")
        print("2. Installed pvcobra: pip install pvcobra")
        print("3. Microphone access")
        print("4. FastAPI server running on localhost:9000")
        print("5. reminder_manager.py running separately for reminders")
    finally:
        # Clean up resources
        if porcupine:
            porcupine.delete()
        if cobra:
            cobra.delete()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Bye!")