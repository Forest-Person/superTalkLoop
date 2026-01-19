#!/usr/bin/env python3
import io
import flask
from flask import request, send_file
import numpy as np
import wave  # Using standard library instead of soundfile for writing

# --- IMPORT THE NEW PACKAGE ---
try:
    from supertonic import TTS
except ImportError:
    print("CRITICAL: 'supertonic' package not found. Run: pip install supertonic")
    exit(1)

app = flask.Flask(__name__)

# --- CONFIGURATION ---
HOST = "0.0.0.0"
PORT = 5002
USE_GPU = False  # Set to True if you have a GPU

# --- GLOBAL STATE ---
tts_engine = None
voice_cache = {}

def initialize_engine():
    global tts_engine, voice_cache
    print(f"--- Starting SuperTonic PyPI Server on Port {PORT} ---")
    
    # 1. Initialize Engine
    print("Initializing SuperTonic TTS Engine...")
    tts_engine = TTS(auto_download=True)
    
    # 2. Pre-load Voice Styles
    target_voices = [
        "M1", "M2", "M3", "M4", "M5", 
        "F1", "F2", "F3", "F4", "F5"
    ]
    
    print("Caching Voice Styles...")
    for v_name in target_voices:
        try:
            # The package handles finding the style file internally
            style = tts_engine.get_voice_style(voice_name=v_name)
            voice_cache[v_name] = style
            print(f"  [OK] Cached {v_name}")
        except Exception as e:
            print(f"  [!!] Could not load {v_name}: {e}")
            
    print("--- Engine Ready ---")

@app.route('/tts', methods=['POST'])
def tts_handler():
    try:
        # 1. Parse Request
        data = request.json
        text = data.get('text', '')
        voice_id = data.get('voice', 'M1')
        speed = float(data.get('speed', 1.0))
        
        if not text:
            return flask.jsonify({"error": "No text provided"}), 400

        # 2. Get Cached Style (Fallback to M1)
        style = voice_cache.get(voice_id)
        if style is None:
            style = voice_cache.get("M1")

        # 3. Generate Speech
        # NOTE: 'total_step' is removed as it caused errors in your version.
        try:
            wav, duration = tts_engine.synthesize(
                text=text, 
                voice_style=style, 
                speed=speed
            )
        except TypeError:
            # Fallback if 'speed' is not supported in specific version
            wav, duration = tts_engine.synthesize(
                text=text, 
                voice_style=style
            )

        # 4. Convert Data for Writing
        # Ensure it's a numpy array on CPU
        if hasattr(wav, 'cpu'): wav = wav.cpu().numpy()
        if hasattr(wav, 'numpy'): wav = wav.numpy()
        wav = np.array(wav)
        
        # Remove batch dimensions if present (1, N) -> (N,)
        wav = np.squeeze(wav)

        # Convert Float32 (-1.0 to 1.0) to Int16 (-32768 to 32767) for 'wave' module
        audio_int16 = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)

        # 5. Write to BytesIO using 'wave' (More robust on Termux)
        mem_file = io.BytesIO()
        with wave.open(mem_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) # 16-bit
            wf.setframerate(tts_engine.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        mem_file.seek(0)

        return send_file(mem_file, mimetype="audio/wav")

    except Exception as e:
        print(f"TTS Generation Error: {e}")
        return flask.jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    initialize_engine()
    app.run(host=HOST, port=PORT, threaded=True)
