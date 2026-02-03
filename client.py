#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversational AI v30.6
- Compatible with SuperTonic PyPI Server
- Robust Error Handling
"""                                                                              
import os
import json
import re
import requests
import wave
import io
import sys
import time
import ast
import signal
import random
import datetime
from collections import deque
import threading
import queue
import subprocess
import select
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --- Imports Check ---
try:
    import numpy as np
    import sounddevice as sd
    import soundfile as sf
    import webrtcvad
except ImportError as e:
    print(f"[CRITICAL] Missing Dependency: {e}")
    sys.exit(1)

# --- Configuration ---
# UPDATE THESE PATHS TO MATCH YOUR SYSTEM
MODEL_PATH = os.path.expanduser("~/whisper.cpp/models/ggml-tiny.en.bin")
WHISPER_EXECUTABLE = os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli")
LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"
TTS_SERVER_URL = "http://localhost:5002/tts"

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)

# --- VAD & INTERRUPTION SETTINGS ---
# --- VAD & INTERRUPTION SETTINGS ---
VAD_INTERRUPT_SENSITIVITY = 1      # Changed from 2 to 1 (More sensitive speech check)
VAD_RECORD_SENSITIVITY = 2         # Changed from 1 to 2 (Matches interrupt strictness)
RMS_THRESHOLD = 0.015              # Decreased from 0.035 (Half volume for easier interruption)
INTERRUPT_FRAME_COUNT = 10         # Decreased from 12 (Slightly snappier response)
SILENCE_THRESHOLD_MS = 1500        # (Unchanged, keeps the listening window natural)
AUTONOMY_TIMEOUT = 2               # Seconds to wait before agents take initiative

# --- Display & Colors ---
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"

AGENT_COLORS = [Colors.GREEN, Colors.YELLOW, Colors.BLUE, Colors.MAGENTA, Colors.CYAN]

# --- Logging Helper ---
def log(subsystem: str, message: str):
    if subsystem in ["AUDIO", "TTS", "ROUTER"]: return
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"{Colors.DIM}[{ts}] [{subsystem}] {message}{Colors.RESET}")

# --- Data Structures ---
@dataclass
class AudioTask:
    turn_id: int
    data: Any # (data, fs)
    is_sentinel: bool = False

# --- Globals ---
audio_queue = queue.Queue()
http_session = requests.Session()
audio_lock = threading.Lock()
interruption_event = threading.Event()
ai_is_speaking_event = threading.Event()
auto_loop = False

global_turn_id = 0
conversation_history = []
current_persona_prompt = "You are SuperTonic, a helpful and precise voice assistant."
FINAL_AUDIO = "whisper_ready.wav"
pending_action = None

# --- CLASSES ---

class ChatAgent:
    def __init__(self, name, system_prompt, voice_id="M1", color=Colors.WHITE):
        self.name = name
        self.base_prompt = system_prompt
        self.voice_id = voice_id
        self.color = color
        self.history = []

    def _get_perspective_history(self):
        perspective_history = []
        for msg in self.history:
            content = msg['content']
            role = msg['role']
            if role == "assistant":
                perspective_history.append({"role": "assistant", "content": content})
            else:
                perspective_history.append({"role": "user", "content": content})
        return perspective_history

    def respond_stream(self, chat_context_messages, director_instruction, other_agent_names=[]):
        perspective_msgs = self._get_perspective_history()

        instruction = (
            f"IDENTITY: You are {self.name}.\n"
            f"DESCRIPTION: {self.base_prompt}\n"
            f"CURRENT TASK: {director_instruction}\n"
            "----------------\n"
            "RULES:\n"
            "1. Speak ONLY as yourself (First Person 'I').\n"
            "2. Do NOT start lines with your name.\n"
            "3. Do NOT use quotation marks.\n"
            "4. Keep it conversational.\n"
            "5. STOP immediately if you finish."
        )

        full_context = [{"role": "system", "content": instruction}] + perspective_msgs + chat_context_messages

        stop_sequences = ["\nUser:", "\nSystem:", f"\n{self.name}:", "\nYOU:"]
        for agent_name in other_agent_names:
            stop_sequences.append(f"\n{agent_name}:")

        try:
            resp = http_session.post(
                LLAMA_SERVER_URL,
                json={
                    "messages": full_context,
                    "temperature": 0.65,
                    "max_tokens": 200,
                    "stream": True,
                    "stop": stop_sequences,
                    "repeat_penalty": 1.12
                },
                stream=True,
                timeout=30
            )
            if resp.status_code == 200:
                full_content = ""
                for chunk in resp.iter_content(None):
                    if chunk:
                        for line in chunk.decode('utf-8').split('\n'):
                            if line.startswith('data: '):
                                try:
                                    j = json.loads(line[6:])
                                    token = j['choices'][0]['delta'].get('content', '')
                                    finish_reason = j['choices'][0].get('finish_reason')

                                    if finish_reason == "stop": break

                                    if token:
                                        if len(full_content) < len(self.name) + 2:
                                            temp_check = (full_content + token).strip()
                                            if temp_check.lower().startswith(f"{self.name.lower()}:"):
                                                token = ""
                                        full_content += token
                                        yield token
                                except Exception: pass

                clean_content = full_content.replace(f"{self.name}:", "").strip()
                clean_content = clean_content.strip('"').strip("'")

                for agent_name in other_agent_names:
                    clean_content = clean_content.split(f"{agent_name}:")[0]

                self.history.append({"role": "user", "content": chat_context_messages[-1]['content']})
                self.history.append({"role": "assistant", "content": clean_content})

                if len(self.history) > 10:
                    self.history = self.history[-10:]

        except Exception as e:
            yield f"[Error: {e}]"

class AgentDirector:
    def __init__(self, session, model_url):
        self.session = session
        self.url = model_url

    def start_conversation(self, active_agents: List[ChatAgent], topic: Optional[str] = None):
        if not active_agents: return None, "Wait for user."

        agent_map = {a.name: a for a in active_agents}
        profiles = "\n".join([f"- {a.name}: {a.base_prompt}" for a in active_agents])

        system_prompt = (
            f"Set the scene with:\n{profiles}\n"
            f"Topic: {topic if topic else 'Interesting banter'}\n"
            "Output JSON: {\"next_agent\": \"Name\", \"instruction\": \"...\"}"
        )

        try:
            resp = self.session.post(
                self.url,
                json={
                    "messages": [{"role": "system", "content": system_prompt}],
                    "temperature": 0.7, "max_tokens": 100
                }, timeout=10
            )
            if resp.status_code == 200:
                content = resp.json()['choices'][0]['message']['content']
                content = content.replace("```json", "").replace("```", "").strip()
                data = json.loads(content)
                name = data.get("next_agent")
                instruction = data.get("instruction", "Hello.")
                if name in agent_map:
                    return agent_map[name], instruction
        except Exception: pass

        target = random.choice(active_agents)
        return target, f"Start a conversation about {topic if topic else 'something interesting'}."

    def select_next_speaker(self, active_agents: List[ChatAgent], last_speaker: str, last_message: str, silence_scores: Dict[str, int], narrative_context: str = ""):
        if not active_agents: return None, "Respond."
        if len(active_agents) == 1: return active_agents[0], "Respond."

        # Filter out the last speaker immediately (Turn Enforcement)
        available_agents = [a for a in active_agents if a.name != last_speaker]
        if not available_agents: available_agents = active_agents 

        agent_map = {a.name: a for a in available_agents}
        # Pass silence scores to influence the LLM (Silence Tracking)
        profiles = "\n".join([f"- {a.name} (Silent for {silence_scores.get(a.name, 0)} turns): {a.base_prompt}" for a in available_agents])

        system_prompt = (
            f"Pick the next speaker. NEVER pick {last_speaker}.\n"
            f"Cast:\n{profiles}\n"
            f"Current Situation/Event: {narrative_context}\n"
            f"Last Word: \"{last_message[:150]}\"\n"
            "Output JSON: {\"next_agent\": \"Name\", \"tone\": \"short vibe\", \"objective\": \"react to situation\"}"
        )
        try:
            resp = self.session.post(
                self.url,
                json={
                    "messages": [{"role": "system", "content": system_prompt}],
                    "temperature": 0.65, "max_tokens": 120, "repeat_penalty": 1.1
                }, timeout=6
            )
            if resp.status_code == 200:
                content = resp.json()['choices'][0]['message']['content']
                content = content.replace("```json", "").replace("```", "").strip()
                data = json.loads(content)
                name = data.get("next_agent")
                tone = data.get("tone", "spicy")
                obj = data.get("objective", "drive the conversation")
                
                if name in agent_map:
                    # Dynamic Goals and Style
                    return agent_map[name], f"Context: {narrative_context}. Style: {tone}. Goal: {obj}"
        except Exception: pass

        target = random.choice(available_agents)
        return target, "Keep it moving."

class MetaDirector:
    def __init__(self, session, model_url):
        self.session = session
        self.url = model_url
        self.current_narrative = "The conversation is just beginning."

    def assess_situation(self, history: List[Dict[str, str]]):
        if not history: return
        
        # Analyze the last few turns
        recent_log = "\n".join([f"{m['role']}: {m['content']}" for m in history[-6:]])
        
        system_prompt = (
            "You are the Meta-Director. You control the plot.\n"
            "Analyze the recent chat. Is it repetitive (looping)? Is it boring?\n"
            "Generate a 'Stage Direction' to advance the story or break the loop.\n"
            "Examples: 'The lights flicker ominously.', 'A loud knock at the door.', 'Everyone realizes they are hungry.', 'Raise the tension.'\n"
            "Output JSON: {\"status\": \"ok/stuck\", \"new_direction\": \"...\"}"
        )

        try:
            resp = self.session.post(
                self.url,
                json={
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Recent Chat:\n{recent_log}"}
                    ],
                    "temperature": 0.7, "max_tokens": 100
                }, timeout=10
            )
            if resp.status_code == 200:
                content = resp.json()['choices'][0]['message']['content']
                content = content.replace("```json", "").replace("```", "").strip()
                data = json.loads(content)
                direction = data.get("new_direction")
                if direction:
                    self.current_narrative = direction
                    log("META", f"New Direction: {direction}")
                    print(f"{Colors.MAGENTA}[Meta Director] ➤ {direction}{Colors.RESET}")
        except Exception as e:
            log("META", f"Error: {e}")

class ChatRoom:
    def __init__(self):
        self.agents: Dict[str, ChatAgent] = {}
        self.turns_since_spoke: Dict[str, int] = {}
        self.director = AgentDirector(http_session, LLAMA_SERVER_URL)
        self.meta_director = MetaDirector(http_session, LLAMA_SERVER_URL)
        self.system_voice = None
        # 10 Voices available
        self.all_voices = [
            "M1", "M2", "M3", "M4", "M5", 
            "F1", "F2", "F3", "F4", "F5"
        ]

    def set_system_voice(self, voice_id):
        self.system_voice = voice_id

    def add_agent(self, name, prompt):
        if len(self.agents) >= 10:
            return "Cannot add agent. Maximum capacity (10) reached."

        used_voices = {agent.voice_id for agent in self.agents.values()}
        if self.system_voice:
            used_voices.add(self.system_voice)

        available_voices = [v for v in self.all_voices if v not in used_voices]

        if not available_voices:
            voice = random.choice(self.all_voices)
        else:
            voice = random.choice(available_voices)
        
        # Assign Color
        color_idx = len(self.agents) % len(AGENT_COLORS)
        color = AGENT_COLORS[color_idx]

        self.agents[name] = ChatAgent(name, prompt, voice_id=voice, color=color)
        self.turns_since_spoke[name] = 10 # Encourage early participation
        log("SYSTEM", f"Created Agent: {name} (Voice: {voice})")
        return f"Created {name}."

    def delete_agent(self, name):
        target = None
        for k in self.agents.keys():
            if name.lower() in k.lower():
                target = k
                break
        if target:
            del self.agents[target]
            if target in self.turns_since_spoke: del self.turns_since_spoke[target]
            return f"Deleted agent {target}."
        return f"Could not find agent {name}."

    def get_agent_list_str(self):
        if not self.agents: return "None."
        return ", ".join([f"{name} ({agent.voice_id})" for name, agent in self.agents.items()])

    def create_default_group(self):
        log("SYSTEM", "Creating Default 4-Agent Group...")
        defaults = [
            ("Orion", "A logical and analytical strategist who loves data."),
            ("Lyra", "A creative and enthusiastic dreamer who loves art."),
            ("Atlas", "A grounded and strong-willed protector."),
            ("Selene", "A mysterious and philosophical observer.")
        ]
        for name, prompt in defaults:
            self.add_agent(name, prompt)

    def run_session(self, participants_str=None, topic=None, auto_start=False):
        global auto_loop
        auto_loop = auto_start
        log("SYSTEM", f"Entering Room. Participants: {participants_str}, Topic: {topic}, AutoLoop: {auto_loop}")
        
        if not self.agents:
            enqueue_tts("No agents found. Creating default group.", global_turn_id)
            self.create_default_group()

        active_agents = []

        if participants_str and "all" not in participants_str.lower():
            requested = [x.strip().lower() for x in participants_str.split(',')]
            for name, agent in self.agents.items():
                if any(r in name.lower() for r in requested):
                    active_agents.append(agent)
        else:
            active_agents = list(self.agents.values())

        if not active_agents:
            enqueue_tts("I couldn't find those agents.", global_turn_id)
            wait_for_turn()
            return

        names = ", ".join([a.name for a in active_agents])
        if not topic:
             enqueue_tts(f"Room ready with: {names}.", global_turn_id)
             wait_for_turn()

        starter_agent, start_instruction = self.director.start_conversation(active_agents, topic)
        last_speaker_name = "User"
        last_message_content = "The room is open."
        session_history = []
        turn_count = 0

        if starter_agent:
            # Silence Tracker logic
            for n in self.turns_since_spoke: self.turns_since_spoke[n] += 1
            self.turns_since_spoke[starter_agent.name] = 0

            print(f"{Colors.DIM}[{starter_agent.name}] Dir: {start_instruction}{Colors.RESET}")
            kickoff_context = [{"role": "system", "content": "The scene is beginning."}]
            other_names = [a.name for a in active_agents if a.name != starter_agent.name]
            other_names.append("User")

            stream_gen = starter_agent.respond_stream(
                kickoff_context,
                director_instruction=start_instruction,
                other_agent_names=other_names
            )
            smart_buffer_stream(stream_gen, starter_agent.name, voice_id=starter_agent.voice_id, color=starter_agent.color)

            if starter_agent.history:
                last_message_content = starter_agent.history[-1]['content']
                last_speaker_name = starter_agent.name
                session_history.append({"role": last_speaker_name, "content": last_message_content})

            wait_for_turn()

        while True:
            interruption_event.clear()
            cleanup_files()

            audio_file = record_with_vad(timeout=AUTONOMY_TIMEOUT)
            user_text = ""

            if audio_file:
                try:
                    cmd = [WHISPER_EXECUTABLE, "-m", MODEL_PATH, "-f", audio_file, "-otxt"]
                    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    user_text = re.sub(r'\[.*?\]', '', res.stdout).strip().replace('\n', ' ')
                except: pass

            if user_text:
                if "exit" in user_text.lower():
                    enqueue_tts("Leaving room.", global_turn_id)
                    wait_for_turn()
                    break
                print(f"\n{Colors.CYAN}[User]: {user_text}{Colors.RESET}")
                last_message_content = user_text
                last_speaker_name = "User"
                session_history.append({"role": "User", "content": user_text})
            elif not active_agents:
                continue

            if not user_text: log("DIRECTOR", "Autonomy Triggered...")

            # --- META DIRECTOR CYCLE ---
            turn_count += 1
            if turn_count % 4 == 0:
                self.meta_director.assess_situation(session_history)

            # Pass silence scores and Meta Narrative to Director
            target_agent, instruction = self.director.select_next_speaker(
                active_agents, 
                last_speaker_name, 
                last_message_content, 
                self.turns_since_spoke,
                narrative_context=self.meta_director.current_narrative
            )
            if not target_agent: continue

            # Silence Tracker logic
            for n in self.turns_since_spoke: self.turns_since_spoke[n] += 1
            self.turns_since_spoke[target_agent.name] = 0

            if interruption_event.is_set(): continue
            print(f"{Colors.DIM}[{target_agent.name}] Dir: {instruction}{Colors.RESET}")

            context_input = f"{last_speaker_name}: {last_message_content}"
            other_names = [a.name for a in active_agents if a.name != target_agent.name]
            other_names.append("User")

            stream_gen = target_agent.respond_stream(
                [{"role": "user", "content": context_input}],
                director_instruction=instruction,
                other_agent_names=other_names
            )

            smart_buffer_stream(stream_gen, target_agent.name, voice_id=target_agent.voice_id, color=target_agent.color)

            if target_agent.history:
                last_message_content = target_agent.history[-1]['content']
            else:
                last_message_content = "..."
            
            session_history.append({"role": target_agent.name, "content": last_message_content})
            if len(session_history) > 15: session_history.pop(0)

            last_speaker_name = target_agent.name
            wait_for_turn()

chat_room = ChatRoom()

# --- AUDIO & WORKER FUNCTIONS ---

def signal_handler(sig, frame):
    interruption_event.set()
    os._exit(0)
signal.signal(signal.SIGINT, signal_handler)

def cleanup_files():
    if os.path.exists(FINAL_AUDIO):
        try: os.remove(FINAL_AUDIO)
        except: pass

def calculate_rms(frame_bytes):
    try:
        samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
        return np.sqrt(np.mean(samples**2)) / 32768.0
    except: return 0.0

def trigger_interruption(source="User"):
    global global_turn_id
    if auto_loop: return
    if not interruption_event.is_set():
        print(f"\n{Colors.RED}[!] 🛑 Interrupted by {source}{Colors.RESET}")
        interruption_event.set()
        global_turn_id += 1
        try:
            while not audio_queue.empty():
                audio_queue.get_nowait()
        except queue.Empty: pass
        audio_queue.put(AudioTask(turn_id=global_turn_id, data=None, is_sentinel=False))

def high_pass_filter(data, alpha=0.95):
    float_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    filtered = np.append(float_data[0], float_data[1:] - alpha * float_data[:-1])
    return np.clip(filtered, -32768, 32767).astype(np.int16).tobytes()

def interruption_listener_worker():
    vad = webrtcvad.Vad(VAD_INTERRUPT_SENSITIVITY)
    while True:
        ai_is_speaking_event.wait()
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', blocksize=FRAME_SIZE) as stream:
                speech_frames = 0
                while ai_is_speaking_event.is_set() and not interruption_event.is_set():
                    if auto_loop: 
                        time.sleep(0.5)
                        continue
                        
                    frame, overflow = stream.read(FRAME_SIZE)
                    if overflow: continue
                    raw_bytes = frame.tobytes()
                    filtered_bytes = high_pass_filter(raw_bytes, alpha=0.97)
                    if calculate_rms(filtered_bytes) < RMS_THRESHOLD:
                        speech_frames = max(0, speech_frames - 1)
                        continue
                    if vad.is_speech(filtered_bytes, SAMPLE_RATE):
                        speech_frames += 1
                        if speech_frames >= INTERRUPT_FRAME_COUNT:
                            trigger_interruption("Voice")
                            break
                    else: speech_frames = 0
        except: time.sleep(1)

def player_worker():
    log("SYSTEM", "Audio Player Started")
    while True:
        try:
            task = audio_queue.get()
            if task.is_sentinel:
                ai_is_speaking_event.clear()
                continue
            if task.data is None:
                ai_is_speaking_event.clear()
                sd.stop()
                continue
            if interruption_event.is_set(): continue
            if task.turn_id != global_turn_id: continue

            if not ai_is_speaking_event.is_set(): ai_is_speaking_event.set()

            data, fs = task.data
            if len(data) == 0: continue

            with audio_lock:
                try:
                    with sd.OutputStream(samplerate=fs, channels=1, dtype='float32') as stream:
                        silence = np.zeros(int(fs * 0.05), dtype=np.float32)
                        stream.write(silence)
                        chunk_size = 1024
                        total_len = len(data)
                        current_pos = 0
                        while current_pos < total_len:
                            if interruption_event.is_set():
                                stream.abort()
                                break
                            end_pos = min(current_pos + chunk_size, total_len)
                            stream.write(data[current_pos:end_pos])
                            current_pos = end_pos
                except Exception as e:
                    log("AUDIO", f"Playback Fail: {e}")
        except Exception as e:
            log("AUDIO", f"Worker Error: {e}")

def enqueue_tts(text, turn_id, voice_id="M1"):
    if interruption_event.is_set() or turn_id != global_turn_id or not text.strip():
        return
    try:
        # Client still sends default parameters, but server handles ignoring them if needed
        payload = {
            "text": text,
            "voice": voice_id,
            "speed": 1.05,
            "steps": 5
        }
        response = http_session.post(TTS_SERVER_URL, json=payload, timeout=10)

        if response.status_code == 200:
            with io.BytesIO(response.content) as buf:
                data, fs = sf.read(buf)
                if data.dtype != np.float32: data = data.astype(np.float32)
                if not interruption_event.is_set() and turn_id == global_turn_id:
                    audio_queue.put(AudioTask(turn_id=turn_id, data=(data, fs)))
    except Exception as e:
        log("TTS", f"Err: {e}")

def wait_for_turn():
    sentinel = AudioTask(turn_id=global_turn_id, data=None, is_sentinel=True)
    audio_queue.put(sentinel)
    while ai_is_speaking_event.is_set() and not interruption_event.is_set():
        time.sleep(0.1)

def play_startup_sound():
    fs = 16000
    duration = 0.3
    f = 440.0
    t = np.arange(int(fs * duration)) / fs
    wave = 0.5 * np.sin(2 * np.pi * f * t)
    wave = wave.astype(np.float32)
    sd.play(wave, fs, blocking=True)

def record_with_vad(timeout: Optional[float] = None):
    global global_turn_id, auto_loop
    ai_is_speaking_event.clear()
    
    if select.select([sys.stdin], [], [], 0.0)[0]:
        sys.stdin.readline()
        auto_loop = not auto_loop
        print(f"\n{Colors.DIM}[System] 🛑 Non-Interactive Mode: {'ON' if auto_loop else 'OFF'}{Colors.RESET}")

    if auto_loop:
        time.sleep(0.5)
        return None

    time.sleep(0.2)
    vad = webrtcvad.Vad(VAD_RECORD_SENSITIVITY)
    start_time = time.time()

    label = "Listening" if timeout is None else f"Auto-Listen ({timeout}s)"
    print(f"\n{Colors.DIM}[Mic] 🎤 {label}...{Colors.RESET}", end="", flush=True)

    try:
        with audio_lock: pass
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', blocksize=FRAME_SIZE) as stream:
            ring_buffer = deque(maxlen=30)
            voiced_frames = []
            is_speaking = False
            silence_counter = 0

            while True:
                if interruption_event.is_set(): return None
                if timeout and not is_speaking:
                    if (time.time() - start_time) > timeout:
                        print(f"{Colors.DIM} (Timeout){Colors.RESET}")
                        return None

                frame, overflow = stream.read(FRAME_SIZE)
                if overflow: continue
                frame_bytes = frame.tobytes()
                is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)

                if not is_speaking:
                    ring_buffer.append(frame_bytes)
                    if is_speech:
                        is_speaking = True
                        silence_counter = 0
                        voiced_frames.extend(ring_buffer)
                        ring_buffer.clear()
                        print(f"{Colors.DIM}●{Colors.RESET}", end="", flush=True)
                else:
                    voiced_frames.append(frame_bytes)
                    if not is_speech:
                        silence_counter += 1
                        if silence_counter >= (SILENCE_THRESHOLD_MS // FRAME_MS):
                            print(f"{Colors.DIM} Done.{Colors.RESET}")
                            break
                    else: silence_counter = 0

            if not voiced_frames: return None

            global_turn_id += 1
            with wave.open(FINAL_AUDIO, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b''.join(voiced_frames))
            return FINAL_AUDIO
    except: return None

def smart_buffer_stream(token_generator, label="AI", voice_id="M1", color=Colors.WHITE):
    sentence_buffer = ""
    is_first_chunk = True
    FIRST_CHUNK_WORDS = 8
    SUBSEQ_CHUNK_WORDS = 25

    print(f"{color}{label}: {Colors.RESET}", end="")

    for token in token_generator:
        if interruption_event.is_set():
            print(f"{Colors.RED} [Interrupted]{Colors.RESET}")
            break

        sys.stdout.write(f"{color}{token}{Colors.RESET}")
        sys.stdout.flush()

        clean_token = token.replace("<|tool_call_start|>", "").replace("<|tool_call_end|>", "")
        if any(char in clean_token for char in "[]()"):
            continue

        sentence_buffer += clean_token
        words = sentence_buffer.split()
        word_count = len(words)
        has_punctuation = bool(re.search(r'[.!?\n]', clean_token))
        should_flush = False

        if is_first_chunk:
            if (word_count >= FIRST_CHUNK_WORDS and has_punctuation) or (has_punctuation and word_count > 0):
                should_flush = True
        else:
            if word_count >= SUBSEQ_CHUNK_WORDS and has_punctuation:
                should_flush = True
            elif word_count >= 50 and " " in clean_token:
                should_flush = True

        if should_flush:
            if sentence_buffer.strip():
                enqueue_tts(sentence_buffer, global_turn_id, voice_id)
                sentence_buffer = ""
                is_first_chunk = False

    if sentence_buffer.strip() and not interruption_event.is_set():
        enqueue_tts(sentence_buffer, global_turn_id, voice_id)
    print("")

TOOLS_SCHEMA = [
    {"name": "wipe_all_memory", "description": "Resets the main chat history. No params.", "parameters": {}},
    {"name": "create_agent", "description": "Create a new agent.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "description": {"type": "string"}}, "required": ["name", "description"]}},
    {"name": "delete_agent", "description": "Deletes an agent (requires confirmation).", "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "enter_chat_room", "description": "Enters chat room. Param 'participants' can be 'all' or comma-list. 'topic' is optional.", "parameters": {"type": "object", "properties": {"participants": {"type": "string"}, "topic": {"type": "string"}}, "required": []}},
    {"name": "list_agents", "description": "Lists all currently active agents.", "parameters": {}},
    {"name": "list_tools", "description": "Lists all available tools and commands.", "parameters": {}},
    {"name": "display_history", "description": "Prints the current conversation history to the console.", "parameters": {}}
]

class ToolManager:
    @staticmethod
    def extract_lfm_call(text):
        python_pattern = r'\[(\w+)\((.*?)\)\]'
        match = re.search(python_pattern, text, re.DOTALL)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2).strip()
            params = {}
            if args_str:
                try:
                    tree = ast.parse(f"call({args_str})")
                    if hasattr(tree.body[0].value, 'keywords'):
                        for keyword in tree.body[0].value.keywords:
                            params[keyword.arg] = ast.literal_eval(keyword.value)
                except: pass
            return {"tool": tool_name, "parameters": params}
        return None

    @staticmethod
    def execute(tool_data):
        global conversation_history, pending_action
        try:
            tool_name = tool_data.get("tool")
            params = tool_data.get("parameters", {})
            log("TOOL", f"Executing {tool_name} {params}")

            if tool_name == "list_tools":
                tools_list = [f"- {t['name']}" for t in TOOLS_SCHEMA]
                return "Tools: " + ", ".join(tools_list)
            elif tool_name == "list_agents":
                return f"Current Agents: {chat_room.get_agent_list_str()}"
            elif tool_name == "display_history":
                if not conversation_history: return "Memory is empty."
                print("\n--- CURRENT MEMORY ---")
                for msg in conversation_history: print(f"[{msg['role'].upper()}]: {msg['content']}")
                print("----------------------\n")
                return f"History displayed. {len(conversation_history)} messages."
            elif tool_name == "wipe_all_memory":
                pending_action = {"type": "wipe_memory"}
                return "Are you sure you want to delete all memory? Say 'Yes' to confirm."
            elif tool_name == "delete_agent":
                name = params.get("name")
                found = False
                for k in chat_room.agents.keys():
                    if name.lower() in k.lower():
                        name = k
                        found = True
                        break
                if not found: return f"Agent '{name}' not found."
                pending_action = {"type": "delete_agent", "name": name}
                return f"Are you sure you want to delete agent '{name}'? Say 'Yes' to confirm."
            elif tool_name == "create_agent":
                return chat_room.add_agent(params.get("name"), params.get("description"))
            elif tool_name == "enter_chat_room":
                p = params.get("participants", "all")
                t = params.get("topic", None)
                # Format: __ENTER_CHAT_ROOM__:participants|topic
                return f"__ENTER_CHAT_ROOM__:{p}|{t if t else ''}"
            return None
        except Exception as e: return f"Tool Error: {e}"

class Orchestrator:
    @staticmethod
    def route_request(user_text):
        ut = user_text.lower()
        triggers = ["create agent","make new agent", "delete agent", "chat room", "reset memory", "wipe memory", "add agent", "list agents", "list tools", "show agents", "what tools", "show history", "show memory", "delete history"]
        if any(x in ut for x in triggers): return "SYSTEM"
        try:
            resp = http_session.post(LLAMA_SERVER_URL,
                json={
                    "messages": [{"role": "system", "content": "Classify: SYSTEM (tools) or CHAT (talk)."}, {"role": "user", "content": user_text}],
                    "temperature": 0.1, "max_tokens": 10
                }, timeout=5)
            if resp.status_code == 200:
                if "SYSTEM" in resp.json()['choices'][0]['message']['content']: return "SYSTEM"
        except: pass
        return "CHAT"

    @staticmethod
    def stream_chat_agent(user_text):
        log("ROUTER", "Routing to CHAT")
        messages = [{"role": "system", "content": current_persona_prompt}] + conversation_history + [{"role": "user", "content": user_text}]
        resp = http_session.post(LLAMA_SERVER_URL,
            json={"messages": messages, "stream": True, "temperature": 0.7},
            stream=True, timeout=60)

        full_response_text = ""
        for chunk in resp.iter_content(None):
            if chunk:
                for line in chunk.decode('utf-8').split('\n'):
                    if line.startswith('data: '):
                        try:
                            j = json.loads(line[6:])
                            token = j['choices'][0]['delta'].get('content', '')
                            if token:
                                full_response_text += token
                                yield token
                        except Exception:
                            pass
        return full_response_text

    @staticmethod
    def stream_admin_agent(user_text):
        log("ROUTER", "Routing to ADMIN")
        current_agents = chat_room.get_agent_list_str()
        system_prompt = (
            "You are the System Admin.\n"
            f"CURRENT AGENTS: {current_agents}\n"
            f"TOOLS: {json.dumps(TOOLS_SCHEMA)}\n\n"
            "EXAMPLES:\n"
            "User: 'Create agent Bob' -> Output: [create_agent(name=\"Bob\", description=\"Builder\")]\n"
            "User: 'List agents' -> Output: [list_agents()]\n"
            "User: 'What tools?' -> Output: [list_tools()]\n"
            "User: 'Chat room about Mars' -> Output: [enter_chat_room(topic=\"Mars\")]\n"
            "User: 'Wipe memory' -> Output: [wipe_all_memory()]\n"
            "IMPORTANT: You must output the tool command [tool_name(...)] to perform the action. Do not just say you did it."
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
        resp = http_session.post(LLAMA_SERVER_URL,
            json={"messages": messages, "stream": True, "temperature": 0.1},
            stream=True, timeout=60)

        full_response_text = ""
        for chunk in resp.iter_content(None):
            if chunk:
                for line in chunk.decode('utf-8').split('\n'):
                    if line.startswith('data: '):
                        try:
                            j = json.loads(line[6:])
                            token = j['choices'][0]['delta'].get('content', '')
                            if token:
                                full_response_text += token
                                yield token
                        except Exception:
                            pass
        return full_response_text

def main():
    global conversation_history, pending_action
    try: sd.query_devices()
    except: return

    threading.Thread(target=player_worker, daemon=True).start()
    threading.Thread(target=interruption_listener_worker, daemon=True).start()

    play_startup_sound()
    log("SYSTEM", "SuperTonic v30.6 - PyPI Robust")

    # Select a random voice for the system/admin for this session
    system_voice = random.choice(chat_room.all_voices)
    chat_room.set_system_voice(system_voice)
    log("SYSTEM", f"System Voice assigned: {system_voice}")

    # --- Initial Selection Mode ---
    print("\n--- Startup Menu ---")
    print("[C]hat Mode (Default)")
    print("[A]gent Creation Mode")
    print("[M]eet the Moms (Quick Start)")
    choice = input("Select Option (c/a/m): ").strip().lower()

    if choice == 'm':
        log("SYSTEM", "Creating Suburban Moms...")
        moms = [
            ("Karen", "Passive-aggressive PTA president who notices everyone's lawn flaws."),
            ("Linda", "Overly competitive baker who insists her cookies are 'organic' and better."),
            ("Susan", "The gossip queen who 'just wants everyone to get along' while stirring the pot.")
        ]
        for n, d in moms:
            chat_room.add_agent(n, d)
        
        start_now = input("Enter Chat Room now? (y/n): ").strip().lower()
        if start_now == 'y':
            auto = input("Enable Auto-Loop (Non-interactive)? (y/n): ").strip().lower() == 'y'
            chat_room.run_session("all", "The upcoming bake sale planning.", auto_start=auto)

    elif choice == 'a':
        print("\n--- Agent Creation Mode (Max 10) ---")
        while len(chat_room.agents) < 10:
            name = input("Agent Name: ").strip()
            if not name: break
            desc = input(f"Description for {name}: ").strip()
            
            res = chat_room.add_agent(name, desc)
            print(f"-> {res}")

            if len(chat_room.agents) >= 10:
                print("Max agents reached.")
                break
            
            cont = input("Create another? (y/n): ").strip().lower()
            if cont != 'y': break
        
        if chat_room.agents:
            start_now = input("Enter Chat Room with these agents? (y/n): ").strip().lower()
            if start_now == 'y':
                topic = input("Topic (optional): ").strip()
                auto = input("Enable Auto-Loop (Non-interactive)? (y/n): ").strip().lower() == 'y'
                chat_room.run_session("all", topic if topic else None, auto_start=auto)

    try:
        while True:
            interruption_event.clear()
            cleanup_files()

            audio_file = record_with_vad(timeout=None)
            if not audio_file: continue

            try:
                cmd = [WHISPER_EXECUTABLE, "-m", MODEL_PATH, "-f", audio_file, "-otxt"]
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                user_text = re.sub(r'\[.*?\]', '', res.stdout).strip().replace('\n', ' ')
                if not user_text: continue
                print(f"User: {user_text}")
            except: continue

            if pending_action:
                is_confirmed = any(x in user_text.lower() for x in ["yes", "confirm", "sure", "do it", "ok"])

                if is_confirmed:
                    if pending_action["type"] == "wipe_memory":
                        conversation_history = []
                        log("SYSTEM", "History cleared.")
                        enqueue_tts("Memory wiped.", global_turn_id, voice_id=system_voice)
                    elif pending_action["type"] == "delete_agent":
                        target = pending_action["name"]
                        msg = chat_room.delete_agent(target)
                        enqueue_tts(msg, global_turn_id, voice_id=system_voice)
                else:
                    enqueue_tts("Action cancelled.", global_turn_id, voice_id=system_voice)

                pending_action = None
                wait_for_turn()
                continue

            route = Orchestrator.route_request(user_text)

            full_response_text = ""
            def capture_generator(gen):
                nonlocal full_response_text
                for token in gen:
                    full_response_text += token
                    yield token

            if route == "SYSTEM":
                stream = capture_generator(Orchestrator.stream_admin_agent(user_text))
                smart_buffer_stream(stream, label="ADMIN", voice_id=system_voice)

                if interruption_event.is_set():
                    time.sleep(0.5)
                    continue

                tool_data = ToolManager.extract_lfm_call(full_response_text)
                if tool_data:
                    res = ToolManager.execute(tool_data)
                    if res and res.startswith("__ENTER_CHAT_ROOM__"):
                        payload = res.split(":", 1)[1]
                        parts = payload.split("|")
                        participants = parts[0]
                        topic = parts[1] if len(parts) > 1 and parts[1] else None

                        chat_room.run_session(participants, topic)
                        continue
                    elif res:
                        enqueue_tts(res, global_turn_id, voice_id=system_voice)
                        wait_for_turn()
            else:
                stream = capture_generator(Orchestrator.stream_chat_agent(user_text))
                smart_buffer_stream(stream, label="AI", voice_id=system_voice)

                if interruption_event.is_set():
                    time.sleep(0.5)
                    continue

                conversation_history.append({"role": "user", "content": user_text})
                conversation_history.append({"role": "assistant", "content": full_response_text.strip()})
                while len(conversation_history) > 10: del conversation_history[0:2]

                wait_for_turn()

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        cleanup_files()

if __name__ == "__main__":
    main()
