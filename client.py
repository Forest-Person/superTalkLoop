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

# --- Logging Helper ---
def log(subsystem: str, message: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] [{subsystem}] {message}")

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

global_turn_id = 0
conversation_history = []
current_persona_prompt = "You are SuperTonic, a helpful and precise voice assistant."
FINAL_AUDIO = "whisper_ready.wav"
pending_action = None

# --- CLASSES ---

class ChatAgent:
    def __init__(self, name, system_prompt, voice_id="M1"):
        self.name = name
        self.base_prompt = system_prompt
        self.voice_id = voice_id
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
                    "temperature": 0.7,
                    "max_tokens": 200,
                    "stream": True,
                    "stop": stop_sequences
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

        if topic:
            direction_prompt = f"The user wants the agents to discuss: '{topic}'. Pick the best agent to start this topic."
        else:
            direction_prompt = (
                "No topic provided. Analyze the 'Description' of the agents.\n"
                "1. Identify a specific interest, conflict, or personality quirk from one agent.\n"
                "2. Have that agent start a conversation about that interest/quirk."
            )

        system_prompt = (
            f"You are a Stage Manager starting a scene.\n"
            f"Active Characters:\n{profiles}\n"
            f"Direction: {direction_prompt}\n"
            "INSTRUCTIONS:\n"
            "1. Select the character who should speak first.\n"
            "2. Provide a 1-sentence instruction on WHAT they should say to kick off the chat.\n"
            "Output JSON ONLY: {\"next_agent\": \"Name\", \"instruction\": \"...\"}"
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
        except Exception as e:
            log("DIRECTOR", f"Kickoff Error: {e}")

        target = random.choice(active_agents)
        return target, f"Start a conversation about {topic if topic else 'something interesting'}."

    def select_next_speaker(self, active_agents: List[ChatAgent], last_speaker: str, last_message: str):
        if not active_agents: return None, "Speak naturally."
        if len(active_agents) == 1: return active_agents[0], "Respond to the user."

        agent_map = {a.name: a for a in active_agents}
        profiles = "\n".join([f"- {a.name}: {a.base_prompt}" for a in active_agents])

        system_prompt = (
            f"You are a Stage Manager. Select the next speaker.\n"
            f"Active Characters:\n{profiles}\n"
            f"Last Speaker: {last_speaker}\n"
            f"Last Message: \"{last_message[:200]}\"\n"
            "INSTRUCTIONS:\n"
            "1. Choose the character who should respond next.\n"
            "2. Provide a 1-sentence instruction on how to respond.\n"
            "Output JSON ONLY: {\"next_agent\": \"Name\", \"instruction\": \"...\"}"
        )
        try:
            resp = self.session.post(
                self.url,
                json={
                    "messages": [{"role": "system", "content": system_prompt}],
                    "temperature": 0.1, "max_tokens": 60
                }, timeout=5
            )
            if resp.status_code == 200:
                content = resp.json()['choices'][0]['message']['content']
                content = content.replace("```json", "").replace("```", "").strip()
                data = json.loads(content)
                name = data.get("next_agent")
                instruction = data.get("instruction", "Respond naturally.")
                if name in agent_map:
                    return agent_map[name], instruction
        except Exception:
            pass

        available = [a for a in active_agents if a.name != last_speaker]
        target = random.choice(available if available else active_agents)
        return target, f"Respond to {last_speaker}."

class ChatRoom:
    def __init__(self):
        self.agents: Dict[str, ChatAgent] = {}
        self.director = AgentDirector(http_session, LLAMA_SERVER_URL)
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
        
        self.agents[name] = ChatAgent(name, prompt, voice_id=voice)
        log("SYSTEM", f"Created Agent: {name} (Voice: {voice})")
        return f"Created {name} with voice style {voice}."

    def delete_agent(self, name):
        target = None
        for k in self.agents.keys():
            if name.lower() in k.lower():
                target = k
                break
        if target:
            del self.agents[target]
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
        
        used_voices = {agent.voice_id for agent in self.agents.values()}
        if self.system_voice:
            used_voices.add(self.system_voice)

        available = [v for v in self.all_voices if v not in used_voices]
        random.shuffle(available)

        for i, (name, prompt) in enumerate(defaults):
            if len(self.agents) < 4 and available:
                voice = available[i % len(available)]
                self.agents[name] = ChatAgent(name, prompt, voice_id=voice)

    def run_session(self, participants_str=None, topic=None):
        log("SYSTEM", f"Entering Room. Participants: {participants_str}, Topic: {topic}")
        
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

        if starter_agent:
            print(f"[{starter_agent.name}] STARTING: {start_instruction}")
            kickoff_context = [{"role": "system", "content": "The scene is beginning."}]
            other_names = [a.name for a in active_agents if a.name != starter_agent.name]
            other_names.append("User")

            stream_gen = starter_agent.respond_stream(
                kickoff_context,
                director_instruction=start_instruction,
                other_agent_names=other_names
            )
            smart_buffer_stream(stream_gen, starter_agent.name, voice_id=starter_agent.voice_id)

            if starter_agent.history:
                last_message_content = starter_agent.history[-1]['content']
                last_speaker_name = starter_agent.name

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
                print(f"\n[User]: {user_text}")
                last_message_content = user_text
                last_speaker_name = "User"
            elif not active_agents:
                continue

            if not user_text: log("DIRECTOR", "Autonomy Triggered...")

            target_agent, instruction = self.director.select_next_speaker(active_agents, last_speaker_name, last_message_content)
            if not target_agent: continue

            if interruption_event.is_set(): continue
            print(f"[{target_agent.name}] Dir: {instruction}")

            context_input = f"{last_speaker_name}: {last_message_content}"
            other_names = [a.name for a in active_agents if a.name != target_agent.name]
            other_names.append("User")

            stream_gen = target_agent.respond_stream(
                [{"role": "user", "content": context_input}],
                director_instruction=instruction,
                other_agent_names=other_names
            )

            smart_buffer_stream(stream_gen, target_agent.name, voice_id=target_agent.voice_id)

            if target_agent.history:
                last_message_content = target_agent.history[-1]['content']
            else:
                last_message_content = "..."

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
    if not interruption_event.is_set():
        print(f"\n[!] 🛑 Interrupted by {source}")
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
    global global_turn_id
    ai_is_speaking_event.clear()
    time.sleep(0.2)
    vad = webrtcvad.Vad(VAD_RECORD_SENSITIVITY)
    start_time = time.time()

    label = "Listening" if timeout is None else f"Auto-Listen ({timeout}s)"
    print(f"\n[Mic] 🎤 {label}...", end="", flush=True)

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
                        print(" (Timeout)")
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
                        print("●", end="", flush=True)
                else:
                    voiced_frames.append(frame_bytes)
                    if not is_speech:
                        silence_counter += 1
                        if silence_counter >= (SILENCE_THRESHOLD_MS // FRAME_MS):
                            print(" Done.")
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

def smart_buffer_stream(token_generator, label="AI", voice_id="M1"):
    sentence_buffer = ""
    is_first_chunk = True
    FIRST_CHUNK_WORDS = 8
    SUBSEQ_CHUNK_WORDS = 25

    print(f"{label}: ", end="")

    for token in token_generator:
        if interruption_event.is_set():
            print(" [Interrupted]")
            break

        sys.stdout.write(token)
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
