# SuperTalkLoop: Local Conversational AI System

This project is a sophisticated, privacy-focused conversational AI system designed to run locally on Android devices (via Termux). It integrates Speech-to-Text (STT), Large Language Models (LLM), and Text-to-Speech (TTS) into a seamless voice assistant experience.

## 🚀 Overview

SuperTalkLoop orchestrates three main components to create a fluid conversation loop:
1.  **Hearing:** Records user audio and converts it to text using `whisper.cpp`.
2.  **Thinking:** Processes text and generates responses using a local Llama model (`llama.cpp`).
3.  **Speaking:** Converts the AI's response to high-quality speech using a custom Python TTS server (`supertonic`).

The system supports **multi-agent conversations**, allowing you to create and interact with different AI personas in a virtual "Chat Room."

## 🏗 System Architecture

The system uses a hybrid environment approach to leverage the best tools available on Android/Termux:

*   **Host Environment (Termux):**
    *   Runs the orchestration script (`ai.sh`).
    *   Runs the **Client** (`client.py`) which handles audio I/O, logic, and API calls.
    *   Runs the **LLM Server** (`llama-server`) for text generation.
    *   Runs `whisper-cli` for Speech-to-Text.
    *   *Why?* Termux provides direct access to hardware (Mic/Speaker) and runs compiled C++ binaries efficiently.

*   **Container Environment (Ubuntu via Proot-Distro):**
    *   Runs the **TTS Server** (`ttsServer.py`).
    *   *Why?* The TTS engine (`supertonic`) likely requires complex Python dependencies (like PyTorch/ONNX) that are easier to manage in a standard Linux environment than in bare Termux.

### Data Flow Diagram

```mermaid
graph TD
    User((User)) <-->|Audio| Client[client.py (Termux)]
    
    subgraph Termux [Termux Host]
        Client -->|Subprocess| Whisper[Whisper CLI]
        Client <-->|HTTP :8080| Llama[Llama Server]
    end
    
    subgraph Proot [Ubuntu Container]
        Client <-->|HTTP :5002| TTS[TTS Server (ttsServer.py)]
    end
    
    Whisper -->|Text| Client
    Llama -->|Text Stream| Client
    TTS -->|WAV Audio| Client
```

## ✨ New Features

### 🎮 Startup Menu
When you launch the system, you are presented with three modes:
*   **[C]hat Mode:** The standard voice-controlled experience.
*   **[A]gent Creation Mode:** Manual text input to define names and descriptions for up to **10 custom agents**.
*   **[M]eet the Moms:** A "Quick Start" preset that loads three suburban moms (Karen, Linda, Susan) into a high-drama rivalry over a bake sale.

### 🤖 Autonomous "Auto-Loop" Mode
You can now step back and let the AI agents converse entirely on their own.
*   **Toggle:** Press **Enter** while in a chat room to toggle "Non-Interactive Mode."
*   **Behavior:** In this mode, agents take initiative automatically. Microphone interruptions are disabled to prevent the AI from interrupting itself with its own echo.
*   **Observation:** The UI is cleanly color-coded so you can watch the drama unfold.

### 🎭 Enhanced AI Director
The "Director" has been upgraded with sophisticated social logic:
*   **Strict Turn-Taking:** Prevents the same agent from speaking twice in a row.
*   **Silence Tracking:** Monitors how long each agent has been quiet and proactively encourages under-participating agents to speak.
*   **Dynamic Styles & Goals:** The Director assigns a specific **Tone** (e.g., "passive-aggressive") and **Objective** (e.g., "subtly insult the cookies") to each turn, ensuring spicy and varied dialogue.

### 🎨 Clean UI & Observability
*   **Color-Coded Text:** Each agent has a unique color. User input is Cyan, and Director instructions are Dimmed.
*   **Quiet Servers:** Verbose logs from `llama.cpp` and background TTS/Audio systems are silenced, leaving only the "neat" conversation on your screen.

## 📂 File Descriptions

### 1. `ai.sh` (The Orchestrator)
**Role:** Startup Script & Process Manager.
*   **Function:**
    1.  Maps the project directory from Termux to the Ubuntu container using `--bind`.
    2.  Starts `ttsServer.py` inside the Ubuntu Proot environment (background process).
    3.  Starts `llama-server` in Termux (background process).
    4.  Launches `client.py` in the foreground.
    5.  Handles cleanup: Kills all background servers when you exit the client.
*   **Key Configuration:** Checks for model paths and virtual environments before starting.

### 2. `client.py` (The Brain)
**Role:** Main User Interface & Logic.
*   **Function:**
    *   **VAD (Voice Activity Detection):** Listens for speech and automatically cuts off recording when silence is detected.
    *   **Interruption Handling:** Allows you to interrupt the AI while it's speaking.
    *   **Orchestrator:** Routes requests to either "Chat Mode" or "System Mode" (for tools like `create_agent`, `wipe_memory`).
    *   **Multi-Agent Manager:** Can simulate a room with multiple AI characters talking to each other.
*   **Dependencies:** `sounddevice`, `numpy`, `requests`, `webrtcvad`.

### 3. `ttsServer.py` (The Voice)
**Role:** Text-to-Speech API.
*   **Function:**
    *   Hosted as a Flask web server on port `5002`.
    *   Accepts JSON payloads (`{"text": "...", "voice": "M1"}`).
    *   Returns WAV audio files generated by the `supertonic` library.
    *   Pre-loads voice styles (M1-M5, F1-F5) for low-latency response.

## 🛠 Prerequisites

### Termux (Host)
*   **Packages:** `python`, `git`, `proot-distro`, `build-essential`.
*   **Binaries:**
    *   `llama-server` (from `llama.cpp`) built and located at `~/llama.cpp/build/bin/`.
    *   `whisper-cli` (from `whisper.cpp`) built and located at `~/whisper.cpp/build/bin/`.
*   **Models:**
    *   Llama Model (e.g., `LFM2.5-1.2B-Instruct-Q4_K_M.gguf`) in `/storage/emulated/0/Download/`.
    *   Whisper Model (e.g., `ggml-tiny.en.bin`) in `~/whisper.cpp/models/`.

### Ubuntu (Proot)
*   **Environment:** An Ubuntu installation via `proot-distro install ubuntu`.
*   **Python:** Python 3 installed.
*   **Virtual Environment:** A venv located at `/root/onnx` containing `flask` and `supertonic`.

## ⚙️ Configuration

Ensure the paths in `ai.sh` match your setup:

```bash
# In ai.sh
LLAMA_BUILD_DIR="$HOME/llama.cpp/build/bin"
LLAMA_MODEL="/storage/emulated/0/Download/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
PROOT_VENV_ACTIVATE="/root/onnx/bin/activate"
```

Ensure the paths in `client.py` match your setup:

```python
# In client.py
MODEL_PATH = os.path.expanduser("~/whisper.cpp/models/ggml-tiny.en.bin")
WHISPER_EXECUTABLE = os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli")
```

## ▶️ Usage

1.  Open Termux.
2.  Navigate to the project directory:
    ```bash
    cd ~/superTalkLoop
    ```
3.  Run the orchestration script:
    ```bash
    ./ai.sh
    ```
4.  **Select a Mode:**
    *   Press **'c'** for the default voice assistant.
    *   Press **'a'** to manually create your own cast of characters.
    *   Press **'m'** to instantly start the "Suburban Moms" drama.
5.  **In the Chat Room:**
    *   **Auto-Loop:** Press **Enter** to toggle "Non-Interactive Mode" ON/OFF.
    *   **Manual Speak:** Just talk normally to participate.
    *   **Exit:** Say **"Exit"** or use `Ctrl+C` to quit.

## 🧩 Troubleshooting

*   **"Missing Dependency":** Check if you are running `client.py` in the correct environment (Termux vs Proot).
*   **Audio Issues:** Ensure Termux has Microphone permission in Android settings.
*   **Server Connection Refused:**
    *   Check if `llama-server` started correctly (check paths in `ai.sh`).
    *   Check if `ttsServer.py` is running (you can manually log into proot and run it to debug).
