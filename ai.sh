#!/bin/bash

# --- Configuration ---

# 1. Paths for Termux (The "Host" - Where the files actually live)
LLAMA_BUILD_DIR="$HOME/llama.cpp/build/bin"
LLAMA_MODEL="/storage/emulated/0/Download/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
CLIENT_DIR="$HOME/superTalkLoop"

# 2. Paths for Ubuntu (The "Container")
# We will map CLIENT_DIR to this path inside Ubuntu
PROOT_PROJECT_DIR="/root/superTalkLoop"  
PROOT_VENV_ACTIVATE="/root/onnx/bin/activate"

# --- Cleanup Function ---
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $PID_LLAMA 2>/dev/null
    # Kill the python process inside proot
    pkill -f "python3 ttsServer.py"
    echo "Done."
    exit
}

trap cleanup SIGINT SIGTERM EXIT

echo "🚀 Starting AI System..."

# --- Step 1: Start TTS Server (in Ubuntu Proot with BIND MOUNT) ---
echo "🔹 Starting TTS Server (Ubuntu Proot)..."

# EXPLANATION OF CHANGE:
# --bind "$CLIENT_DIR:$PROOT_PROJECT_DIR"
# This tells Proot: "Take the folder $HOME/superTalkLoop from Termux 
# and make it appear at /root/superTalkLoop inside Ubuntu."

proot-distro login ubuntu \
    --bind "$CLIENT_DIR:$PROOT_PROJECT_DIR" \
    -- bash -c "source $PROOT_VENV_ACTIVATE && cd $PROOT_PROJECT_DIR && python3 ttsServer.py" &

PID_TTS=$!
# Give it a moment to spin up the python server
sleep 5

# --- Step 2: Start Llama Server (in Termux) ---
echo "🔹 Starting Llama Server (Termux)..."
if [ -d "$LLAMA_BUILD_DIR" ]; then
    "$LLAMA_BUILD_DIR/llama-server" -m "$LLAMA_MODEL" -t 4 -c 5000 &
    PID_LLAMA=$!
    sleep 10
else
    echo "❌ Error: Could not find llama-server directory at: $LLAMA_BUILD_DIR"
    exit 1
fi

# --- Step 3: Start Client Script (in Termux) ---
echo "🔹 Starting Client Script..."
cd "$CLIENT_DIR" || { echo "❌ Could not find client directory!"; exit 1; }

# Since we are in the same folder, the client can now easily 'see' ttsServer.py 
# if it needs to check if it exists, though they run in different environments.
python3 client.py
