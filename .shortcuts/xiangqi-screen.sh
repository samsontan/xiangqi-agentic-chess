#!/bin/bash

# Xiangqi Game Launcher with Screen Session
# Launch the game in a detached screen session

echo "🎮 Starting Chinese Chess in background screen..."

# Change to home directory
cd /data/data/com.termux/files/home

# Kill any existing xiangqi screen sessions
screen -S xiangqi -X quit 2>/dev/null

# Start new screen session with the game
screen -dmS xiangqi bash -c "python xiangqi_fixed.py"

# Wait a moment for the session to start
sleep 1

# Connect to the screen session
echo "🔗 Connecting to Xiangqi game..."
screen -r xiangqi