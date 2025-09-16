#!/bin/bash

# Xiangqi Chinese Chess Game Launcher
# Launch the fixed Xiangqi game with proper validation

echo "🎮 Launching Chinese Chess (Xiangqi)..."
echo "🔴 You play as RED vs ⚫ AI BLACK"

# Change to home directory where game files are located
cd /data/data/com.termux/files/home

# Check if the game file exists
if [ ! -f "xiangqi_fixed.py" ]; then
    echo "❌ Game file not found! Make sure xiangqi_fixed.py is in your home directory."
    exit 1
fi

# Launch the game
echo "🚀 Starting game..."
python xiangqi_fixed.py