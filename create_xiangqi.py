#!/usr/bin/env python3
"""
Direct Chinese Chess Game Creation using Single Agent
"""

import os
from local_agent_swarm import OpenAIProvider, SmartAgent

def create_xiangqi_game():
    """Create complete Chinese chess game in one go"""
    print("🎮 Creating Chinese Chess (Xiangqi) Game with AI")
    print("=" * 50)

    provider = OpenAIProvider(model="gpt-4o-mini")
    game_dev = SmartAgent("XiangqiDeveloper", provider, "expert Chinese chess game developer")

    prompt = """
    Create a complete, playable Chinese Chess (Xiangqi) game in Python. Requirements:

    1. **Complete Game**: Single Python file that runs immediately
    2. **Board**: 9x10 board with proper palace and river
    3. **All Pieces**: General, Advisor, Elephant, Horse, Chariot, Cannon, Soldier
    4. **Correct Rules**: Palace restrictions, river crossing, flying general
    5. **AI Opponent**: Minimax algorithm with position evaluation
    6. **Text Interface**: Clear display, coordinate input (like a1, b2)
    7. **Game Flow**: Human vs AI, turn management, move validation
    8. **Win Detection**: Checkmate and stalemate

    Make it production-ready with:
    - Clear piece symbols (♔♕♖♗♘♙ for red, ♚♛♜♝♞♟ for black)
    - Intuitive coordinate system
    - Move format examples
    - Game state display
    - Error handling

    The game should be immediately playable after running the file.
    """

    print("🤖 Agent creating complete game...")
    result = game_dev.think(prompt)

    return result

def main():
    """Create and save the game"""
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not set!")
        return

    try:
        # Generate the game
        game_code = create_xiangqi_game()

        # Extract Python code
        lines = game_code.split('\n')
        in_code_block = False
        extracted_lines = []

        for line in lines:
            if line.strip().startswith('```python'):
                in_code_block = True
                continue
            elif line.strip() == '```' and in_code_block:
                in_code_block = False
                continue
            elif in_code_block:
                extracted_lines.append(line)

        # Save the game
        if extracted_lines:
            with open('/data/data/com.termux/files/home/xiangqi_game.py', 'w') as f:
                f.write('\n'.join(extracted_lines))

            print("✅ Chinese Chess game created successfully!")
            print("📁 Saved as: xiangqi_game.py")
            print("🚀 Run with: python xiangqi_game.py")

            # Show preview
            print(f"\n📝 Code preview:")
            print('\n'.join(extracted_lines[:30]) + "\n...")

        else:
            print("❌ No code blocks found in response")
            print("📝 Raw response:")
            print(game_code[:1000] + "...")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()