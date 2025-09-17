#!/usr/bin/env python3
"""
Create a proper Chinese Chess (Xiangqi) game with correct rules and AI
"""

import os
from local_agent_swarm import OpenAIProvider, SmartAgent

def create_proper_xiangqi():
    """Create a complete Chinese chess game with proper rules"""
    print("🎮 Creating Proper Chinese Chess (Xiangqi) Game")
    print("=" * 50)

    provider = OpenAIProvider(model="gpt-4o-mini")
    xiangqi_expert = SmartAgent("XiangqiExpert", provider, "Chinese chess master and Python expert")

    prompt = """
    Create a complete Chinese Chess (Xiangqi) game in Python with authentic rules. Requirements:

    BOARD SETUP:
    - 10x9 board (10 ranks, 9 files)
    - Palace: 3x3 area at each end for General and Advisors
    - River: divides board between ranks 5 and 6

    PIECES (with proper Chinese names and movement):
    1. General (帥/將): Moves 1 point within palace, cannot face opponent's general
    2. Advisor (仕/士): Moves diagonally 1 point within palace
    3. Elephant (相/象): Moves 2 points diagonally, cannot cross river, cannot jump
    4. Horse (馬): Moves like chess knight but can be blocked
    5. Chariot (車): Moves like chess rook
    6. Cannon (炮): Moves like rook but must jump to capture
    7. Soldier (兵/卒): Moves forward, sideways only after crossing river

    IMPLEMENTATION:
    - Text-based interface with clear Unicode symbols
    - Coordinate system: a1-i10 (files a-i, ranks 1-10)
    - Input validation and move legality checking
    - Basic AI using minimax (depth 3) with position evaluation
    - Game state management (turn, check, checkmate)

    Make it a single, complete, immediately playable Python file.
    Use proper Chinese chess symbols and include move examples in the interface.
    """

    print("🤖 Creating authentic Xiangqi game...")
    result = xiangqi_expert.think(prompt)

    return result

def main():
    """Generate and save the proper game"""
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not set!")
        return

    try:
        # Generate the game
        game_code = create_proper_xiangqi()

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

        # Save the improved game
        if extracted_lines:
            with open('/data/data/com.termux/files/home/xiangqi_proper.py', 'w') as f:
                f.write('\n'.join(extracted_lines))

            print("✅ Proper Chinese Chess game created!")
            print("📁 Saved as: xiangqi_proper.py")
            print("🚀 Run with: python xiangqi_proper.py")

            # Show preview
            print(f"\n📝 Game preview (first 40 lines):")
            for i, line in enumerate(extracted_lines[:40]):
                print(f"{i+1:2}: {line}")

        else:
            # No code block found, save the raw response
            print("⚠️  No code blocks found, saving raw response")
            with open('/data/data/com.termux/files/home/xiangqi_raw.txt', 'w') as f:
                f.write(game_code)
            print("📝 Raw response saved to: xiangqi_raw.txt")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()