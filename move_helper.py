#!/usr/bin/env python3
"""
Move format helper for the professional xiangqi game
"""

def convert_move_format():
    """Help convert moves to the correct format"""
    print("🎯 XIANGQI MOVE FORMAT CONVERTER")
    print("=" * 50)
    print("The professional game uses format: h2e2 (no dash)")
    print("Your format was: c3-c4")
    print()

    # Common opening moves in correct format
    print("🚀 COMMON OPENING MOVES (correct format):")
    print("Instead of 'e4-e5' → use: e3e4")
    print("Instead of 'c4-c5' → use: c3c4")
    print("Instead of 'b3-e3' → use: b2e2")
    print("Instead of 'h3-e3' → use: h2e2")
    print("Instead of 'b1-c3' → use: b0c2")
    print()

    print("📍 COORDINATE SYSTEM:")
    print("Files: a b c d e f g h i (left to right)")
    print("Ranks: 0 1 2 3 4 5 6 7 8 9 (bottom to top for RED)")
    print()
    print("So the starting position:")
    print("- RED pieces are on ranks 0, 2, 3")
    print("- BLACK pieces are on ranks 6, 7, 9")
    print()

    print("🎯 FOR YOUR MOVE:")
    print("You played c3-c4 → try: c3c4")
    print("If that fails, try: c2c3")
    print()

    print("✅ VALID OPENING MOVES TO TRY:")
    moves = [
        "e3e4",  # Center soldier advance
        "c3c4",  # Left-center soldier
        "g3g4",  # Right-center soldier
        "b2e2",  # Cannon to center
        "h2e2",  # Right cannon to center
        "b0c2",  # Horse development
        "h0g2"   # Right horse development
    ]

    for i, move in enumerate(moves, 1):
        print(f"{i}. {move}")

if __name__ == "__main__":
    convert_move_format()