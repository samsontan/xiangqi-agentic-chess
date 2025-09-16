#!/usr/bin/env python3
"""
Debug the AI move issue
"""

from xiangqi_fast import FastXiangqiGame

def debug_ai_move():
    print("🔍 DEBUGGING AI MOVE VALIDATION")
    print("=" * 50)

    game = FastXiangqiGame()

    print("Initial board state:")
    print("BLACK pieces at top:")
    for row in range(3):
        for col in range(9):
            piece = game.board[row][col]
            if piece:
                files = 'abcdefghi'
                rank = 10 - row
                print(f"  {files[col]}{rank}: {piece[1]} ({piece[0]})")

    print("\nChecking move b10-a10:")
    start = (0, 1)  # b10 in array coordinates
    end = (0, 0)    # a10 in array coordinates

    piece_at_b10 = game.board[start[0]][start[1]]
    print(f"Piece at b10: {piece_at_b10}")

    if piece_at_b10:
        print(f"Piece type: {piece_at_b10[1]}")
        print(f"Piece color: {piece_at_b10[0]}")

    print(f"Is this move valid? {game.is_valid_move(start, end)}")

    print("\nThis move should be INVALID because:")
    print("- Horse at b10 should move in L-shape only")
    print("- b10-a10 is horizontal movement")
    print("- Horse cannot move horizontally")

if __name__ == "__main__":
    debug_ai_move()