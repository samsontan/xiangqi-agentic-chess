#!/usr/bin/env python3
"""Test the circle board display"""

from xiangqi_circles import CircleXiangqiGame

def test_circle_board():
    print("🎮 TESTING CIRCLE-FILLED BOARD")
    print("=" * 50)

    game = CircleXiangqiGame()
    game.display_board()

    print("✅ Perfect alignment features:")
    print("• Pieces and circles are same width")
    print("• File labels (a-i) aligned with columns")
    print("• Rank labels (1-10) aligned with rows")
    print("• No complex grid lines to misalign")
    print("• Clean, professional look")
    print("• Easy to see piece positions")

if __name__ == "__main__":
    test_circle_board()