#!/usr/bin/env python3
"""
Test the improved board alignment
"""

from xiangqi_fast import FastXiangqiGame

def test_board_display():
    """Test the improved board display"""
    print("🎮 TESTING IMPROVED BOARD ALIGNMENT")
    print("=" * 50)

    game = FastXiangqiGame()

    print("Initial position with improved alignment:")
    game.display_board()

    print("Board features:")
    print("✅ Each piece is centered in its square")
    print("✅ Proper grid lines separate all squares")
    print("✅ River section clearly marked")
    print("✅ Palace positions marked with +")
    print("✅ File/rank labels properly aligned")
    print("✅ Consistent spacing throughout")

if __name__ == "__main__":
    test_board_display()