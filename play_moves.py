#!/usr/bin/env python3
"""
Interactive Xiangqi Move Helper
"""

from xiangqi_complete import XiangqiGame

def suggest_good_opening_moves():
    """Suggest good opening moves for RED"""
    moves = [
        ("e4-e5", "Advance center soldier - Classic opening"),
        ("c4-c5", "Advance left-center soldier"),
        ("g4-g5", "Advance right-center soldier"),
        ("b3-e3", "Move cannon to center file"),
        ("h3-e3", "Move right cannon to center"),
        ("b1-c3", "Develop horse"),
        ("h1-g3", "Develop right horse")
    ]

    print("🎯 SUGGESTED OPENING MOVES FOR RED:")
    print("=" * 50)
    for i, (move, description) in enumerate(moves, 1):
        print(f"{i}. {move} - {description}")

    return moves

def analyze_current_position():
    """Show current position analysis"""
    game = XiangqiGame()

    print("\n📊 CURRENT POSITION ANALYSIS:")
    print("=" * 50)
    print("🔴 RED (You) pieces:")
    print("• 帥 General at e1 (protected in palace)")
    print("• 仕 Advisors at d1, f1 (guarding general)")
    print("• 相 Elephants at c1, g1 (defensive)")
    print("• 馬 Horses at b1, h1 (ready to develop)")
    print("• 車 Chariots at a1, i1 (powerful pieces)")
    print("• 炮 Cannons at b3, h3 (need platform to attack)")
    print("• 兵 Soldiers at a4, c4, e4, g4, i4 (ready to advance)")

    print("\n⚫ BLACK (AI) pieces:")
    print("• 將 General at e10 (in palace)")
    print("• 士 Advisors at d10, f10")
    print("• 象 Elephants at c10, g10")
    print("• 馬 Horses at b10, h10")
    print("• 車 Chariots at a10, i10")
    print("• 炮 Cannons at b8, h8")
    print("• 卒 Soldiers at a7, c7, e7, g7, i7")

def explain_move_strategy():
    """Explain Xiangqi strategy"""
    print("\n🧠 XIANGQI STRATEGY TIPS:")
    print("=" * 50)
    print("1. 🎯 OPENING PRINCIPLES:")
    print("   • Advance center soldiers first (e4-e5)")
    print("   • Develop cannons to center files")
    print("   • Move horses to active squares")
    print("   • Don't move same piece twice early")

    print("\n2. 🏰 PIECE VALUES:")
    print("   • Chariot (車): 9 points - Most powerful")
    print("   • Cannon (炮): 4.5 points - Jump to capture")
    print("   • Horse (馬): 4 points - Can be blocked")
    print("   • Elephant (相/象): 2 points - Defense only")
    print("   • Advisor (仕/士): 2 points - Palace only")
    print("   • Soldier (兵/卒): 1 point - Gains power after river")

    print("\n3. 🎲 TACTICAL PATTERNS:")
    print("   • River crossing: Soldiers become more mobile")
    print("   • Cannon battery: Two cannons on same file")
    print("   • Horse pins: Block opponent's horse")
    print("   • Palace tactics: Attack general through advisors")

def recommend_first_move():
    """Recommend the first move"""
    print("\n🚀 RECOMMENDED FIRST MOVE:")
    print("=" * 50)
    print("Best move: e4-e5")
    print("Why this move is excellent:")
    print("✅ Advances the center soldier")
    print("✅ Crosses the river (gains sideways movement)")
    print("✅ Controls center of board")
    print("✅ Prepares for cannon development")
    print("✅ Classical opening move")

    print("\nAfter e4-e5, good follow-ups:")
    print("• b3-e3 (cannon to center)")
    print("• b1-c3 (develop horse)")
    print("• h3-e3 (if AI allows, double cannons)")

def show_move_format():
    """Show how to input moves"""
    print("\n📝 HOW TO ENTER MOVES:")
    print("=" * 50)
    print("Format: from-to (like e4-e5)")
    print("Files: a b c d e f g h i (left to right)")
    print("Ranks: 1-10 (bottom to top for RED)")

    print("\nExamples:")
    print("• e4-e5: Move soldier from e4 to e5")
    print("• b3-e3: Move cannon from b3 to e3")
    print("• b1-c3: Move horse from b1 to c3")
    print("• quit: Exit the game")

def main():
    """Main helper function"""
    print("🎮 XIANGQI MOVE HELPER")
    print("=" * 60)

    analyze_current_position()
    suggest_good_opening_moves()
    explain_move_strategy()
    recommend_first_move()
    show_move_format()

    print("\n" + "=" * 60)
    print("🎯 YOU'RE READY TO PLAY!")
    print("Your game is waiting for your move.")
    print("I recommend starting with: e4-e5")
    print("=" * 60)

if __name__ == "__main__":
    main()