#!/usr/bin/env python3
"""
Simple board display for Chinese chess
"""

def show_simple_board():
    """Show a simple representation of the Chinese chess starting position"""

    print("🎮 CHINESE CHESS (XIANGQI) STARTING POSITION")
    print("=" * 60)
    print()
    print("BLACK (AI) - Top")
    print("┌─────────────────────────────────────────┐")
    print("│ 車 馬 象 士 將 士 象 馬 車  │ Rank 10")
    print("│ ·  ·  ·  ·  ·  ·  ·  ·  ·  │ Rank 9")
    print("│ ·  炮 ·  ·  ·  ·  ·  炮 ·  │ Rank 8")
    print("│ 卒 ·  卒 ·  卒 ·  卒 ·  卒 │ Rank 7")
    print("│ ~  ~  ~  ~  ~  ~  ~  ~  ~  │ Rank 6")
    print("├─────── RIVER ───────────────┤")
    print("│ ~  ~  ~  ~  ~  ~  ~  ~  ~  │ Rank 5")
    print("│ 兵 ·  兵 ·  兵 ·  兵 ·  兵 │ Rank 4")
    print("│ ·  炮 ·  ·  ·  ·  ·  炮 ·  │ Rank 3")
    print("│ ·  ·  ·  ·  ·  ·  ·  ·  ·  │ Rank 2")
    print("│ 車 馬 相 仕 帥 仕 相 馬 車  │ Rank 1")
    print("└─────────────────────────────────────────┘")
    print("RED (You) - Bottom")
    print()
    print("Files: a  b  c  d  e  f  g  h  i")
    print()

    print("🎯 PIECE MEANINGS:")
    print("BLACK pieces (AI):")
    print("  將 = General   士 = Advisor   象 = Elephant")
    print("  馬 = Horse     車 = Chariot   炮 = Cannon")
    print("  卒 = Soldier")
    print()
    print("RED pieces (You):")
    print("  帥 = General   仕 = Advisor   相 = Elephant")
    print("  馬 = Horse     車 = Chariot   炮 = Cannon")
    print("  兵 = Soldier")
    print()

    print("🎮 HOW TO PLAY:")
    print("• You are RED pieces (bottom)")
    print("• AI is BLACK pieces (top)")
    print("• Move format: e4-e5 (from e4 to e5)")
    print("• Example first move: e4-e5 (advance center soldier)")
    print()

    print("🚀 TO START PLAYING:")
    print("Run this command in your Termux terminal:")
    print("   python xiangqi_complete.py")
    print()

def show_game_example():
    """Show what a move looks like"""
    print("📝 EXAMPLE MOVE:")
    print("After RED plays e4-e5:")
    print()
    print("┌─────────────────────────────────────────┐")
    print("│ 車 馬 象 士 將 士 象 馬 車  │ Rank 10")
    print("│ ·  ·  ·  ·  ·  ·  ·  ·  ·  │ Rank 9")
    print("│ ·  炮 ·  ·  ·  ·  ·  炮 ·  │ Rank 8")
    print("│ 卒 ·  卒 ·  卒 ·  卒 ·  卒 │ Rank 7")
    print("│ ~  ~  ~  ~  ~  ~  ~  ~  ~  │ Rank 6")
    print("├─────── RIVER ───────────────┤")
    print("│ ~  ~  ~  ~ 兵  ~  ~  ~  ~  │ Rank 5 ← Soldier moved here!")
    print("│ 兵 ·  兵 ·  ·  ·  兵 ·  兵 │ Rank 4")
    print("│ ·  炮 ·  ·  ·  ·  ·  炮 ·  │ Rank 3")
    print("│ ·  ·  ·  ·  ·  ·  ·  ·  ·  │ Rank 2")
    print("│ 車 馬 相 仕 帥 仕 相 馬 車  │ Rank 1")
    print("└─────────────────────────────────────────┘")
    print("The center soldier crossed the river!")
    print("Now it can move sideways too!")

if __name__ == "__main__":
    show_simple_board()
    show_game_example()

    print("=" * 60)
    print("🎉 Your Chinese Chess game is ready!")
    print("🎮 Built using agentic swarm coding!")
    print("=" * 60)