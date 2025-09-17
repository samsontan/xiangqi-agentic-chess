#!/usr/bin/env python3
"""
Demo the fast AI playing Chinese chess
"""

from xiangqi_fast import FastXiangqiGame, format_move
import time

def demo_fast_ai():
    """Demonstrate the fast AI in action"""
    print("🎮 FAST AI DEMO - Watch the AI Play!")
    print("=" * 50)

    game = FastXiangqiGame()

    # Demo moves
    demo_moves = ["e4-e5", "c4-c5", "b3-e3"]

    for i, human_move in enumerate(demo_moves):
        print(f"\n{'='*60}")
        print(f"MOVE {i*2 + 1}: Human plays {human_move}")
        print("="*60)

        # Human move
        start, end = game.parse_move(human_move)
        if start and end:
            game.make_move(start, end)
            game.display_board()

        # AI move with timing
        if not game.game_over:
            print(f"\nMOVE {i*2 + 2}: AI thinking... (max 2 seconds)")
            start_time = time.time()

            ai_move = game.get_ai_move_fast(time_limit=2.0)
            think_time = time.time() - start_time

            if ai_move:
                start, end = ai_move
                game.make_move(start, end)
                move_str = format_move(start, end)
                print(f"🤖 AI plays: {move_str} (thought for {think_time:.1f}s)")
                game.display_board()
            else:
                print("AI couldn't find a move!")
                break

        time.sleep(1)  # Pause for readability

    print(f"\n{'='*60}")
    print("✅ DEMO COMPLETED!")
    print("The AI responds quickly (under 2 seconds)")
    print("Your fast game is ready to play!")
    print("="*60)

if __name__ == "__main__":
    demo_fast_ai()