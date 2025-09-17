#!/usr/bin/env python3
"""
Quick Chinese Chess Game Demo
"""

from xiangqi_complete import XiangqiGame, format_move

def quick_demo():
    """Show a quick game demo"""
    print("🎮 Chinese Chess: RED (You) vs BLACK (AI)")
    print("=" * 50)

    game = XiangqiGame()

    print("STARTING POSITION:")
    game.display_board()

    # Simulate a few key moves
    moves = [
        ("e4-e5", "RED: Advance center soldier"),
        ("c7-c6", "BLACK: Advance soldier"),
        ("b3-e3", "RED: Move cannon to center"),
        ("b8-e8", "BLACK: Move cannon"),
        ("e3-e6", "RED: Cannon attacks!"),
    ]

    for i, (move_str, description) in enumerate(moves):
        print(f"\n{'='*50}")
        print(f"Move {i+1}: {description}")

        start, end = game.parse_move(move_str)
        if start and end:
            success = game.make_move(start, end)
            if success:
                print(f"✅ {move_str} - {description}")
                game.display_board()

                # Check game state
                if game.is_in_check(game.current_player):
                    color = game.current_player.upper()
                    print(f"⚠️  {color} is in CHECK!")

                if game.game_over:
                    winner = "RED (You)" if game.winner == 'red' else "BLACK (AI)"
                    print(f"🏆 GAME OVER! {winner} wins!")
                    break
            else:
                print(f"❌ Invalid move: {move_str}")

    print(f"\n{'='*50}")
    print("🎯 Demo completed!")
    print("This shows the authentic Chinese chess gameplay")
    print("🚀 To play the full game: python xiangqi_complete.py")

if __name__ == "__main__":
    quick_demo()