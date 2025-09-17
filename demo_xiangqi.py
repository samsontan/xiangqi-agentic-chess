#!/usr/bin/env python3
"""
Demo the Chinese Chess game with a few moves
"""

# Import the complete game
from xiangqi_complete import XiangqiGame, format_move

def demo_game():
    """Demonstrate the game with some moves"""
    print("🎮 Chinese Chess Game Demo")
    print("=" * 40)

    game = XiangqiGame()

    # Show initial position
    print("Initial position:")
    game.display_board()

    # Make some demo moves
    demo_moves = [
        ("e4-e5", "Red soldier advances"),
        ("AI move", "AI responds"),
        ("b3-e3", "Red cannon moves"),
        ("AI move", "AI responds")
    ]

    for i, (move_desc, description) in enumerate(demo_moves):
        print(f"\nMove {i+1}: {description}")

        if move_desc == "AI move":
            if game.current_player == 'black':
                ai_move = game.get_ai_move()
                if ai_move:
                    start, end = ai_move
                    game.make_move(start, end)
                    move_str = format_move(start, end)
                    print(f"🤖 AI plays: {move_str}")
        else:
            start, end = game.parse_move(move_desc)
            if start and end and game.is_valid_move(start, end):
                game.make_move(start, end)
                print(f"✅ Move: {move_desc}")
            else:
                print(f"❌ Invalid move: {move_desc}")

        game.display_board()

        if game.game_over:
            print("Game ended!")
            break

    print("\n🎉 Demo completed!")
    print("🚀 To play the full game, run: python xiangqi_complete.py")

if __name__ == "__main__":
    demo_game()