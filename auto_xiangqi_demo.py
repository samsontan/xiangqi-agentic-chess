#!/usr/bin/env python3
"""
Automated Chinese Chess Demo - Playing as RED against AI
"""

import time
from xiangqi_complete import XiangqiGame, format_move

def play_demo_game():
    """Play an automated demo game as RED against AI"""
    print("🎮 Chinese Chess Demo: RED (You) vs BLACK (AI)")
    print("=" * 60)
    print("🔴 Playing as RED against the AI...")
    print("=" * 60)

    game = XiangqiGame()

    # Predefined opening moves for demonstration
    red_demo_moves = [
        "e4-e5",  # Advance center soldier
        "b3-e3",  # Move cannon to center
        "a4-a5",  # Advance left soldier
        "e3-e7",  # Cannon attacks
        "h3-h4",  # Move right cannon
        "h4-h7",  # Attack with right cannon
        "i4-i5",  # Advance right soldier
    ]

    move_count = 0
    red_move_index = 0

    while not game.game_over and move_count < 15:  # Limit to 15 total moves
        print(f"\n{'='*60}")
        print(f"Move {move_count + 1}")
        game.display_board()

        if game.current_player == 'red':
            # RED (Human representative) move
            print("🔴 RED's turn:")
            if game.is_in_check('red'):
                print("⚠️  RED is in CHECK!")

            # Use predefined moves or let AI choose
            if red_move_index < len(red_demo_moves):
                move_str = red_demo_moves[red_move_index]
                red_move_index += 1
            else:
                # Get a random valid move for red
                valid_moves = game.get_all_valid_moves('red')
                if valid_moves:
                    start, end = valid_moves[0]  # Take first valid move
                    move_str = format_move(start, end)
                else:
                    print("❌ No valid moves for RED!")
                    break

            print(f"🔴 RED plays: {move_str}")

            start, end = game.parse_move(move_str)
            if start and end and game.is_valid_move(start, end):
                game.make_move(start, end)
                print("✅ Move successful!")
            else:
                print(f"❌ Invalid move: {move_str}")
                # Try to get any valid move
                valid_moves = game.get_all_valid_moves('red')
                if valid_moves:
                    start, end = valid_moves[0]
                    game.make_move(start, end)
                    move_str = format_move(start, end)
                    print(f"🔴 RED plays alternative: {move_str}")

        else:
            # BLACK (AI) move
            print("⚫ BLACK's turn (AI thinking...):")
            if game.is_in_check('black'):
                print("⚠️  BLACK is in CHECK!")

            time.sleep(1)  # Simulate thinking time

            ai_move = game.get_ai_move()
            if ai_move:
                start, end = ai_move
                game.make_move(start, end)
                move_str = format_move(start, end)
                print(f"🤖 AI (BLACK) plays: {move_str}")
            else:
                print("❌ AI has no valid moves!")
                break

        move_count += 1
        time.sleep(1.5)  # Pause between moves for readability

    # Final position
    print(f"\n{'='*60}")
    print("FINAL POSITION")
    print(f"{'='*60}")
    game.display_board()

    if game.game_over:
        if game.winner:
            winner_name = "RED (You)" if game.winner == 'red' else "BLACK (AI)"
            print(f"🏆 Game Over! {winner_name} wins!")
        else:
            print("🤝 Game ended in a draw!")
    else:
        print("🎯 Demo ended - game still in progress")

    # Show final statistics
    print(f"\n📊 Game Statistics:")
    print(f"Total moves played: {move_count}")
    print(f"Current turn: {game.current_player.upper()}")

    # Count remaining pieces
    red_pieces = black_pieces = 0
    for row in range(10):
        for col in range(9):
            piece = game.board[row][col]
            if piece:
                if piece[0] == 'red':
                    red_pieces += 1
                else:
                    black_pieces += 1

    print(f"Pieces remaining - RED: {red_pieces}, BLACK: {black_pieces}")

def main():
    """Run the demo"""
    print("🎮 Automated Chinese Chess Demonstration")
    print("This will show you playing as RED against the AI")
    input("Press Enter to start the demo...")

    play_demo_game()

    print(f"\n{'='*60}")
    print("🎉 Demo completed!")
    print("🚀 To play interactively: python xiangqi_complete.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()