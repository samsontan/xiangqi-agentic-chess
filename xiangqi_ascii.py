#!/usr/bin/env python3
"""
ASCII Xiangqi - Uses single-width ASCII characters for guaranteed alignment
"""

import random
import time
from xiangqi_fast import FastXiangqiGame

def draw_ascii_board(game):
    """Draw board using ASCII characters for perfect alignment"""
    piece_map = {
        ('red', 'general'): 'K', ('black', 'general'): 'k',
        ('red', 'advisor'): 'A', ('black', 'advisor'): 'a',
        ('red', 'elephant'): 'E', ('black', 'elephant'): 'e',
        ('red', 'horse'): 'H', ('black', 'horse'): 'h',
        ('red', 'chariot'): 'R', ('black', 'chariot'): 'r',
        ('red', 'cannon'): 'C', ('black', 'cannon'): 'c',
        ('red', 'soldier'): 'P', ('black', 'soldier'): 'p'
    }

    # ASCII legend
    print("\n🎮 RED (uppercase): K=General A=Advisor E=Elephant H=Horse R=Chariot C=Cannon P=Soldier")
    print("⚫ BLACK (lowercase): k=general a=advisor e=elephant h=horse r=chariot c=cannon p=soldier")
    print()

    # Perfect ASCII alignment
    print("  a b c d e f g h i")

    for row in range(10):
        rank = 10 - row
        line = f"{rank:2}"

        for col in range(9):
            piece = game.board[row][col]
            if piece:
                symbol = piece_map[piece]
                if piece[0] == 'red':
                    line += f"\033[91m{symbol}\033[0m "
                else:
                    line += f"\033[94m{symbol}\033[0m "
            else:
                line += ". "

        line += f"{rank:2}"
        print(line)

        # River marker
        if row == 4:
            print("  - - - RIVER - - -")

    print("  a b c d e f g h i\n")

class ASCIIXiangqiGame(FastXiangqiGame):
    def display_board(self):
        """Use ASCII board display"""
        draw_ascii_board(self)

def main():
    """Main game with ASCII perfect alignment"""
    print("🎮 ASCII XIANGQI - PERFECT ALIGNMENT")
    print("=" * 50)
    print("🔴 You are RED (UPPERCASE), ⚫ AI is BLACK (lowercase)")
    print("🚀 AI responds quickly (under 2 seconds)")
    print("Enter moves like: e4-e5 (with dash)")
    print("Type 'quit' to exit")
    print("=" * 50)

    game = ASCIIXiangqiGame()
    move_count = 0

    while not game.game_over and move_count < 100:
        game.display_board()

        if game.current_player == 'red':
            print(f"Move {move_count + 1} - Your turn (RED):")

            # Show some good opening moves
            if move_count == 0:
                print("💡 Good first moves: e4-e5, c4-c5, g4-g5")

            move_input = input("Enter your move: ").strip()

            if move_input.lower() == 'quit':
                print("Thanks for playing!")
                break

            start, end = game.parse_move(move_input)
            if start and end:
                if game.make_move(start, end):
                    print(f"✅ Move made: {move_input}")
                    move_count += 1
                else:
                    print("❌ Invalid move! Try again.")
                    print("💡 Make sure the piece can legally move there")
            else:
                print("❌ Invalid format! Use format like 'e4-e5'")

        else:
            print("🤖 AI thinking... (max 2 seconds)")
            start_time = time.time()

            ai_move = game.get_ai_move_fast(time_limit=2.0)
            think_time = time.time() - start_time

            if ai_move:
                start, end = ai_move
                game.make_move(start, end)
                from xiangqi_fast import format_move
                move_str = format_move(start, end)
                print(f"🤖 AI plays: {move_str} (thought for {think_time:.1f}s)")
                move_count += 1
            else:
                print("AI has no valid moves!")
                game.game_over = True

        if move_count >= 50:
            print("🤝 Game ended after 50 moves!")
            break

    print(f"\n🎉 Game finished after {move_count} moves!")
    print("Thanks for playing Chinese Chess!")

if __name__ == "__main__":
    main()