#!/usr/bin/env python3
"""
Chinese Chess with Perfect Board Alignment
"""

from xiangqi_fast import FastXiangqiGame

class AlignedXiangqiGame(FastXiangqiGame):
    def display_board(self):
        """Display board with PERFECT alignment"""
        piece_symbols = {
            ('red', 'general'): '帥', ('black', 'general'): '將',
            ('red', 'advisor'): '仕', ('black', 'advisor'): '士',
            ('red', 'elephant'): '相', ('black', 'elephant'): '象',
            ('red', 'horse'): '馬', ('black', 'horse'): '馬',
            ('red', 'chariot'): '車', ('black', 'chariot'): '車',
            ('red', 'cannon'): '炮', ('black', 'cannon'): '炮',
            ('red', 'soldier'): '兵', ('black', 'soldier'): '卒'
        }

        print("\n    a   b   c   d   e   f   g   h   i")
        print("  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐")

        for row in range(10):
            rank = 10 - row
            line = f"{rank:2}│"

            for col in range(9):
                piece = self.board[row][col]
                if piece:
                    symbol = piece_symbols[piece]
                    if piece[0] == 'red':
                        line += f" \033[91m{symbol}\033[0m │"
                    else:
                        line += f" \033[94m{symbol}\033[0m │"
                else:
                    if row == 4 or row == 5:  # River
                        line += " ~ │"
                    elif (row in [0,1,2,7,8,9] and col in [3,4,5]):  # Palace
                        line += " + │"
                    else:
                        line += " · │"

            print(line)

            # Horizontal separators
            if row == 4:  # River
                print("  ├═══╪═══╪═══╪═══╪═══╪═══╪═══╪═══╪═══┤")
                print("  │           R I V E R           │")
                print("  ├───┼───┼───┼───┼───┼───┼───┼───┼───┤")
            elif row < 9:
                print("  ├───┼───┼───┼───┼───┼───┼───┼───┼───┤")

        print("  └───┴───┴───┴───┴───┴───┴───┴───┴───┘")
        print("    a   b   c   d   e   f   g   h   i\n")

def main():
    """Main game with perfect alignment"""
    print("🎮 Chinese Chess - PERFECTLY ALIGNED!")
    print("=" * 50)
    print("🔴 You are RED, ⚫ AI is BLACK")
    print("🚀 AI responds quickly (under 2 seconds)")
    print("Enter moves like: e4-e5")
    print("Type 'quit' to exit")
    print("=" * 50)

    game = AlignedXiangqiGame()
    move_count = 0

    while not game.game_over and move_count < 100:
        game.display_board()

        if game.current_player == 'red':
            print(f"Move {move_count + 1} - Your turn (RED):")
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
            else:
                print("❌ Invalid format! Use format like 'e4-e5'")

        else:
            print("🤖 AI thinking... (max 2 seconds)")
            import time
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

if __name__ == "__main__":
    main()