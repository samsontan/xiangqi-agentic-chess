#!/usr/bin/env python3
"""
Visual Chinese Chess Board Renderer
"""

from xiangqi_complete import XiangqiGame

def render_board_ascii(game):
    """Render board with clear ASCII art"""

    # Piece mapping to readable symbols
    piece_symbols = {
        ('red', 'general'): '帥', ('black', 'general'): '將',
        ('red', 'advisor'): '仕', ('black', 'advisor'): '士',
        ('red', 'elephant'): '相', ('black', 'elephant'): '象',
        ('red', 'horse'): '馬', ('black', 'horse'): '馬',
        ('red', 'chariot'): '車', ('black', 'chariot'): '車',
        ('red', 'cannon'): '炮', ('black', 'cannon'): '炮',
        ('red', 'soldier'): '兵', ('black', 'soldier'): '卒'
    }

    print("\n" + "="*70)
    print("     🎮 CHINESE CHESS (XIANGQI) BOARD 🎮")
    print("="*70)
    print("     BLACK (AI) ⚫ - Top")
    print("     RED (You) 🔴 - Bottom")
    print("="*70)

    # Column headers
    print("      a   b   c   d   e   f   g   h   i")
    print("    ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐")

    for row in range(10):
        rank = 10 - row

        # Row number
        print(f" {rank:2} │", end="")

        # Pieces in row
        for col in range(9):
            piece = game.board[row][col]
            if piece:
                symbol = piece_symbols[piece]
                if piece[0] == 'red':
                    print(f" {symbol} │", end="")  # Red pieces
                else:
                    print(f" {symbol} │", end="")  # Black pieces
            else:
                # Empty squares with special markings
                if row == 4 or row == 5:  # River
                    print(" ~ │", end="")
                elif (row in [0,1,2,7,8,9] and col in [3,4,5]):  # Palace
                    print(" · │", end="")
                else:
                    print("   │", end="")

        print(f" {rank}")

        # Horizontal separators
        if row == 4:  # Before river
            print("    ├───┼───┼───┼───┼───┼───┼───┼───┼───┤")
            print("    │ ~ │ ~ │ ~ │ R I V E R │ ~ │ ~ │ ~ │")
            print("    ├───┼───┼───┼───┼───┼───┼───┼───┼───┤")
        elif row < 9:
            print("    ├───┼───┼───┼───┼───┼───┼───┼───┼───┤")

    print("    └───┴───┴───┴───┴───┴───┴───┴───┴───┘")
    print("      a   b   c   d   e   f   g   h   i")

def render_board_simple(game):
    """Render board in simplified format"""

    piece_symbols = {
        ('red', 'general'): 'K', ('black', 'general'): 'k',
        ('red', 'advisor'): 'A', ('black', 'advisor'): 'a',
        ('red', 'elephant'): 'E', ('black', 'elephant'): 'e',
        ('red', 'horse'): 'H', ('black', 'horse'): 'h',
        ('red', 'chariot'): 'R', ('black', 'chariot'): 'r',
        ('red', 'cannon'): 'C', ('black', 'cannon'): 'c',
        ('red', 'soldier'): 'P', ('black', 'soldier'): 'p'
    }

    print("\n" + "="*50)
    print("    XIANGQI BOARD (Simplified View)")
    print("="*50)
    print("    Files: a b c d e f g h i")
    print("    BLACK pieces: lowercase (AI)")
    print("    RED pieces: UPPERCASE (You)")
    print("-"*50)

    for row in range(10):
        rank = 10 - row
        print(f"R{rank:2}│", end="")

        for col in range(9):
            piece = game.board[row][col]
            if piece:
                symbol = piece_symbols[piece]
                print(f" {symbol}", end="")
            else:
                if row == 4 or row == 5:  # River
                    print(" ~", end="")
                else:
                    print(" ·", end="")

        print(f" │R{rank}")

        if row == 4:
            print("   │ ~ ~ ~ RIVER ~ ~ ~ │")

    print("   └─────────────────────┘")
    print("    Files: a b c d e f g h i")

def show_piece_legend():
    """Show what each piece represents"""
    print("\n" + "="*50)
    print("           PIECE LEGEND")
    print("="*50)
    print("Chinese │ English    │ RED │ BLACK")
    print("────────┼────────────┼─────┼──────")
    print("  帥/將  │ General    │  帥  │  將  ")
    print("  仕/士  │ Advisor    │  仕  │  士  ")
    print("  相/象  │ Elephant   │  相  │  象  ")
    print("   馬    │ Horse      │  馬  │  馬  ")
    print("   車    │ Chariot    │  車  │  車  ")
    print("   炮    │ Cannon     │  炮  │  炮  ")
    print("  兵/卒  │ Soldier    │  兵  │  卒  ")
    print("="*50)

def show_game_state(game):
    """Show current game state info"""
    print("\n" + "="*50)
    print("           GAME STATE")
    print("="*50)
    print(f"Current Player: {game.current_player.upper()}")

    # Check status
    if game.is_in_check('red'):
        print("⚠️  RED is in CHECK!")
    if game.is_in_check('black'):
        print("⚠️  BLACK is in CHECK!")

    if game.game_over:
        if game.winner:
            winner = "RED (You)" if game.winner == 'red' else "BLACK (AI)"
            print(f"🏆 WINNER: {winner}")
        else:
            print("🤝 DRAW")
    else:
        print("🎮 Game in progress")

    # Count pieces
    red_count = black_count = 0
    for row in range(10):
        for col in range(9):
            piece = game.board[row][col]
            if piece:
                if piece[0] == 'red':
                    red_count += 1
                else:
                    black_count += 1

    print(f"Pieces remaining - RED: {red_count}, BLACK: {black_count}")
    print("="*50)

def main():
    """Show visual representations of the Chinese chess board"""

    # Create game with demo moves
    game = XiangqiGame()

    print("🎮 CHINESE CHESS VISUAL RENDERER")
    print("Showing different visual representations of the board")

    # Show initial position
    print("\n" + "🟦"*25)
    print("         INITIAL POSITION")
    print("🟦"*25)

    render_board_ascii(game)
    show_piece_legend()
    show_game_state(game)

    # Make a few moves for demonstration
    moves = [
        ("e4-e5", "RED advances center soldier"),
        ("c7-c6", "BLACK advances soldier"),
        ("b3-e3", "RED moves cannon to center")
    ]

    for i, (move_str, description) in enumerate(moves):
        start, end = game.parse_move(move_str)
        if start and end and game.is_valid_move(start, end):
            game.make_move(start, end)

            print(f"\n{'🟩'*25}")
            print(f"    AFTER MOVE {i+1}: {description}")
            print(f"         {move_str}")
            print("🟩"*25)

            render_board_ascii(game)
            show_game_state(game)

    # Show simplified version
    print(f"\n{'🟨'*25}")
    print("     SIMPLIFIED BOARD VIEW")
    print("🟨"*25)
    render_board_simple(game)

    print(f"\n{'='*70}")
    print("🎉 Visual rendering complete!")
    print("🚀 To play the full interactive game: python xiangqi_complete.py")
    print("="*70)

if __name__ == "__main__":
    main()