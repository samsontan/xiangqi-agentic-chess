#!/usr/bin/env python3
"""
Fast Chinese Chess (Xiangqi) Game with Time-Limited AI
Optimized for quick play
"""

import random
import copy
import time
import threading
from typing import Optional, Tuple, List

class FastXiangqiGame:
    def __init__(self):
        self.board = self.setup_board()
        self.current_player = 'red'
        self.game_over = False
        self.winner = None
        self.ai_thinking = False

    def setup_board(self):
        """Setup initial Xiangqi board position"""
        board = [[None for _ in range(9)] for _ in range(10)]

        # Red pieces (bottom - rows 7-9)
        pieces = [
            (9, 0, ('red', 'chariot')), (9, 8, ('red', 'chariot')),
            (9, 1, ('red', 'horse')), (9, 7, ('red', 'horse')),
            (9, 2, ('red', 'elephant')), (9, 6, ('red', 'elephant')),
            (9, 3, ('red', 'advisor')), (9, 5, ('red', 'advisor')),
            (9, 4, ('red', 'general')),
            (7, 1, ('red', 'cannon')), (7, 7, ('red', 'cannon')),
            (6, 0, ('red', 'soldier')), (6, 2, ('red', 'soldier')),
            (6, 4, ('red', 'soldier')), (6, 6, ('red', 'soldier')), (6, 8, ('red', 'soldier'))
        ]

        # Black pieces (top - rows 0-2)
        black_pieces = [
            (0, 0, ('black', 'chariot')), (0, 8, ('black', 'chariot')),
            (0, 1, ('black', 'horse')), (0, 7, ('black', 'horse')),
            (0, 2, ('black', 'elephant')), (0, 6, ('black', 'elephant')),
            (0, 3, ('black', 'advisor')), (0, 5, ('black', 'advisor')),
            (0, 4, ('black', 'general')),
            (2, 1, ('black', 'cannon')), (2, 7, ('black', 'cannon')),
            (3, 0, ('black', 'soldier')), (3, 2, ('black', 'soldier')),
            (3, 4, ('black', 'soldier')), (3, 6, ('black', 'soldier')), (3, 8, ('black', 'soldier'))
        ]

        for row, col, piece in pieces + black_pieces:
            board[row][col] = piece

        return board

    def display_board(self):
        """Display the current board state with proper alignment"""
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
            print(f"{rank:2}│", end="")
            for col in range(9):
                piece = self.board[row][col]
                if piece:
                    symbol = piece_symbols[piece]
                    color_code = '\033[91m' if piece[0] == 'red' else '\033[94m'
                    print(f" {color_code}{symbol}\033[0m ", end="│")
                else:
                    if row == 4 or row == 5:  # River
                        print(" ~ ", end="│")
                    elif (row in [0,1,2,7,8,9] and col in [3,4,5]):  # Palace
                        print(" + ", end="│")
                    else:
                        print(" · ", end="│")
            print(f"{rank:2}")

            # Draw horizontal dividers
            if row == 4:  # River
                print("  ├═══╪═══╪═══╪═══╪═══╪═══╪═══╪═══╪═══┤")
                print("  │           R I V E R           │")
                print("  ├───┼───┼───┼───┼───┼───┼───┼───┼───┤")
            elif row < 9:  # Not the last row
                print("  ├───┼───┼───┼───┼───┼───┼───┼───┼───┤")

        print("  └───┴───┴───┴───┴───┴───┴───┴───┴───┘")
        print("    a   b   c   d   e   f   g   h   i\n")

    def parse_move(self, move_str):
        """Parse move string like 'e1-e2' to coordinates"""
        try:
            start, end = move_str.split('-')
            start_col = ord(start[0].lower()) - ord('a')
            start_row = 10 - int(start[1:])
            end_col = ord(end[0].lower()) - ord('a')
            end_row = 10 - int(end[1:])
            return (start_row, start_col), (end_row, end_col)
        except:
            return None, None

    def is_valid_move(self, start, end):
        """Simplified move validation for speed"""
        start_row, start_col = start
        end_row, end_col = end

        # Basic bounds check
        if not (0 <= start_row < 10 and 0 <= start_col < 9 and
                0 <= end_row < 10 and 0 <= end_col < 9):
            return False

        piece = self.board[start_row][start_col]
        if not piece or piece[0] != self.current_player:
            return False

        target = self.board[end_row][end_col]
        if target and target[0] == self.current_player:
            return False

        # Simplified piece movement (fast validation)
        return True  # For speed, we'll allow most moves

    def make_move(self, start, end):
        """Make a move on the board"""
        if not self.is_valid_move(start, end):
            return False

        start_row, start_col = start
        end_row, end_col = end

        # Make the move
        self.board[end_row][end_col] = self.board[start_row][start_col]
        self.board[start_row][start_col] = None

        # Switch turns
        self.current_player = 'red' if self.current_player == 'black' else 'black'
        return True

    def get_all_valid_moves(self, color):
        """Get all valid moves for a color (simplified)"""
        moves = []
        for start_row in range(10):
            for start_col in range(9):
                piece = self.board[start_row][start_col]
                if piece and piece[0] == color:
                    # Generate some reasonable moves
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            end_row, end_col = start_row + dr, start_col + dc
                            if (0 <= end_row < 10 and 0 <= end_col < 9):
                                target = self.board[end_row][end_col]
                                if not target or target[0] != color:
                                    moves.append(((start_row, start_col), (end_row, end_col)))
        return moves[:50]  # Limit moves for speed

    def evaluate_position(self):
        """Fast position evaluation"""
        piece_values = {
            'general': 1000, 'advisor': 20, 'elephant': 20,
            'horse': 40, 'chariot': 90, 'cannon': 45, 'soldier': 10
        }

        score = 0
        for row in range(10):
            for col in range(9):
                piece = self.board[row][col]
                if piece:
                    color, piece_type = piece
                    value = piece_values[piece_type]
                    if color == 'black':  # AI is black
                        score += value
                    else:
                        score -= value
        return score

    def get_ai_move_fast(self, time_limit=2.0):
        """Get AI move with time limit"""
        start_time = time.time()
        best_move = None
        moves = self.get_all_valid_moves('black')

        if not moves:
            return None

        # Quick evaluation of moves
        best_score = float('-inf')

        for move in moves[:20]:  # Only check first 20 moves for speed
            if time.time() - start_time > time_limit:
                break

            start, end = move

            # Make move
            original_piece = self.board[end[0]][end[1]]
            self.board[end[0]][end[1]] = self.board[start[0]][start[1]]
            self.board[start[0]][start[1]] = None

            # Quick evaluation
            score = self.evaluate_position()

            # Undo move
            self.board[start[0]][start[1]] = self.board[end[0]][end[1]]
            self.board[end[0]][end[1]] = original_piece

            if score > best_score:
                best_score = score
                best_move = move

        # Fallback to random move if no good move found
        if best_move is None and moves:
            best_move = random.choice(moves)

        return best_move

def format_move(start, end):
    """Format move as string"""
    start_row, start_col = start
    end_row, end_col = end
    start_str = chr(ord('a') + start_col) + str(10 - start_row)
    end_str = chr(ord('a') + end_col) + str(10 - end_row)
    return f"{start_str}-{end_str}"

def main():
    """Main game loop with fast AI"""
    print("🎮 Fast Chinese Chess (Xiangqi) - Quick AI!")
    print("=" * 50)
    print("You are RED (帥), AI is BLACK (將)")
    print("AI thinks for maximum 2 seconds per move")
    print("Enter moves as: e1-e2 (from e1 to e2)")
    print("Type 'quit' to exit")
    print("=" * 50)

    game = FastXiangqiGame()
    move_count = 0

    while not game.game_over and move_count < 100:  # Limit game length
        game.display_board()

        if game.current_player == 'red':
            # Human turn
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
                print("❌ Invalid format! Use format like 'e1-e2'")

        else:
            # AI turn (fast)
            print("🤖 AI thinking... (max 2 seconds)")
            start_time = time.time()

            ai_move = game.get_ai_move_fast(time_limit=2.0)
            think_time = time.time() - start_time

            if ai_move:
                start, end = ai_move
                game.make_move(start, end)
                move_str = format_move(start, end)
                print(f"🤖 AI plays: {move_str} (thought for {think_time:.1f}s)")
                move_count += 1
            else:
                print("AI has no valid moves!")
                game.game_over = True

        # Simple game end condition
        if move_count >= 50:
            print("🤝 Game ended after 50 moves!")
            break

    print(f"\n🎉 Game finished after {move_count} moves!")
    print("Thanks for playing Chinese Chess!")

if __name__ == "__main__":
    main()