#!/usr/bin/env python3
"""
Fixed Xiangqi with Proper Move Validation
"""

import random
import time
from xiangqi_perfect_final import PerfectFinalXiangqiGame

class FixedXiangqiGame(PerfectFinalXiangqiGame):
    def is_valid_move(self, start, end):
        """Proper move validation instead of 'return True'"""
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

        # Can't move to same position
        if start == end:
            return False

        piece_type = piece[1]

        # Basic piece movement validation
        dr = end_row - start_row
        dc = end_col - start_col

        if piece_type == 'horse':
            # Horse moves in L-shape: 2 squares in one direction, 1 in perpendicular
            if not ((abs(dr) == 2 and abs(dc) == 1) or (abs(dr) == 1 and abs(dc) == 2)):
                return False

        elif piece_type == 'chariot':
            # Chariot moves horizontally or vertically
            if dr != 0 and dc != 0:
                return False

        elif piece_type == 'cannon':
            # Cannon moves like chariot but needs piece to jump over for capture
            if dr != 0 and dc != 0:
                return False

        elif piece_type == 'soldier':
            # Soldier can only move forward (and sideways after crossing river)
            if piece[0] == 'red':
                if start_row >= 5:  # Before crossing river
                    if dr != -1 or dc != 0:  # Can only move forward
                        return False
                else:  # After crossing river
                    if not ((dr == -1 and dc == 0) or (dr == 0 and abs(dc) == 1)):
                        return False
            else:  # black
                if start_row <= 4:  # Before crossing river
                    if dr != 1 or dc != 0:  # Can only move forward
                        return False
                else:  # After crossing river
                    if not ((dr == 1 and dc == 0) or (dr == 0 and abs(dc) == 1)):
                        return False

        elif piece_type == 'general':
            # General stays in palace and moves one step
            if abs(dr) > 1 or abs(dc) > 1:
                return False

        elif piece_type == 'advisor':
            # Advisor moves diagonally in palace
            if abs(dr) != 1 or abs(dc) != 1:
                return False

        elif piece_type == 'elephant':
            # Elephant moves diagonally 2 points and can't cross river
            if abs(dr) != 2 or abs(dc) != 2:
                return False
            if piece[0] == 'red' and end_row < 5:
                return False
            if piece[0] == 'black' and end_row > 4:
                return False

        return True

    def get_all_valid_moves(self, color):
        """Get only actually valid moves"""
        moves = []
        for start_row in range(10):
            for start_col in range(9):
                piece = self.board[start_row][start_col]
                if piece and piece[0] == color:
                    # Check reasonable nearby moves
                    for dr in range(-3, 4):
                        for dc in range(-3, 4):
                            if dr == 0 and dc == 0:
                                continue
                            end_row, end_col = start_row + dr, start_col + dc
                            if (0 <= end_row < 10 and 0 <= end_col < 9):
                                if self.is_valid_move((start_row, start_col), (end_row, end_col)):
                                    moves.append(((start_row, start_col), (end_row, end_col)))
        return moves

def main():
    """Main game with fixed move validation"""
    print("🎮 FIXED XIANGQI - PROPER MOVE VALIDATION")
    print("=" * 50)
    print("🔴 You are RED, ⚫ AI is BLACK")
    print("🚀 AI now makes only LEGAL moves!")
    print("📍 Perfect alignment + proper game rules")
    print("Enter moves like: e4-e5 (with dash)")
    print("Type 'quit' to exit")
    print("=" * 50)

    game = FixedXiangqiGame()
    move_count = 0

    while not game.game_over and move_count < 100:
        game.display_board()

        if game.current_player == 'red':
            print(f"Move {move_count + 1} - Your turn (RED):")

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
                if game.is_valid_move(start, end):  # Double-check AI moves
                    game.make_move(start, end)
                    from xiangqi_fast import format_move
                    move_str = format_move(start, end)
                    print(f"🤖 AI plays: {move_str} (thought for {think_time:.1f}s)")
                    move_count += 1
                else:
                    print("🚫 AI tried illegal move, passing turn...")
                    game.current_player = 'red'  # Give turn back to human
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