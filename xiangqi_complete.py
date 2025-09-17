#!/usr/bin/env python3
"""
Complete Chinese Chess (Xiangqi) Game with AI
Built using agentic swarm development approach
"""

import random
import copy

class XiangqiGame:
    def __init__(self):
        # Board representation: 10 rows x 9 columns
        # Row 0-9 (rank 1-10), Col 0-8 (file a-i)
        self.board = self.setup_board()
        self.current_player = 'red'  # Red moves first
        self.game_over = False
        self.winner = None

    def setup_board(self):
        """Setup initial Xiangqi board position"""
        board = [[None for _ in range(9)] for _ in range(10)]

        # Red pieces (bottom - rows 7-9)
        # Chariots (車)
        board[9][0] = ('red', 'chariot')
        board[9][8] = ('red', 'chariot')

        # Horses (馬)
        board[9][1] = ('red', 'horse')
        board[9][7] = ('red', 'horse')

        # Elephants (相)
        board[9][2] = ('red', 'elephant')
        board[9][6] = ('red', 'elephant')

        # Advisors (仕)
        board[9][3] = ('red', 'advisor')
        board[9][5] = ('red', 'advisor')

        # General (帥)
        board[9][4] = ('red', 'general')

        # Cannons (炮)
        board[7][1] = ('red', 'cannon')
        board[7][7] = ('red', 'cannon')

        # Soldiers (兵)
        for col in range(0, 9, 2):
            board[6][col] = ('red', 'soldier')

        # Black pieces (top - rows 0-2)
        # Chariots (車)
        board[0][0] = ('black', 'chariot')
        board[0][8] = ('black', 'chariot')

        # Horses (馬)
        board[0][1] = ('black', 'horse')
        board[0][7] = ('black', 'horse')

        # Elephants (象)
        board[0][2] = ('black', 'elephant')
        board[0][6] = ('black', 'elephant')

        # Advisors (士)
        board[0][3] = ('black', 'advisor')
        board[0][5] = ('black', 'advisor')

        # General (將)
        board[0][4] = ('black', 'general')

        # Cannons (炮)
        board[2][1] = ('black', 'cannon')
        board[2][7] = ('black', 'cannon')

        # Soldiers (卒)
        for col in range(0, 9, 2):
            board[3][col] = ('black', 'soldier')

        return board

    def display_board(self):
        """Display the current board state"""
        piece_symbols = {
            ('red', 'general'): '帥', ('black', 'general'): '將',
            ('red', 'advisor'): '仕', ('black', 'advisor'): '士',
            ('red', 'elephant'): '相', ('black', 'elephant'): '象',
            ('red', 'horse'): '馬', ('black', 'horse'): '馬',
            ('red', 'chariot'): '車', ('black', 'chariot'): '車',
            ('red', 'cannon'): '炮', ('black', 'cannon'): '炮',
            ('red', 'soldier'): '兵', ('black', 'soldier'): '卒'
        }

        print("\n    a b c d e f g h i")
        print("  ┌─────────────────────┐")

        for row in range(10):
            print(f"{10-row:2}│", end="")
            for col in range(9):
                piece = self.board[row][col]
                if piece:
                    symbol = piece_symbols[piece]
                    color_code = '\033[91m' if piece[0] == 'red' else '\033[94m'
                    print(f"{color_code}{symbol}\033[0m", end=" ")
                else:
                    # Show river and palace markings
                    if row == 4 or row == 5:  # River
                        print("~", end=" ")
                    elif (row in [0, 1, 2, 7, 8, 9] and col in [3, 4, 5]):  # Palace
                        print("·", end=" ")
                    else:
                        print("·", end=" ")
            print(f"│{10-row}")

            # Show river
            if row == 4:
                print("  │    ~ R I V E R ~     │")

        print("  └─────────────────────┘")
        print("    a b c d e f g h i\n")

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
        """Check if a move is valid according to Xiangqi rules"""
        start_row, start_col = start
        end_row, end_col = end

        # Check bounds
        if not (0 <= start_row < 10 and 0 <= start_col < 9 and
                0 <= end_row < 10 and 0 <= end_col < 9):
            return False

        piece = self.board[start_row][start_col]
        if not piece or piece[0] != self.current_player:
            return False

        target = self.board[end_row][end_col]
        if target and target[0] == self.current_player:
            return False  # Can't capture own piece

        color, piece_type = piece

        # Validate move based on piece type
        if piece_type == 'general':
            return self.is_valid_general_move(start, end)
        elif piece_type == 'advisor':
            return self.is_valid_advisor_move(start, end)
        elif piece_type == 'elephant':
            return self.is_valid_elephant_move(start, end)
        elif piece_type == 'horse':
            return self.is_valid_horse_move(start, end)
        elif piece_type == 'chariot':
            return self.is_valid_chariot_move(start, end)
        elif piece_type == 'cannon':
            return self.is_valid_cannon_move(start, end)
        elif piece_type == 'soldier':
            return self.is_valid_soldier_move(start, end)

        return False

    def is_valid_general_move(self, start, end):
        """General can only move 1 step within palace"""
        start_row, start_col = start
        end_row, end_col = end

        # Check if within palace
        if self.current_player == 'red':
            if not (7 <= end_row <= 9 and 3 <= end_col <= 5):
                return False
        else:
            if not (0 <= end_row <= 2 and 3 <= end_col <= 5):
                return False

        # Check if move is exactly 1 step orthogonally
        return abs(end_row - start_row) + abs(end_col - start_col) == 1

    def is_valid_advisor_move(self, start, end):
        """Advisor moves diagonally 1 step within palace"""
        start_row, start_col = start
        end_row, end_col = end

        # Check if within palace
        if self.current_player == 'red':
            if not (7 <= end_row <= 9 and 3 <= end_col <= 5):
                return False
        else:
            if not (0 <= end_row <= 2 and 3 <= end_col <= 5):
                return False

        # Check diagonal movement
        return abs(end_row - start_row) == 1 and abs(end_col - start_col) == 1

    def is_valid_elephant_move(self, start, end):
        """Elephant moves 2 points diagonally, can't cross river"""
        start_row, start_col = start
        end_row, end_col = end

        # Can't cross river
        if self.current_player == 'red':
            if end_row < 5:
                return False
        else:
            if end_row > 4:
                return False

        # Must move exactly 2 points diagonally
        if abs(end_row - start_row) != 2 or abs(end_col - start_col) != 2:
            return False

        # Check if path is blocked
        block_row = start_row + (end_row - start_row) // 2
        block_col = start_col + (end_col - start_col) // 2
        return self.board[block_row][block_col] is None

    def is_valid_horse_move(self, start, end):
        """Horse moves in L-shape but can be blocked"""
        start_row, start_col = start
        end_row, end_col = end

        row_diff = end_row - start_row
        col_diff = end_col - start_col

        # Check L-shape movement
        if not ((abs(row_diff) == 2 and abs(col_diff) == 1) or
                (abs(row_diff) == 1 and abs(col_diff) == 2)):
            return False

        # Check blocking point
        if abs(row_diff) == 2:
            block_row = start_row + row_diff // 2
            block_col = start_col
        else:
            block_row = start_row
            block_col = start_col + col_diff // 2

        return self.board[block_row][block_col] is None

    def is_valid_chariot_move(self, start, end):
        """Chariot moves like a rook"""
        start_row, start_col = start
        end_row, end_col = end

        # Must move in straight line
        if start_row != end_row and start_col != end_col:
            return False

        # Check path is clear
        if start_row == end_row:  # Horizontal movement
            step = 1 if end_col > start_col else -1
            for col in range(start_col + step, end_col, step):
                if self.board[start_row][col] is not None:
                    return False
        else:  # Vertical movement
            step = 1 if end_row > start_row else -1
            for row in range(start_row + step, end_row, step):
                if self.board[row][start_col] is not None:
                    return False

        return True

    def is_valid_cannon_move(self, start, end):
        """Cannon moves like chariot but must jump to capture"""
        start_row, start_col = start
        end_row, end_col = end

        # Must move in straight line
        if start_row != end_row and start_col != end_col:
            return False

        target = self.board[end_row][end_col]

        # Count pieces between start and end
        pieces_between = 0
        if start_row == end_row:  # Horizontal
            step = 1 if end_col > start_col else -1
            for col in range(start_col + step, end_col, step):
                if self.board[start_row][col] is not None:
                    pieces_between += 1
        else:  # Vertical
            step = 1 if end_row > start_row else -1
            for row in range(start_row + step, end_row, step):
                if self.board[row][start_col] is not None:
                    pieces_between += 1

        # If capturing, must have exactly 1 piece between
        # If not capturing, must have 0 pieces between
        if target:
            return pieces_between == 1
        else:
            return pieces_between == 0

    def is_valid_soldier_move(self, start, end):
        """Soldier moves forward, sideways only after crossing river"""
        start_row, start_col = start
        end_row, end_col = end

        row_diff = end_row - start_row
        col_diff = end_col - start_col

        # Must move exactly 1 step
        if abs(row_diff) + abs(col_diff) != 1:
            return False

        if self.current_player == 'red':
            # Red soldiers move up (decreasing row numbers)
            if row_diff > 0:  # Moving backwards
                return False

            # Can move sideways only after crossing river (row < 5)
            if col_diff != 0 and start_row >= 5:
                return False
        else:
            # Black soldiers move down (increasing row numbers)
            if row_diff < 0:  # Moving backwards
                return False

            # Can move sideways only after crossing river (row > 4)
            if col_diff != 0 and start_row <= 4:
                return False

        return True

    def make_move(self, start, end):
        """Make a move on the board"""
        if not self.is_valid_move(start, end):
            return False

        start_row, start_col = start
        end_row, end_col = end

        # Make the move
        self.board[end_row][end_col] = self.board[start_row][start_col]
        self.board[start_row][start_col] = None

        # Check for game over conditions
        if self.is_checkmate():
            self.game_over = True
            self.winner = 'red' if self.current_player == 'black' else 'black'
        else:
            # Switch turns
            self.current_player = 'red' if self.current_player == 'black' else 'black'

        return True

    def find_general(self, color):
        """Find the general of given color"""
        for row in range(10):
            for col in range(9):
                piece = self.board[row][col]
                if piece and piece == (color, 'general'):
                    return (row, col)
        return None

    def is_in_check(self, color):
        """Check if the general of given color is in check"""
        general_pos = self.find_general(color)
        if not general_pos:
            return False

        # Check if any opponent piece can capture the general
        opponent = 'red' if color == 'black' else 'black'
        for row in range(10):
            for col in range(9):
                piece = self.board[row][col]
                if piece and piece[0] == opponent:
                    # Temporarily switch player to check valid moves
                    original_player = self.current_player
                    self.current_player = opponent
                    if self.is_valid_move((row, col), general_pos):
                        self.current_player = original_player
                        return True
                    self.current_player = original_player

        return False

    def is_checkmate(self):
        """Check if current player is in checkmate"""
        if not self.is_in_check(self.current_player):
            return False

        # Try all possible moves to see if check can be escaped
        for start_row in range(10):
            for start_col in range(9):
                piece = self.board[start_row][start_col]
                if piece and piece[0] == self.current_player:
                    for end_row in range(10):
                        for end_col in range(9):
                            if self.is_valid_move((start_row, start_col), (end_row, end_col)):
                                # Try the move
                                original_piece = self.board[end_row][end_col]
                                self.board[end_row][end_col] = self.board[start_row][start_col]
                                self.board[start_row][start_col] = None

                                # Check if still in check
                                still_in_check = self.is_in_check(self.current_player)

                                # Undo the move
                                self.board[start_row][start_col] = self.board[end_row][end_col]
                                self.board[end_row][end_col] = original_piece

                                if not still_in_check:
                                    return False

        return True

    def get_all_valid_moves(self, color):
        """Get all valid moves for a color"""
        moves = []
        original_player = self.current_player
        self.current_player = color

        for start_row in range(10):
            for start_col in range(9):
                piece = self.board[start_row][start_col]
                if piece and piece[0] == color:
                    for end_row in range(10):
                        for end_col in range(9):
                            if self.is_valid_move((start_row, start_col), (end_row, end_col)):
                                # Check if move leaves king in check
                                original_piece = self.board[end_row][end_col]
                                self.board[end_row][end_col] = self.board[start_row][start_col]
                                self.board[start_row][start_col] = None

                                if not self.is_in_check(color):
                                    moves.append(((start_row, start_col), (end_row, end_col)))

                                # Undo move
                                self.board[start_row][start_col] = self.board[end_row][end_col]
                                self.board[end_row][end_col] = original_piece

        self.current_player = original_player
        return moves

    def evaluate_position(self):
        """Evaluate current position for AI"""
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

    def minimax(self, depth, maximizing_player, alpha=float('-inf'), beta=float('inf')):
        """Minimax algorithm with alpha-beta pruning"""
        if depth == 0 or self.game_over:
            return self.evaluate_position()

        if maximizing_player:
            max_eval = float('-inf')
            for move in self.get_all_valid_moves('black'):
                start, end = move

                # Make move
                original_piece = self.board[end[0]][end[1]]
                self.board[end[0]][end[1]] = self.board[start[0]][start[1]]
                self.board[start[0]][start[1]] = None

                eval_score = self.minimax(depth - 1, False, alpha, beta)

                # Undo move
                self.board[start[0]][start[1]] = self.board[end[0]][end[1]]
                self.board[end[0]][end[1]] = original_piece

                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break

            return max_eval
        else:
            min_eval = float('inf')
            for move in self.get_all_valid_moves('red'):
                start, end = move

                # Make move
                original_piece = self.board[end[0]][end[1]]
                self.board[end[0]][end[1]] = self.board[start[0]][start[1]]
                self.board[start[0]][start[1]] = None

                eval_score = self.minimax(depth - 1, True, alpha, beta)

                # Undo move
                self.board[start[0]][start[1]] = self.board[end[0]][end[1]]
                self.board[end[0]][end[1]] = original_piece

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break

            return min_eval

    def get_ai_move(self):
        """Get best move for AI using minimax"""
        best_move = None
        best_value = float('-inf')

        for move in self.get_all_valid_moves('black'):
            start, end = move

            # Make move
            original_piece = self.board[end[0]][end[1]]
            self.board[end[0]][end[1]] = self.board[start[0]][start[1]]
            self.board[start[0]][start[1]] = None

            move_value = self.minimax(3, False)  # Depth 3

            # Undo move
            self.board[start[0]][start[1]] = self.board[end[0]][end[1]]
            self.board[end[0]][end[1]] = original_piece

            if move_value > best_value:
                best_value = move_value
                best_move = move

        return best_move

def format_move(start, end):
    """Format move as string"""
    start_row, start_col = start
    end_row, end_col = end
    start_str = chr(ord('a') + start_col) + str(10 - start_row)
    end_str = chr(ord('a') + end_col) + str(10 - end_row)
    return f"{start_str}-{end_str}"

def main():
    """Main game loop"""
    print("🎮 Welcome to Chinese Chess (Xiangqi)!")
    print("=" * 50)
    print("You are RED (帥), AI is BLACK (將)")
    print("Enter moves as: e1-e2 (from e1 to e2)")
    print("Type 'quit' to exit")
    print("=" * 50)

    game = XiangqiGame()

    while not game.game_over:
        game.display_board()

        if game.current_player == 'red':
            # Human turn
            print(f"Your turn (RED):")
            if game.is_in_check('red'):
                print("⚠️  You are in CHECK!")

            move_input = input("Enter your move: ").strip()

            if move_input.lower() == 'quit':
                print("Thanks for playing!")
                break

            start, end = game.parse_move(move_input)
            if start and end:
                if game.make_move(start, end):
                    print(f"✅ Move made: {move_input}")
                else:
                    print("❌ Invalid move! Try again.")
            else:
                print("❌ Invalid format! Use format like 'e1-e2'")

        else:
            # AI turn
            print("🤖 AI is thinking...")
            if game.is_in_check('black'):
                print("AI is in CHECK!")

            ai_move = game.get_ai_move()
            if ai_move:
                start, end = ai_move
                game.make_move(start, end)
                move_str = format_move(start, end)
                print(f"🤖 AI plays: {move_str}")
            else:
                print("AI has no valid moves!")
                game.game_over = True

    if game.game_over:
        game.display_board()
        if game.winner:
            winner_name = "You" if game.winner == 'red' else "AI"
            print(f"🎉 Game Over! {winner_name} wins!")
        else:
            print("🤝 Game ended in a draw!")

if __name__ == "__main__":
    main()