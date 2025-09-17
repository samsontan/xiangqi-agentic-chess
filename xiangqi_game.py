import random
import sys

class Piece:
    def __init__(self, name, color):
        self.name = name
        self.color = color

    def __str__(self):
        return self.name

class Board:
    def __init__(self):
        self.board = [
            [Piece('♖', 'red'), Piece('♘', 'red'), Piece('♗', 'red'), Piece('♕', 'red'), Piece('♔', 'red'), Piece('♗', 'red'), Piece('♘', 'red'), Piece('♖', 'red'), None],
            [None, None, None, None, None, None, None, None, None],
            [None, Piece('♙', 'red'), None, None, None, None, None, Piece('♙', 'red'), None],
            [None, None, None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None, None],
            [None, Piece('♟', 'black'), None, None, None, None, None, Piece('♟', 'black'), None],
            [None, None, None, None, None, None, None, None, None],
            [None, Piece('♙', 'black'), None, None, None, None, None, Piece('♙', 'black'), None],
            [Piece('♜', 'black'), Piece('♞', 'black'), Piece('♝', 'black'), Piece('♛', 'black'), Piece('♚', 'black'), Piece('♝', 'black'), Piece('♞', 'black'), Piece('♜', 'black'), None],
        ]

    def display(self):
        print("  a b c d e f g h i")
        for i, row in enumerate(self.board):
            print(i + 1, end=' ')
            for piece in row:
                if piece is None:
                    print("·", end=' ')
                else:
                    print(piece, end=' ')
            print()

    def move(self, start, end):
        start_row, start_col = start
        end_row, end_col = end
        self.board[end_row][end_col] = self.board[start_row][start_col]
        self.board[start_row][start_col] = None

    def is_valid_move(self, start, end):
        # This should contain the rules for each piece
        # For now, we'll allow any move
        return True

class Game:
    def __init__(self):
        self.board = Board()
        self.current_turn = 'red'

    def switch_turn(self):
        self.current_turn = 'black' if self.current_turn == 'red' else 'red'

    def play(self):
        while True:
            self.board.display()
            print(f"{self.current_turn}'s turn")
            if self.current_turn == 'red':
                move = input("Enter your move (e.g., a1 a2): ")
                if move.lower() == 'exit':
                    print("Exiting game.")
                    break
                start, end = self.parse_move(move)
                if self.board.is_valid_move(start, end):
                    self.board.move(start, end)
                    self.switch_turn()
                else:
                    print("Invalid move. Try again.")
            else:
                print("AI is thinking...")
                # AI move logic can be implemented here
                self.switch_turn()

    def parse_move(self, move):
        start, end = move.split()
        start_row, start_col = int(start[1]) - 1, ord(start[0]) - ord('a')
        end_row, end_col = int(end[1]) - 1, ord(end[0]) - ord('a')
        return (start_row, start_col), (end_row, end_col)

if __name__ == "__main__":
    game = Game()
    game.play()