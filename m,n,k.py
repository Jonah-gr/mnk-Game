import numpy as np
from random import randint
from collections import defaultdict
from itertools import permutations

class Board():
    def __init__(self, m=5, n=5, k=4):
        self.m = m
        self.n = n
        self.k = k

    def initialize(self):
        self.board = np.zeros((self.m, self.n))        

    def display(self):
        print(self.board)
        print("----------------------------------------------------------------------")
    
    def has_won(self, player_number):
        all_fields = []
        
        # prüft ob spieler mit player_number gewonnen hat
        for zeile in range(5):
            for spalte in range(5):
                counter = 0
                for k in range(4):
                    if zeile+k <= 4 and spalte+k <= 4:
                        if self.board[zeile+k][spalte+k] == player_number:
                            counter += 1
                if counter == 4:
                    return player_number
                counter = 0
                for k in range(4):
                    if spalte+k <= 4:
                        if self.board[zeile][spalte+k] == player_number:
                            counter += 1
                if counter == 4:
                    return player_number
                counter = 0
                for k in range(4):
                    if zeile+k <= 4:
                        if self.board[zeile+k][spalte] == player_number:
                            counter += 1
                if counter == 4:
                    return player_number
                counter = 0
                for k in range(4):
                    if zeile+k <= 4 and spalte-k >= 0:
                        if self.board[zeile+k][spalte-k] == player_number:
                            counter += 1
                if counter == 4:
                    return player_number
                counter = 0

        # prüft ob alle Felder schon belegt sind
        for row in range(self.m):
            for col in range(self.n):
                all_fields.append(self.board[row][col])
        if all(all_fields) == True:
            return 3
        return 0


#----------------------------------------------------------------------  

class Game():
    def __init__(self, player1, player2, board=Board(), m=5, n=5, k=4):
        self.m = m
        self.n = n
        self.k = k
        self.player1 = player1
        self.player2 = player2
        self.board = board

    def start(self):
        self.gameloop()

    def gameloop(self):
        self.board.initialize()
        self.board.display()
        while True:
            self.player1.make_move(self.board)
            self.board.display()
            if self.board.has_won(self.player1.player_number) == self.player1.player_number:
                print(f"Player{self.player1.player_number} has won!!")
                break
            elif self.board.has_won(self.player1.player_number) == 3:
                print("Draw!!")
                break
            self.player2.make_move(self.board)
            self.board.display()
            if self.board.has_won(self.player2.player_number) == self.player2.player_number:
                print(f"Player{self.player2.player_number} has won!!")
                break
            

#----------------------------------------------------------------------        

class Player():
    def __init__(self, name, symbol, player_number, m=5, n=5, k=4):
        self.m = m
        self.n = n
        self.k = k 
        self.name = name
        self.symbol = symbol
        self.player_number = player_number

    def make_move(self, board):
        x,y = input("Your move: ").split(" ")
        x,y = int(x), int(y)
        while x > self.m or y > self.n or board.board[int(x)][int(y)] != 0: #### weitere falsche Engaben aussortieren
            print("Unvalid move!")
            x,y = input("Your move: ").split(" ")
            x,y = int(x), int(y)
        board.board[x][y] = self.player_number
    


#---------------------------------------------------------------------- 
    
class simple_AI(Player):
    def __init__(self, symbol, player_number, name="AI",m=5, n=5, k=4):
        super().__init__(name, symbol, player_number, m, n, k)
    
    def make_move(self, board):
        x = randint(0, self.m-1)
        y = randint(0, self.n-1)
        while board.board[int(x)][int(y)] != 0:
            x = randint(0, self.m-1)
            y = randint(0, self.n-1)
        board.board[x][y] = self.player_number


#----------------------------------------------------------------------  

class middle_AI(Player):
    pass ### geht über jedes leere Feld und spielt dahin wo er gewinnt, sonst random
   

#---------------------------------------------------------------------- 

class hard_AI(Player):
    def __init__(self, symbol, player_number, name="AI",m=5, n=5, k=4):
        super().__init__(name, symbol, player_number, m, n, k)


    def get_empty_squares(self, board):
        empty_squares = []
        for row in range(self.m):
            for col in range(self.n):
                if board[row][col] == 0:
                    empty_squares.append((row, col))
        return empty_squares


    def get_tree(self, board, depth= 5):
        a = Board()
        a.board = board
        board = a
        if self.player_number == 1:
            other_player = 2
        elif self.player_number == 2:
            other_player = 1
        tree = defaultdict(list)
        single_poss = self.get_empty_squares(board.board)
        single_poss_permutations = permutations(single_poss)
        
        all_permutations_to_depth = []
        for permutation in single_poss_permutations:
            perm = permutation[:depth]
            all_permutations_to_depth.append(perm)
        
        all_permutations_to_depth = set(all_permutations_to_depth)

        for permutation in all_permutations_to_depth:
            zug = []
            board_copy = Board()
            board_copy.board = board.board.copy()
            AI_to_move = True
            z,s = permutation[0]
            for row, col in permutation:
                if AI_to_move:
                    board_copy.board[row][col] = self.player_number
                elif not AI_to_move:
                    board_copy.board[row][col] = other_player
                AI_to_move = not AI_to_move


                if board_copy.has_won(1) != 0 or board_copy.has_won(2) != 0:
                    break
            
            zug.append(board_copy.board.copy())
            tree[(z,s)].append(zug.copy())
            
        return tree

        

    def minimax(self, board):
        if self.player_number == 1:
            maximize = True
        else:
            maximize = False
        tree = self.get_tree(board)
        bewertung = defaultdict(int)
        for key in tree.keys():
            liste = []
            for array in tree[key]:
                cboard = Board()
                cboard.board = array[0]

                if cboard.has_won(1) == 0 or cboard.has_won(1) == 3:
                    liste.append(0)
                elif (cboard.has_won(1)) == 1:
                    liste.append(1)
                elif (cboard.has_won(2)) == 2:
                    liste.append(-1)
    
            bewertung[key] = liste

        victory_perc = []
        for valuelist in bewertung.values():
            sum = 0
            for value in valuelist:
                sum += value
            victory_perc.append(sum)
        if maximize:
            index = victory_perc.index(max(victory_perc))
        else:
            index = victory_perc.index(min(victory_perc))
        return list(bewertung.keys())[index]


    def one_move_win(self, board):
        copy_board = Board()
        copy_board.board = board.board.copy()
        if self.player_number == 1:
            other_player = 2
        elif self.player_number == 2:
            other_player = 1

        single_poss = self.get_empty_squares(board.board)
        for row,col in single_poss:
            copy_board.board[row][col] = self.player_number
            if copy_board.has_won(self.player_number) == self.player_number:
                return (row, col)
            copy_board.board[row][col] = 0

        for row,col in single_poss:
            copy_board.board[row][col] = other_player
            if copy_board.has_won(other_player) == other_player:
                return (row, col)
            copy_board.board[row][col] = 0




    def make_move(self, board):
        if self.one_move_win(board) != None:
            x,y = self.one_move_win(board)
        elif len(self.get_empty_squares(board.board)) >= 10:
            x = randint(0, self.m-1)
            y = randint(0, self.n-1)
            while board.board[int(x)][int(y)] != 0:
                x = randint(0, self.m-1)
                y = randint(0, self.n-1)
        else:
            x,y = self.minimax(board.board)
         
        print(f"hard_AI{self.player_number} chooses {(x,y)}")
        board.board[int(x)][int(y)] = self.player_number





#---------------------------------------------------------------------- 

if __name__ == "__main__":
    ### Player vs. Player:
    # game = Game(Player("player1", "X", 1), Player("player1", symbol="O", player_number=2))

    ### Player vs. simple_AI:
    # game = Game(Player("player1", "X", 1), simple_AI("O", 2))

    ### simple_AI vs. Player:
    # game = Game(simple_AI("X", 1), Player("player2", "O", 2))

    ### simple_AI vs. simple AI:
    game = Game(simple_AI("X", 1), simple_AI("O", 2))

    ### Player vs hard_AI:
    # game = Game(Player("player1", "X", 1), hard_AI("O", 2))

    ### hard_AI vs. hard_AI
    # game = Game(hard_AI("X", 1), hard_AI("O", 2))


    game.start()

    
############################################################
# TO DO's
# - board.display
# - player.make_move
# - board.has_won
############################################################
