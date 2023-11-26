import numpy as np
import random 
from collections import defaultdict
from itertools import permutations, product


class Board():
    def __init__(self, m=5, n=5, k=4):
        self.m = m
        self.n = n
        self.k = k
        self.array = np.zeros((self.m, self.n))

    # Display method prints the current game board array to the console.
    def display(self):
        alphabet = [chr(value) for value in range(97, 123)]
        print("  "+"|", *alphabet[:self.n])
        print("--+" + '-' * (self.n * 2+1))
        array_list = list(map(list, [row for row in self.array]))
        for k in range(len(array_list)):
            array_list[k] = list(map(int, array_list[k]))
            print(k+1, "|", *array_list[k])
        print("_" * (2 * self.n + 4))

    # Check if any player has won the game by having k consecutive marks in a row, column, or diagonal.
    def has_won(self):
        player_numbers = [1,2]  # Check for both player numbers.
        for player_number in player_numbers:
            for row in range(self.m):
                for col in range(self.n):
                    counter = 0
                    for k in range(self.k):
                        # Check the main diagonal
                        if row+k <= self.m-1 and col+k <= self.n-1:
                            if self.array[row+k][col+k] == player_number:
                                counter += 1
                    if counter == self.k:
                        return player_number
                    counter = 0
                    for k in range(self.k):
                        # Check the horizontal direction
                        if col+k <= self.n-1:
                            if self.array[row][col+k] == player_number:
                                counter += 1
                    if counter == self.k:
                        return player_number
                    counter = 0
                    for k in range(self.k):
                        # Check the vertical direction
                        if row+k <= self.m-1:
                            if self.array[row+k][col] == player_number:
                                counter += 1
                    if counter == self.k:
                        return player_number
                    counter = 0
                    for k in range(self.k):
                        # Check the secondary diagonal
                        if row+k <= self.m-1 and col-k >= 0:
                            if self.array[row+k][col-k] == player_number:
                                counter += 1
                    if counter == self.k:
                        return player_number
                    counter = 0
        return 0



class Game():
    def __init__(self, player1, player2, board=Board(), m=5, n=5, k=4):
        self.m = m
        self.n = n
        self.k = k
        self.player1 = player1
        self.player2 = player2
        self.board = board

    # Start the game by calling the gameloop.
    def start(self):
        self.gameloop()

    # The gameloop is active as long as the game is running.
    def gameloop(self):
        game_on = True  # Variable indicating whether the game is running (True) or not (False)

        alphabet = [chr(value) for value in range(97, 123)]  # The following is for display purposes in the console
        row_dict = {}
        col_dict = {}
        for k in range(self.m):
            row_dict[k] = k + 1
        for k in range(self.n):
            col_dict[k] = alphabet[k]

        self.board.display()
        while game_on:
            for player in [self.player1, self.player2]:
                x, y = player.make_move(self.board)  # The player taking the turn selects the coordinates (row, column) for their move
                self.board.array[x][y] = player.player_number  # Update the game board with the last move
                print(f"{player.name} chooses {str(((row_dict[x]), col_dict[y]))}")
                self.board.display()
                if self.board.has_won() == player.player_number:  # Check if a player has won
                    print(f"{player.name} has won!!")
                    game_on = False  # End the game
                    break
                elif self.board.array.all():  # Check if all fields are occupied
                    print("Draw!!")
                    game_on = False  # End the game
                    break
            


class Player():
    def __init__(self, name, player_number):
        self.name = name
        self.player_number = player_number

    # Player.make_move() receives input from the player. If it is valid,
    # the method returns a tuple with the corresponding row and column.
    # Otherwise, the player has to provide input again.
    def make_move(self, board):
        m, n = board.array.shape
        alphabet = [chr(value) for value in range(97, 123)]  # alphabet, row_dict, and col_dict are used for converting user input to the tuple with coordinates on the numpy array
        row_dict = {}
        col_dict = {}
        for k in range(1, m+1):
            row_dict[k] = k - 1
        for k in range(1, n+1):
            col_dict[alphabet[k-1]] = k - 1
        try:  # Check if the move is valid
            move = list(input("Your move: "))
            x, y = move if len(move) == 2 else [move[0]+move[1], move[2]]  # x, y represent row, col, and in the case where the number of rows is 2 digits, this is correctly recognized
            x, y = row_dict[int(x)], col_dict[y]
            if board.array[x][y] != 0:  
                assert False  # Since the move is not valid, an error is raised here
            return (x, y)
        except:
            print("Invalid move!!")
            return self.make_move(board)  # Since the move is not valid, the function is called again


 
class RandomAI(Player):
    def __init__(self, player_number, name="RandomAI"):
        super().__init__(name, player_number)

    # RandomAI.get_empty_squares() receives a Board.array() and returns a list 
    # of tuples with all (row, column) coordinates that are equal to 0.
    def get_empty_squares(self, array):
        empty_squares = []
        for row in range(self.m):
            for col in range(self.n):
                if array[row][col] == 0:
                    empty_squares.append((row, col))
        return empty_squares

    # RandomAI.make_move receives all valid moves from RandomAI.get_empty_squares() 
    # and selects a random move from this list, then returns it.
    def make_move(self, board): 
        self.m, self.n = board.array.shape  # The size of the game board is obtained from the Board object
        x, y = random.choice(self.get_empty_squares(board.array))  # A random tuple from the list of possible moves is selected
        return (x, y)



class Minimax(Player):
    def __init__(self, player_number, name="Minimax"):
        super().__init__(name, player_number)
        self.other_player_number = 2 if self.player_number == 1 else 1

    # Minimax.get_empty_squares() receives a Board.array() and returns a list 
    # of tuples with all (row, column) coordinates that are equal to 0.
    def get_empty_squares(self, array):
        empty_squares = []
        for row in range(self.m):
            for col in range(self.n):
                if array[row][col] == 0:
                    empty_squares.append((row, col))
        return empty_squares

    # Minimax.get_tree() receives a Board object and a depth, and returns a 
    # dictionary with all possible moves as keys and all end results as arrays 
    # within a list.
    def get_tree(self, board, depth=4):
        tree = defaultdict(list)

        if np.count_nonzero(board.array == 0) > 16:
            self.small_board = Board(self.m-2, self.n-2, self.k-1)
            self.small_board.array = board.array[1:self.m-1, 1:self.n-1]
            self.m, self.n, self.k = self.m-2, self.n-2, self.k-1
            empty_squares = list(map(lambda x: (x[0]+1, x[1]+1), self.get_empty_squares(self.small_board.array)))
            depth = len(empty_squares)
        else:
            empty_squares = self.get_empty_squares(board.array)
            depth = len(empty_squares) if len(empty_squares) < depth else depth

        all_permutations = list(permutations(empty_squares, depth))

        for permutation in all_permutations:
            moves = []
            board_copy = Board()
            board_copy.array = board.array.copy()
            AI_to_move = True
            start_x, start_y = permutation[0] 
            for row, col in permutation:
                board_copy.array[row][col] = self.player_number if AI_to_move else self.other_player_number
                AI_to_move = not AI_to_move

                if np.count_nonzero(board.array == 0) > 20:
                    if self.small_board.has_won() != 0:
                        break
                else:
                    if board_copy.has_won() != 0:
                        break
            moves.append(board_copy.array.copy())
            tree[(start_x, start_y)].append(moves.copy())
        return tree

    # Minimax.minimax receives a Board object and returns a tuple (row, column). 
    # First, the game tree is generated, and then each result is evaluated.
    def minimax(self, board):
        maximize = True if self.player_number == 1 else False
        tree = self.get_tree(board)
        results = defaultdict(int)
        for key in tree.keys():
            results[key] = 0
            for array in tree[key]:
                if np.count_nonzero(board.array == 0) > 16:
                    self.small_board.array = array[0][1:self.m+1, 1:self.n+1]
                    if self.small_board.has_won() == 0:
                        results[key] += 0
                    elif self.small_board.has_won() == 1:
                        results[key] += 1
                    elif self.small_board.has_won() == 2:
                        results[key] -= 1
                else:
                    board_copy = Board()
                    board_copy.array = array[0]
                    if board_copy.has_won() == 0:
                        results[key] += 0
                    elif board_copy.has_won() == 1:
                        results[key] += 1
                    elif board_copy.has_won() == 2:
                        results[key] -= 1
        index = list(results.values()).index(max(results.values())) if maximize else list(results.values()).index(min(results.values()))
        return list(results.keys())[index]

    # Minimax.one_move_win receives a Board object and checks if Minimax or 
    # the opponent can win in the next move, returning the corresponding 
    # field as a tuple (row, column). If not, nothing is returned.
    def one_move_win(self, board):
        board_copy = Board(self.m, self.n, self.k)
        board_copy.array = board.array.copy() 

        empty_squares = self.get_empty_squares(board.array)
        for player_number in [self.player_number, self.other_player_number]:
            for row,col in empty_squares:
                board_copy.array[row][col] = player_number
                if board_copy.has_won() == player_number:
                    return (row, col)
                board_copy.array[row][col] = 0

    # Minimax.make_move receives a Board object and returns a tuple (row, column).
    def make_move(self, board):
        self.m, self.n = board.array.shape
        self.k = board.k 
        if self.one_move_win(board) != None:
            x,y = self.one_move_win(board)
        elif board.array[self.m // 2][self.n // 2] == 0:
            x,y = self.m // 2, self.n // 2
        elif np.count_nonzero(board.array == 0) == 24:
            smaller_board_squares = list(product(np.arange(self.m-self.k, self.k), np.arange(self.m-self.k, self.k)))
            smaller_board_squares = [i for i in smaller_board_squares if board.array[i[0]][i[1]] == 0]
            x,y = smaller_board_squares[random.randint(0,len(smaller_board_squares)-1)]
        else:
            x,y = self.minimax(board)

        return (x,y)



class MCTS(Player):
    def __init__(self, player_number, parent=None, parent_action=None, name="MCTS"):
        super().__init__(name, player_number)
        self.parent = parent
        self.parent_action = parent_action
        self.other_player_number = 1 if self.player_number == 2 else 2
        self.children = []
        self.number_of_visits = 0
        self.results = 0

    # MCTS.make_move() receives a Board object and returns a tuple (row, column).
    def make_move(self, board):
        self.m, self.n = board.array.shape
        self.k = board.k
        self.mcts_to_move = True
        self.children = []
        self.number_of_visits = 0
        self.results = 0
        self.array_edges = np.concatenate([board.array[0,:-1], board.array[:-1,-1], board.array[-1,::-1], board.array[-2:0:-1,0]])
        if not self.array_edges.any() and not board.array[1:self.m-1, 1:self.n-1].all():
            self.small_board = True
            self.board = Board(self.m-2, self.n-2, self.k-1)
            self.board.array = board.array[1:self.m-1, 1:self.n-1]
        else:
            self.small_board = False
            self.board = Board(self.m, self.n, self.k)
            self.board.array = board.array.copy()
        self.untried_actions = self.get_empty_squares(self.board.array)
        self.all_actions = self.untried_actions.copy()
        selected_node = self.best_action()
        x,y = selected_node.parent_action
        if self.small_board:
            x += 1
            y += 1
        return (x,y)

    # MCTS.get_empty_squares() receives a Board.array() and returns a list of 
    # tuples with all (row, column) coordinates that are equal to 0.
    def get_empty_squares(self, array):
        empty_squares = []
        for row in range(array.shape[0]):
            for col in range(array.shape[1]):
                if array[row][col] == 0:
                    empty_squares.append((row, col))
        return empty_squares

    # MCTS.expand() creates all possible child nodes for the current game state.
    def expand(self):
        for _ in range(len(self.untried_actions)):
            action = self.untried_actions.pop()
            array_copy = self.board.array.copy()
            array_copy[action[0]][action[1]] = self.player_number
            child_node = MCTS(player_number=self.player_number, parent=self, parent_action=action, name="MCTS")
            child_node.mcts_to_move = False
            child_node.board = Board(self.board.m, self.board.n, self.board.k)
            child_node.board.array = array_copy
            child_node.m, child_node.n = child_node.board.array.shape
            child_node.k = self.board.k
            self.children.append(child_node)

    # MCTS.is_terminal_board() returns True if a player has won, otherwise False.
    def is_terminal_board(self):
        if self.board.has_won() == 0:
            return False
        return True

    # MCTS.rollout() simulates all moves of a game round and returns a rating 
    # using MCTS.game_result().
    def rollout(self):
        current_rollout_state = self.board
        while current_rollout_state.has_won() == 0 and len(self.get_empty_squares(current_rollout_state.array)) != 0:
            possible_moves = self.get_empty_squares(current_rollout_state.array)
            action = self.rollout_policy(possible_moves)
            self.move(action)
            self.mcts_to_move = not self.mcts_to_move
        return self.game_result()

    # MCTS.move receives a field coordinate as a tuple (row, column) and 
    # simulates a move by the player who must make a move.
    def move(self, position):
        if self.mcts_to_move == True:
            self.board.array[position[0]][position[1]] = self.player_number
        else:
            self.board.array[position[0]][position[1]] = self.other_player_number

    # MCTS.backpropagate() receives a rating and updates the number_of_visits 
    # and results of itself or the parent node.
    def backpropagate(self, result):
        self.number_of_visits += 1
        self.results += result
        self.parent.number_of_visits += 1

    # MCTS.rollout_policy() receives a list of all possible moves and returns 
    # a random move from that list.
    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    # MCTS.game_result() evaluates the simulation and returns this evaluation.
    def game_result(self):
        result = 0
        if self.board.has_won() == self.player_number:
            result += 1 + np.count_nonzero(self.board.array == 0)
            if len(self.parent.all_actions) == np.count_nonzero(self.board.array == 0) + 1:
                result += 100
        elif self.board.has_won() == self.other_player_number:
            result -= 1 + np.count_nonzero(self.board.array == 0)
            if len(self.parent.all_actions) == np.count_nonzero(self.board.array == 0) + 1:
                result -= 100

        if self.parent_action == (self.m//2, self.n//2):
            result += 15

        if self.two_move_win(self.parent.board.array) == self.player_number:
            result += 1 + np.count_nonzero(self.board.array == 0)
        elif self.two_move_win(self.parent.board.array) == self.other_player_number:
            result -= 1 + np.count_nonzero(self.board.array == 0)
        return result

    # MCTS_two_move_win() receives an array and checks if MCTS or the opponent 
    # can win within two moves. If this is the case, the corresponding 
    # player_number is returned.
    def two_move_win(self, array):
        board_copy = Board(self.parent.board.m, self.parent.board.n, self.parent.board.k)
        board_copy.array = array.copy()
        empty_squares = self.get_empty_squares(board_copy.array)
        for player_number in [self.player_number, self.other_player_number]:
            for row,col in empty_squares:
                board_copy.array[row][col] = player_number
                if board_copy.has_won() == player_number:
                    return player_number

    # MCTS.best_child() selects the best child node using an algorithm and 
    # returns it.
    def best_child(self):   
        for child in self.children:
            if child.number_of_visits == 0 and child.results == 0:
                return child
        choices_weights = [(c.results / c.number_of_visits) + np.sqrt(2) * np.sqrt((np.log(self.number_of_visits) / c.number_of_visits)) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    # MCTS.best_action returns the best child node after 1000 simulations.
    def best_action(self):
        simulation_no = 1000
        self.expand()
        for _ in range(simulation_no): 
            best_child = self.best_child()
            reward = best_child.rollout()
            best_child.backpropagate(reward)
        return self.best_child()



# game_menu() starts a selection window where the user can choose game settings.
# Then the game is started with the corresponding settings.
def game_menu():
    print(" +----------------------------------+\n",
          "|            Welcome to            |\n",
          "|           m,n,k-Game!!           |\n",
          "+----------------------------------+")
    
    print("Select a game mode.")
    print("1: Singleplayer \n2: Multiplayer \n3: AI vs. AI" )
    game_mode = input("Selection: ")  # The user chooses the game mode
    while game_mode != "1" and game_mode != "2" and game_mode != "3":  # Input is checked, and if not valid, the user must choose again
        game_mode = input("Please select a valid game mode: ")

    if game_mode == "1":  # Checks if Singleplayer mode is chosen
        user_number = input("Do you want to play as Player 1 or Player 2? ")  # The user chooses their player number
        while user_number != "1" and user_number != "2":  # Input is checked, and if not valid, the user must choose again
            user_number = input("Please enter a valid number: ") 
        ai_number = 1 if int(user_number) == 2 else 2  # The player number of the AI is chosen based on the previous input
        user_name = input("Enter your name. ")  # The user chooses their player name
        user = Player(user_name, int(user_number))
        print("Select an AI you want to challenge.") 
        print("1: random AI \n2: Minimax \n3: Monte Carlo tree search")
        ai_choice = input("Selection: ")  # The user chooses which AI to play against
        while ai_choice != "1" and ai_choice != "2" and ai_choice != "3":  # Input is checked, and if not valid, the user must choose again
            ai_choice = input("Please enter a valid number: ")
        if ai_choice == "1":  # The AI is chosen based on the input
            ai = RandomAI(ai_number)
        elif ai_choice == "2":
            ai = Minimax(ai_number)
        else:
            ai = MCTS(ai_number)
 
        player1 = user if user_number == "1" else ai  # player1 is the one with player number 1
        player2 = user if player1 == ai else ai

    elif game_mode == "2":  # Checks if Multiplayer mode is chosen
        print("Enter your names.") 
        player1 = Player(input("Player 1: "), 1)  # Player 1 chooses their name
        player2 = Player(input("Player 2: "), 2)  # Player 2 chooses their name

    else:  # If neither Singleplayer nor Multiplayer is chosen, "AI vs. AI" is selected
        print("Select AIs that should battle each other.") 
        print("1: random AI \n2: Minimax \n3: Monte Carlo tree search")

        for ai_number in [1, 2]:
            player = "ai" + str(ai_number)  # A dynamic variable name is defined
             
            ai = input("First AI:" ) if ai_number == 1 else input("Second AI:" )  # Depending on ai_number, the first or second AI is asked
            while ai != "1" and ai != "2" and ai != "3":  # Input is checked, and if not valid, the user must choose again
                ai = input("Please enter a valid number: ")
            if ai == "1":  # The AI is chosen based on the input
                globals()[player] = RandomAI(ai_number)  # The global symbol table is accessed, and a new dynamic variable with the name defined in "Player" is created 
            elif ai == "2":
                globals()[player] = Minimax(ai_number)
            else:
                globals()[player] = MCTS(ai_number)  
        player1 = ai1  # player1 becomes the dynamic variable where the first AI was defined
        player2 = ai2  # player2 becomes the dynamic variable where the second AI was defined

    print("Enter the board size.")  # The user chooses the number of rows, columns, and stones needed to win in a row
    m = 0
    while not (type(m) == int and (0 < m < 100)):  # Input is checked, and if not valid, the user must choose again. In the first iteration, the loop is always called
        try:  # Try/except is necessary here since int(input()) can raise an error
            m = int(input("Number of rows: "))  # The user chooses the number of rows (limited to 99)
            if not (type(m) == int and (0 < m < 100)):  # Input is checked, and if not valid, an error is raised, and the user must choose again
                assert False
        except:
            print("Please enter a valid number!")

    n = 0
    while not (type(n) == int and (0 < n < 27)):
        try:
            n = int(input("Number of columns: "))  # The user chooses the number of columns (limited to 26)
            if not (type(n) == int and (0 < n < 27)):
                assert False
        except:
            print("Please enter a valid number!")

    k = 0
    while not (type(k) == int and (0 < k < max(m, n) + 1)):
        try:
            k = int(input("Number of stones you need in a row to win: "))  # The user chooses the number of stones needed to win in a row (limited to the maximum of the number of columns and rows)
            if not (type(k) == int and (0 < k < max(m, n) + 1)):
                assert False
        except:
            print("Please enter a valid number!")

    game = Game(player1, player2, Board(m, n, k), m, n, k)  # A Game object with the previously selected settings is created 
    game.start()  # The game is started


if __name__ == "__main__":
    game_menu()  # When the program is started, the Game menu is opened
