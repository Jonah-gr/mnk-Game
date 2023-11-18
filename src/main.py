import numpy as np
import random 
from collections import defaultdict
from itertools import permutations, product


#----------------------------------------------------------------------

class Board():
    def __init__(self, m=5, n=5, k=4):
        self.m = m
        self.n = n
        self.k = k
        self.array = np.zeros((self.m, self.n)) 
        

    # Board.display() gibt das aktuelle Board.array(), also das Spielbrett
    # in die Konsole aus.
    def display(self): 
        alphabet = [chr(value) for value in range(97, 123)]
        print("  "+"|", *alphabet[:self.n])
        print("--+" + '-' * (self.n * 2+1))
        array_list = list(map(list, [row for row in self.array]))
        for k in range(len(array_list)):
            array_list[k] = list(map(int, array_list[k]))
            print(k+1,"|",*array_list[k])
        print("_" * (2 * self.n + 4))
     

    # Board.has_won() überprüft, ob k-viele Einsen oder Zweien in einer 
    # Reihe liegen, also ob ein Spieler gewonnen hat.
    def has_won(self): 
        player_numbers = [1,2] # es wird für beide Spieler-Nummern geprüft, ob der jeweilige Spieler gewonnen hat 
        for player_number in player_numbers:
            for row in range(self.m):
                for col in range(self.n):
                    counter = 0
                    for k in range(self.k): #überprüft die erste Diagonale
                        if row+k <= self.m-1 and col+k <= self.n-1:
                            if self.array[row+k][col+k] == player_number:
                                counter += 1
                    if counter == self.k:
                        return player_number
                    counter = 0
                    for k in range(self.k): #überprüft die Waargerechte
                        if col+k <= self.n-1:
                            if self.array[row][col+k] == player_number:
                                counter += 1
                    if counter == self.k:
                        return player_number
                    counter = 0
                    for k in range(self.k): # überprüft die Senkrechte
                        if row+k <= self.m-1:
                            if self.array[row+k][col] == player_number:
                                counter += 1
                    if counter == self.k:
                        return player_number
                    counter = 0
                    for k in range(self.k): # überprüft die zweite Diagonale
                        if row+k <= self.m-1 and col-k >= 0:
                            if self.array[row+k][col-k] == player_number:
                                counter += 1
                    if counter == self.k:
                        return player_number
                    counter = 0
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


    # Game.start() startet Game.gameloop() und somit das Spiel.
    def start(self):
        self.gameloop()


    # Game.gameloop() ist aktiv, solange das Spiel läuft.
    def gameloop(self): 
        game_on = True # Variable, die anzeigt, ob das Spiel läuft (True) oder nicht (False)

        alphabet = [chr(value) for value in range(97, 123)] # das folgenden dient nur der Anzeige in der Konsole
        row_dict = {}
        col_dict = {}
        for k in range(self.m):
            row_dict[k] = k + 1
        for k in range(self.n):
            col_dict[k] = alphabet[k]
        
        self.board.display()
        while game_on:
            for player in [self.player1, self.player2]:
                x,y = player.make_move(self.board) # der Spieler, der an der Reihe ist wählt die Koordinaten (Zeile, Spalte), auf denen er seinen Zug machen möchte
                self.board.array[x][y] = player.player_number # das Spielbrett wird um den letzten Zug erweitert
                print(f"{player.name} chooses {str(((row_dict[x]), col_dict[y]))}")
                self.board.display()
                if self.board.has_won() == player.player_number: # überprüft, ob ein Spieler gewonnen hat
                    print(f"{player.name} has won!!")
                    game_on = False # das Spiel ist beendet
                    break
                elif self.board.array.all(): # überprüft, ob alle Felder besetzt sind
                    print("Draw!!")
                    game_on = False # das Spiel ist beendet
                    break
            
#----------------------------------------------------------------------        

class Player():
    def __init__(self, name, player_number):
        self.name = name
        self.player_number = player_number


    # Player.make_move() bekommt vom Spieler einen Input. Wenn dieser 
    # legitim ist, gibt die Methode einen Tupel mit der entsprechenden
    # Zeile und Saplte zurück. Anderenfalls muss der Spieler erneut 
    # einen Input geben.
    def make_move(self, board):
        m,n = board.array.shape
        alphabet = [chr(value) for value in range(97, 123)] # alphabet, row_dict und col_dict dienen nur der umrechnung der User-Eingabe zum Tupel mit den Koordinaten auf dem numpy-array
        row_dict = {}
        col_dict = {}
        for k in range(1, m+1):
            row_dict[k] = k - 1
        for k in range(1, n+1):
            col_dict[alphabet[k-1]] = k -1
        try: # überprüft, ob der Zug legitim ist
            move = list(input("Your move: "))
            x,y = move if len(move) == 2 else [move[0]+move[1], move[2]] # x,y steht für row, col, und im Falle, dass die Anzahl der Zeilen 2-stellig ist, wird dies auch richtig erkannt
            x,y = row_dict[int(x)], col_dict[y]
            if board.array[x][y] != 0:  
                assert False # da der Zug nicht legitim ist, wird hier ein Fehler ausgegeben
            return (x,y)
        except: # 
            print("Unvalid move!!")
            return self.make_move(board) # da der Zug nicht legitim ist, wird die Funktion erneut aufgerufen

#---------------------------------------------------------------------- 
 
class random_AI(Player):
    def __init__(self, player_number, name="random_AI"):
        super().__init__(name, player_number)


    # random_AI.get_empty_squares() bekommt ein Board.array() übergeben 
    # und gibt eine Liste  mit Tupeln mit allen (Zeile, Spalte)-Koordinaten, 
    # die gleich 0 sind, zurück.
    def get_empty_squares(self, array):
        empty_squares = []
        for row in range(self.m):
            for col in range(self.n):
                if array[row][col] == 0:
                    empty_squares.append((row, col))
        return empty_squares
    

    # random__AI.make_move bekommt alle gültigen Züge von 
    # random_AI.get_empty_squares() und wählt aus dieser Liste einen 
    # zufälligen Zug aus und gibt diesen zurück.
    def make_move(self, board): 
        self.m, self.n = board.array.shape # die Spielbrettgröße wird dem Board-Objekt entnommen
        x,y = random.choice(self.get_empty_squares(board.array)) # ein zufälligen Tupel aus der Liste der möglichen Züge wird ausgewählt
        return (x,y)

#----------------------------------------------------------------------  

class Minimax(Player):
    def __init__(self, player_number, name="Minimax"):
        super().__init__(name, player_number)
        self.other_player_number = 2 if self.player_number == 1 else 1


    # minimax.get_empty_squares() bekommt ein Board.array() übergeben 
    # und gibt eine Liste  mit allen (Zeile, Spalte)-Koordinaten, die 
    # gleich 0 sind, zurück.
    def get_empty_squares(self, array):
        empty_squares = []
        for row in range(self.m):
            for col in range(self.n):
                if array[row][col] == 0:
                    empty_squares.append((row, col))
        return empty_squares


    # minimax.get_tree() bekommt ein Board-Objekt und eine Tiefe übergeben
    # und gibt ein Dictionary mit allen möglichen ZÜgen als key's und alle
    # Endresultate als Arrays innerhalb einer Liste zurück.
    def get_tree(self, board, depth=4):
        tree = defaultdict(list)

        if np.count_nonzero(board.array == 0) > 16: # wenn noch zu wenig Züge gespielt wurden, zieht minimax nur die inneren Felder des Arrays in betracht, als mögliche Spielzüge
            self.small_board = Board(self.m-2, self.n-2, self.k-1) # generiert ein neues Board-Objekt mit einem kleineren Array
            self.small_board.array = board.array[1:self.m-1, 1:self.n-1]
            self.m, self.n, self.k = self.m-2, self.n-2, self.k-1
            empty_squares = list(map(lambda x: (x[0]+1, x[1]+1), self.get_empty_squares(self.small_board.array))) # generiert alle möglichen Züge innerhalb des kleineren Arrays und passt die Züge an das große Board an (row+1, col+1)
            depth = len(empty_squares) # wenn das Spielfeld klein ist, rechnet minimax den kompletten Spielbaum aus
        else: # wenn mehr als 8 Züge gespielt wurden, betrachtet Minimax das gesamte Spielbrett
            empty_squares = self.get_empty_squares(board.array) # eine Liste mit allen möglichen Zügen als Tupel (Zeile, Spalte) wird generiert
            depth = len(empty_squares) if len(empty_squares) < depth else depth # wenn es weniger Züge als die eingestellte depth=4 gibt, wird diese angepasst
            
        all_permutations = list(permutations(empty_squares, depth)) # generiert eine Liste mit allen möglichen Spielkombinationen bis zur eingestellten Tiefe

        for permutation in all_permutations:
            moves = []
            board_copy = Board() # erstellt ein neues Board-Objekt, auf dem die Spiele simuliert werden
            board_copy.array = board.array.copy()
            AI_to_move = True # gibt an ob der nächste Zug von Minimax simuliert wird, oder der von Gegner
            start_x, start_y = permutation[0] 
            for row, col in permutation:
                board_copy.array[row][col] = self.player_number if AI_to_move else self.other_player_number # überschreibt ein Feld mit der entsprechenden Spieler-Nummer (simuliert einen Zug)
                AI_to_move = not AI_to_move # der jeweils nächste Spieler ist mit dem nächsten Zug an der Reihe 

                if np.count_nonzero(board.array == 0) > 20: # überprüft ob das kleine Board betrachtet wird, oder nicht
                    if self.small_board.has_won() != 0: # wenn schon ein Spieler gewonnen hat, wird die Simulation abgebrochen
                        break
                else:
                    if board_copy.has_won() != 0:
                        break
            moves.append(board_copy.array.copy())
            tree[(start_x, start_y)].append(moves.copy())
        return tree

        
    # minimax.minimax bekommt ein Board-Objekt übergeben und gibt ein Tupel 
    # (Zeile, Spalte) zurück. Zuerst wird der Spielbaum generiert und dann 
    # wird jedes Resultat bewertet.
    def minimax(self, board):
        maximize = True if self.player_number == 1 else False # entscheidet ob die kleinste oder höchste Bewertung ausgewählt wird
        tree = self.get_tree(board) # generiert den Spielbaum
        results = defaultdict(int)
        for key in tree.keys():
            results[key] = 0
            for array in tree[key]:
                if np.count_nonzero(board.array == 0) > 16: # überprüft, ob auf kleinem Array gespielt wird
                    self.small_board.array = array[0][1:self.m+1, 1:self.n+1] # das zu bewertende Array wird aus dem Spielbaum geholt
                    if self.small_board.has_won() == 0: # bewertet das Array
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
        index = list(results.values()).index(max(results.values())) if maximize else list(results.values()).index(min(results.values())) # der Index der größten bzw. kleinsten Bewertung 
        return list(results.keys())[index] # eine Liste mit dem Tupel mit der höchsten oder niedrigsten Bewertung wird zurückgegeben
    

    # minimax.one_move_win() bekommt ein Board-Objekt übergeben und überprüft,
    # ob minimax oder der Gegner im nächsten Zug gewinnen kann und gibt dann 
    # das entsprechende Feld als Tupel (Zeile, Spalte) zurück. Ist dies
    # nicht der Fall, wird nichts zurückgegeben
    def one_move_win(self, board):
        board_copy = Board(self.m, self.n, self.k) # ein neuen Board-Objekt wird zum simulieren erstellt
        board_copy.array = board.array.copy() 

        empty_squares = self.get_empty_squares(board.array)
        for player_number in [self.player_number, self.other_player_number]:
            for row,col in empty_squares:
                board_copy.array[row][col] = player_number
                if board_copy.has_won() == player_number:
                    return (row, col)
                board_copy.array[row][col] = 0


    # minimax.make_move() bekommt ein Board-Objekt übergeben und gibt ein
    # Tupel (Zeile, Spalte) zurück. 
    def make_move(self, board):
        self.m, self.n = board.array.shape # Anzahl der Zeilen und Spalten werden dem Board-Objekt entnommen
        self.k = board.k 
        if self.one_move_win(board) != None: # überprüft, ob Minimax oder der Gegner im nächsten Zu gewinnen kann und spielt gegebenenfalls dorthin
            x,y = self.one_move_win(board)
        elif board.array[self.m // 2][self.n // 2] == 0: # überprüft, ob das mittlere Feld frei ist und spielt gegebenenfalls dorthin
            x,y = self.m // 2, self.n // 2
        elif np.count_nonzero(board.array == 0) == 24: # falls Minimax den insgesamt zweiten Zug macht, wird so nah in die Mitte gespielt, wie möglich
            smaller_board_squares = list(product(np.arange(self.m-self.k, self.k), np.arange(self.m-self.k, self.k))) # Variable mit allen Zügen auf dem inneren Spielfeld 
            smaller_board_squares = [i for i in smaller_board_squares if board.array[i[0]][i[1]] == 0] # nur die Züge die tatsächlich möglich sind werden in einer Liste gespeichert
            x,y = smaller_board_squares[random.randint(0,len(smaller_board_squares)-1)] # ein zufälliges Tupel wird aus der Liste gewählt              
        else:
            x,y = self.minimax(board)
         
        return (x,y)

#----------------------------------------------------------------------

class MCTS(Player):
    def __init__(self, player_number, parent=None, parent_action=None, name="MCTS"):
        super().__init__(name, player_number)
        self.parent = parent
        self.parent_action = parent_action
        self.other_player_number = 1 if self.player_number == 2 else 2
        self.children = []
        self.number_of_visits = 0
        self.results = 0


    # MCTS.make_move() bekommt ein Board-Objekt übergeben und gibt ein
    # Tupel (Zeile, Spalte) zurück.
    def make_move(self, board):
        self.m, self.n = board.array.shape
        self.k = board.k
        self.mcts_to_move = True # zeigt, ob der MCTS als nächsten dran ist, oder der Gegner
        self.children = []
        self.number_of_visits = 0 # Anzahl, wie häufig diese Node betrachtet wurde
        self.results = 0 # Bewertung der Node
        self.array_edges = np.concatenate([board.array[0,:-1], board.array[:-1,-1], board.array[-1,::-1], board.array[-2:0:-1,0]]) # die Einträge an den Rändern des aktuellen Board.array
        if not self.array_edges.any() and not board.array[1:self.m-1, 1:self.n-1].all(): # überprüft ob noch nicht auf den Rändern gespielt wurde
            self.small_board = True
            self.board = Board(self.m-2, self.n-2, self.k-1) # erschafft ein neues Board-Objekt mit kleinerem Array
            self.board.array = board.array[1:self.m-1, 1:self.n-1]
        else:
            self.small_board = False 
            self.board = Board(self.m, self.n, self.k) # erschafft ein neues Board-Objekt mit dem selben Array wie das aktuelle Spielbrett
            self.board.array = board.array.copy()
        self.untried_actions = self.get_empty_squares(self.board.array) # alle noch spielbaren Felder als Tupel (Zeile, Spalte) in einer Liste
        self.all_actions = self.untried_actions.copy()
        selected_node = self.best_action() # sucht nach dem bestmöglichen Zug
        x,y = selected_node.parent_action
        if self.small_board: # bei kleinem Board werden die Koordinaten wieder ans große Board angepasst
            x += 1
            y += 1
        return (x,y)


    # minimax.get_empty_squares() bekommt ein Board.array() übergeben 
    # und gibt eine Liste  mit allen (Zeile, Spalte)-Koordinaten, die 
    # gleich 0 sind, zurück.
    def get_empty_squares(self, array):
        empty_squares = []
        for row in range(array.shape[0]):
            for col in range(array.shape[1]):
                if array[row][col] == 0:
                    empty_squares.append((row, col))
        return empty_squares


    # MCTS.expand() erstellt alle möglichen Child-Nodes zum aktuellen 
    # Spielstand.
    def expand(self):
        for _ in range(len(self.untried_actions)):
            action = self.untried_actions.pop() #wählt einen der möglichen Züge aus
            array_copy = self.board.array.copy()
            array_copy[action[0]][action[1]] = self.player_number # simuliert den ausgewählten Zug auf dem kopierten Array
            child_node = MCTS(player_number=self.player_number, parent=self, parent_action=action, name="MCTS") # erstellt eine Child-Node als MCTS-Objekt 
            child_node.mcts_to_move = False 
            child_node.board = Board(self.board.m, self.board.n, self.board.k) # erstellt ein neues Board-Objekt mit dem Array der Child-Node
            child_node.board.array = array_copy
            child_node.m, child_node.n = child_node.board.array.shape
            child_node.k = self.board.k
            self.children.append(child_node) # die Liste mit allen Child-Nodes wird um das aktuelle Child erweitert


    # MCTS.is_terminal_board() gibt True, falls ein Spieler gewonnen hat,
    # anderenfalls False    
    def is_terminal_board(self):
        if self.board.has_won() == 0:
            return False
        return True

    
    # MCTS.rollout() simuliert alle Züge eines Spiel-Durchgangs und gibt
    # eine Bewertung mittels MCTS.game_result() zurück.
    def rollout(self):
        current_rollout_state = self.board
        while current_rollout_state.has_won() == 0 and len(self.get_empty_squares(current_rollout_state.array)) != 0: # überprüft, ob noch gespielt werden kann
            possible_moves = self.get_empty_squares(current_rollout_state.array) # generiert alle möglichen Züge innerhalb der Simulation
            action = self.rollout_policy(possible_moves) # wählt einen zufälligen Zug mittels MCTS.rollout_policy() aus
            self.move(action) # simuliert den vorher ausgewählten Zug
            self.mcts_to_move = not self.mcts_to_move # entscheidet, welcher Spieler als nächstes einen Zug machen muss
        return self.game_result()
    
    
    # MCTS.move bekommt eine Feld-Koordinate als Tupel (Zeile, Spalte) 
    # übergeben und simuliert einen Zug von dem Spieler, der einen Zug
    # machen muss.
    def move(self, position):
        if self.mcts_to_move == True:
            self.board.array[position[0]][position[1]] = self.player_number
        else:
            self.board.array[position[0]][position[1]] = self.other_player_number


    # MCTS.backpropagate() bekommt eine Bewertung übergeben und aktualisiert
    # die number_of_visits und results von sich bzw. der Parent-Node.
    def backpropagate(self, result):
        self.number_of_visits += 1
        self.results += result
        self.parent.number_of_visits += 1


    # MCTS.rollout_policy() bekommt eine Liste mit allen möglichen Zügen
    # übergeben und gibt einen zufälligen Zug davon zurück.
    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    
    # MCTS.game_result() bewertet die Simulation und gibt diese Bewertung
    # zurück.
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


    # MCTS_two_move_win() bekommt ein Array übergeben und überprüft, ob MCTS
    # oder der Gegner innerhalb von zwei Zügen gewinnen kann. Ist dies der
    # Fall wird entsprechende player_number zurückgegeben
    def two_move_win(self, array):
        board_copy = Board(self.parent.board.m, self.parent.board.n, self.parent.board.k)
        board_copy.array = array.copy()
        empty_squares = self.get_empty_squares(board_copy.array)
        for player_number in [self.player_number, self.other_player_number]:
            for row,col in empty_squares:
                board_copy.array[row][col] = player_number
                if board_copy.has_won() == player_number:
                    return player_number
                board_copy.array[row][col] = 0

   
   # MCTS.best_child() wählt die beste Child-Node mittels eines Algorithmus' 
   # aus und gibt dieses zurück.
    def best_child(self):   
        for child in self.children:
            if child.number_of_visits == 0 and child.results == 0: # überprüft, ob eine Kind noch gar nicht betrachtet wurde und gibt gegebenenfalls dieses zurück
                return child
        choices_weights = [(c.results / c.number_of_visits) + np.sqrt(2) * np.sqrt((np.log(self.number_of_visits) / c.number_of_visits)) for c in self.children] # Algorithmus zur Bestimmung der bestens Child-Node wird bei jeder Child-Node angewandt
        return self.children[np.argmax(choices_weights)] # gibt die Child-Node mit der höchsten Bewertung zurück

    
    # MCTS.best_action gibt nach 1000 Simulationen die beste Cild-Node zurück.
    def best_action(self):
        # simulation_no = int(np.sin(np.pi / (len(self.all_actions) + 1)) * 1000)
        simulation_no = 1000
        # simulation_no = (25000 - (1000 * len(self.all_actions))) // len(self.all_actions)
        self.expand()
        for _ in range(simulation_no): 
            best_child = self.best_child() # bestimmt die beste Child-Node
            # array_copy = best_child.board.array.copy() 
            reward = best_child.rollout() # Simuliert über die beste Child-Node
            # best_child.board.array = array_copy
            best_child.backpropagate(reward) # aktualisiert die number_of_visits und results
        return self.best_child()


#---------------------------------------------------------------------- 

# game_menu() startet ein Auswahlfenster, indem der User die Spiel-
# einstellungen selber wählen kann. Dann wird das Spiel mit den
# entsprechenenden Einstellungen gestartet.
def game_menu():
    print(" +----------------------------------+\n",
          "|            Welcome to            |\n",
          "|           m,n,k-Game!!           |\n",
          "+----------------------------------+")
    
    print("Select a game mode.")
    print("1: Singleplayer \n2: Multiplayer \n3: AI vs. AI" )
    game_mode = input("Selection: ") # der User wählt das Spiel-Format
    while game_mode != "1" and game_mode != "2" and game_mode != "3": # Eingabe wird überprüft und falls nicht legitim muss neu gewählt werden
        game_mode = input("Please select a valid game mode: ")

    if game_mode == "1": # überprüft, ob das Singleplayer-Format gewählt wurde
        user_number = input("Do you want to play as Player 1 or Player 2? ") # der User wählt seine Spieler-Nummer
        while user_number != "1" and user_number != "2": # Eingabe wird überprüft und falls nicht legitim muss neu gewählt werden
            user_number = input("Please enter a valid number: ") 
        ai_number = 1 if int(user_number) == 2 else 2 # die player_number der KI wird anhand der vorherigen Eingabe gewählt
        user_name = input("Enter your name. ") # der User wählt seinen Spieler-Namen
        user = Player(user_name, int(user_number))
        print("Select an AI you want to challenge.") 
        print("1: random AI \n2: Minimax \n3: Monte Carlo tree search")
        ai_choice = input("Selection: ") # der User wählt gegen welche KI er spielen möchte
        while ai_choice != "1" and ai_choice != "2"  and ai_choice != "3":  # Eingabe wird überprüft und falls nicht legitim muss neu gewählt werden
            ai_choice = input("Please enter a valid number: ")
        if ai_choice == "1": # die KI wird anhand der Eingabe gewählt
            ai = random_AI(ai_number)
        elif ai_choice == "2":
            ai = Minimax(ai_number)
        else:
            ai = MCTS(ai_number)
 
        player1 = user if user_number == "1" else ai # player1 wird derjenige, mit Spieler-Nummer 1
        player2 = user if player1 == ai else ai

    elif game_mode == "2": # überprüft, ob das Multiplayer-Format gewählt wurde
        print("Enter your names.") 
        player1 = Player(input("Player 1: "), 1) # Spieler 1 wählt seinen Namen
        player2 = Player(input("Player 2: "), 2) # Spieler 2 wählt seinen Namen

    else: # wenn weder Singleplayer noch Multiplayer gewählt wurde, wurde "AI vs. AI" gewählt
        print("Select AI's that should battle each other.") 
        print("1: random AI \n2: Minimax \n3: Monte Carlo tree search")

        for ai_number in [1,2]:
            player = "ai" + str(ai_number) # der name einer dynamischen Variable wird definiert
             
            ai = input("First AI:" ) if ai_number == 1 else input("Second AI:" ) # je nach ai_number wird nach der ersten oder zweiten KI gefragt
            while ai != "1" and ai != "2"  and ai != "3":  # Eingabe wird überprüft und falls nicht legitim muss neu gewählt werden
                ai = input("Please enter a valid number: ")
            if ai == "1": # die KI wird anhand der Eingabe gewählt
                globals()[player] = random_AI(ai_number) # die globale Symbol-Tabelle wird aufgerufen und eine neue dynamische Variable mit dem vorher in "Player" definierten Namen wird definiert 
            elif ai == "2":
                globals()[player] = Minimax(ai_number)
            else:
                globals()[player] = MCTS(ai_number)  
        player1 = ai1 # player1 wird zur dynamischen Variable, in der die erste KI definiert wurde
        player2 = ai2 # player2 wird zur dynamischen Variable, in der die zweite KI definiert wurde



    print("Enter the board size.") # im folgenden wählt der User die Anzahl der Zeilen, Spalten und Steine, die man zum gewinnen in einer Reihe braucht
    m = 0
    while not (type(m) == int and (0 < m < 100)): # Eingabe wird überprüft und falls nicht legitim muss neu gewählt werden, im erstne Durchlauf wird die Schleife immer aufgerufen
        try: # an dieser Stelle ist try/except nötig da int(input()) einen Error hervorrufen kann
            m = int(input("Number of rows: ")) # der User wählt die Anzahl der Zeilen (auf 99 begrenzt)
            if not (type(m) == int and (0 < m < 100)): # Eingabe wird überprüft und falls nicht legitim muss wird ein Fehler ausgegeben und es muss erneut gewählt werden
                assert False
        except:
            print("Please enter a valid number!")

    n = 0
    while not (type(n) == int and (0 < n < 27)):
        try:
            n = int(input("Number of columns: ")) # der User wählt die Anzahl der Spalten (auf 26 begrenzt)
            if not (type(n) == int and (0 < n < 27)):
                assert False
        except:
            print("Please enter a valid number!")

    k = 0
    while not (type(k) == int and (0 < k < max(m,n)+1)):
        try:
            k = int(input("Number of stones you need in a row to win: ")) # der User wählt die Anzahl der Steinen, die zum gewinnen in einer Reihe liegen müssen (auf das Maximum von Anzahl Spalten und Anzahl Zeilen begrenzt)
            if not (type(k) == int and (0 < k < max(m,n)+1)):
                assert False
        except:
            print("Please enter a valid number!")


    game = Game(player1, player2, Board(m,n,k), m,n,k) # ein Game-Objekt mit den vom User vorher ausgewählten Einstellungen wird erstellt 
    game.start() # das Spiel wird gestartet



if __name__ == "__main__":
    game_menu() # beim starten des Programm wird zuerst das Game-Menü geöffnet





    
############################################################
# TO DO's
# dokumentieren
# Player.Make_move für alle m,n,k anpassen
# game Menü - input Fehler nicht zulassen
# py to exe
############################################################



