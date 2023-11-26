from src.main import RandomAI, Minimax, MCTS, Board, Game

def test_main():
    # Create AI players
    ai1 = RandomAI(1)
    ai2 = Minimax(2)
    ai3 = MCTS(1)

    # Create a game with AI players
    game = Game(ai1, ai2)

    # Start the game
    game.start()

    # Create another game with different AI players
    game2 = Game(ai2, ai3)

    # Start the second game
    game2.start()

    # You can continue adding more tests or variations as needed

if __name__ == "__main__":
    test_main()
