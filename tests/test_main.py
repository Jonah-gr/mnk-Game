from src.main import RandomAI, Minimax, MCTS, Board, Game
import pytest

@pytest.mark.parametrize("ai1, ai2",
                        [(RandomAI(1), Minimax(2)),
                         (Minimax(1), MCTS(2)),
                         (MCTS(1), RandomAI(2))])
def test_main(ai1, ai2):
    game = Game(ai1, ai2)
    game.start()