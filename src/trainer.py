from main import Board, Player, Game, MCTS
from agent import Agent


class Trainer:
    def __init__(self, agent, num_episodes):
        self.agent = agent
        self.num_episodes = num_episodes

    def train_agent(self, game):
        for episode in range(self.num_episodes):
            print(f"Episode {episode + 1}/{self.num_episodes}")

            alphabet = [chr(value) for value in range(97, 123)] # das folgenden dient nur der Anzeige in der Konsole
            row_dict = {}
            col_dict = {}
            for k in range(self.m):
                row_dict[k] = k + 1
            for k in range(self.n):
                col_dict[k] = alphabet[k]

            game_on = True

            # Reset the game for a new episode
            game.board = Board()
            
            # Initial state
            state = game.board
            total_reward = 0

            while game_on:
                for model in [game.player1, game.player2]:
                    # Agent's move
                    action_x, action_y = model.make_move(state)

                    # Store the current state
                    current_state = state

                    # Make the move
                    game.board.array[action_x][action_y] = model.player_number
                    print(f"{model.name} chooses {str(((row_dict[action_x]), col_dict[action_y]))}")
                    game.board.display()

                    # Check if the move results in a win or draw
                    if game.board.has_won() == self.agent.player_number:
                        reward = 1.0
                        game_on = False
                    elif game.board.array.all():
                        reward = 0.5  # Consider a draw as a positive outcome
                        game_on = False
                    else:
                        reward = 0.0

                    total_reward += reward

                    # Get the next state
                    next_state = game.board

                    # Train the agent
                    self.agent.train(current_state, action_x * game.n + action_y, reward, next_state)

                    # Check for the end of the game
                    if reward != 0.0:
                        break

            print(f"Total Reward: {total_reward}")


player1 = MCTS("MCTS", 1)
agent = Agent("Agent", 2, board_size=25)
game = Game(player1, agent)

trainer = Trainer(agent, num_episodes=100)
trainer.train_agent(game)
