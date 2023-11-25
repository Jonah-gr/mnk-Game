import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from main import Player, Game

# Define the neural network for the agent
# Modify the QNetwork class
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Modify the Agent class
class Agent(Player):
    def __init__(self, name, player_number, board_size):
        super(Agent, self).__init__(name, player_number)
        self.name = "Agent"
        self.player_number = player_number
        self.board_size = board_size
        self.q_network = QNetwork(input_size=board_size, output_size=board_size - 1)  # One less action after one move
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

    # Convert the board state to a format suitable for the neural network
    def preprocess_state(self, board):
        # Convert the board to a flattened numpy array
        state = board.array.flatten()
        return torch.tensor(state, dtype=torch.float32).view(1, -1)

    # Make a move using the neural network
    def make_move(self, board):
        state = self.preprocess_state(board)
        q_values = self.q_network(state)
        valid_moves = [i for i in range(len(q_values[0])) if board.array.flatten()[i] == 0]
        action = np.random.choice(valid_moves)
        x, y = divmod(action, board.n)
        return x, y

    # Train the agent using reinforcement learning
    def train(self, state, action, reward, next_state):
        self.q_network.train()
        self.optimizer.zero_grad()

        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)

        current_q = self.q_network(state)
        next_q = self.q_network(next_state)

        target_q = current_q.clone().detach()
        target_q[0][action] = reward + 0.9 * torch.max(next_q)

        loss = F.mse_loss(current_q, target_q)
        loss.backward()
        self.optimizer.step()


# Example usage
player1 = Player("Player1", 1)
agent = Agent("Agent", 2, 25)

game = Game(player1, agent)
game.start()
