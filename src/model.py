import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from main import Game, Player, Board, MCTS  # Import the game-related classes


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)
    


class QLearningAgent(Player):
    def __init__(self, name, player_number, learning_rate=0.001, discount_factor=0.9, epsilon=0.1):
        super().__init__(name, player_number)
        self.state_size = 5 * 5  # Reduced state size for simplicity
        self.action_size = 5 * 5  # Reduced action size for simplicity
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        try : 
            self.q_network.load_state_dict(torch.load("agent_q_network.pth"))
        except:
            self.q_network = QNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            print("###############")
            return random.choice([i for i in range(self.action_size) if state[i] == 0])
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def update_q_network(self, state, action, reward, next_state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        q_values = self.q_network(state_tensor)
        next_q_values = self.q_network(next_state_tensor)

        best_next_action = torch.argmax(next_q_values).item()
        target = q_values.clone().detach()
        target[0][action] = reward + self.discount_factor * next_q_values[0][best_next_action]

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def make_move(self, board):
        state = self.encode_state(board)
        action = self.select_action(state)
        x, y = divmod(action, board.n)
        return x, y

    def encode_state(self, board):
        # Encode the board state into a 1D array
        encoded_state = [cell for row in board.array for cell in row]
        return [1 if x == self.player_number else -1 if x != 0 else 0 for x in encoded_state]



if __name__ == "__main__":
    # Define the Q-learning agents
    agent1 = QLearningAgent("Agent1", 1)
    agent2 = QLearningAgent("Agent2", 2)
    print(agent1)

    # Train the agents by playing against each other
    for _ in range(10):
        game = Game(agent1, agent2, Board())
        game.start()


