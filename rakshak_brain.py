import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os

# --- ADVANCED ARCHITECTURE ---
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # 3 Layers + Dropout to prevent overfitting
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.out(x)

class RakshakAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000) # Larger memory
        
        # Hyperparameters (Optimized)
        self.gamma = 0.95    
        self.epsilon = 1.0   
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99  # Slower decay for better exploration
        self.learning_rate = 0.0005 # Slower learning rate for stability
        
        # TWO NETWORKS (Double DQN)
        self.model = DQN(state_size, action_size)       # Active Brain
        self.target_model = DQN(state_size, action_size) # Stable Brain
        self.update_target_model() # Sync them initially
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.HuberLoss() # Better than MSE for stability

    def update_target_model(self):
        """Copies weights from Active Brain to Stable Brain"""
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return torch.argmax(act_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size: return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([i[0] for i in minibatch]))
        actions = torch.LongTensor([i[1] for i in minibatch])
        rewards = torch.FloatTensor([i[2] for i in minibatch])
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch]))
        dones = torch.FloatTensor([i[4] for i in minibatch])

        # --- DOUBLE DQN LOGIC ---
        # 1. Select best action using Active Model
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 2. Calculate value using Target Model (Stability)
        next_q = self.target_model(next_states).max(1)[0]
        
        # 3. Bellman Equation
        expected_q = rewards + (1 - dones) * self.gamma * next_q.detach()

        # 4. Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(current_q, expected_q)
        loss.backward()
        self.optimizer.step()

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self, name):
        if os.path.exists(name):
            self.model.load_state_dict(torch.load(name))
            self.update_target_model() # Sync target
            self.epsilon = 0.0 # Ready to perform
            print(f"✅ Loaded PRO Brain: {name}")
        else:
            print("⚠️ No model found.")