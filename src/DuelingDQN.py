import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


class ExperienceReplay:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.memory)


class DuelingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units):
        super().__init__()
        
       
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_units[0]),
            nn.ReLU()
        )
        
       
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], 1)
        )
        
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], output_dim)
        )

    def forward(self, x):
        features = self.shared_layers(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
       
        return value + (advantages - advantages.mean(dim=1, keepdim=True))


class DuelingAgent:
    def __init__(self, state_dim, action_dim, hidden_units,
                 gamma=0.99, epsilon=0.7, epsilon_decay=0.995, tau=0.01):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.min_epsilon = 0.05
        self.BATCH_SIZE = 128

        self.policy_net = DuelingNetwork(state_dim, action_dim, hidden_units)
        self.target_net = DuelingNetwork(state_dim, action_dim, hidden_units)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ExperienceReplay()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
    
    def select_action(self, state):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()
    
    def update_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.BATCH_SIZE)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)
        
        return loss.item()