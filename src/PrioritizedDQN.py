import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


class PrioritizedExperienceReplay:
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha  
        self.beta = beta 
        self.beta_increment = beta_increment
        self.max_priority = 1.0 

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        sampling_probs = priorities ** self.alpha
        sampling_probs /= sampling_probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=sampling_probs)
        samples = [self.memory[i] for i in indices]

        weights = (len(self.memory) * sampling_probs[indices]) ** (-self.beta)
        weights /= weights.max()  

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
            torch.LongTensor(indices),
            torch.FloatTensor(weights)
        )

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = (abs(error) + 1e-5)
        self.max_priority = max(self.priorities)

    def __len__(self):
        return len(self.memory)


class PriorityQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for units in hidden_units:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.ReLU())
            prev_dim = units
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class PrioritizedDQNAgent:
    def __init__(self, state_dim, action_dim, hidden_units,
                 gamma=0.99, epsilon=0.7, epsilon_decay=0.995, 
                 alpha=0.6, beta=0.4, beta_increment=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.05
        self.BATCH_SIZE = 128

        self.policy_net = PriorityQNetwork(state_dim, action_dim, hidden_units)
        self.target_net = PriorityQNetwork(state_dim, action_dim, hidden_units)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = PrioritizedExperienceReplay(alpha=alpha, beta=beta, beta_increment=beta_increment)
        
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
        
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.BATCH_SIZE)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q = rewards + self.gamma * self.target_net(next_states).max(1)[0] * (1 - dones)
        
        td_errors = (target_q - current_q).abs().detach().cpu().numpy()
        loss = (weights * (current_q - target_q).pow(2)).mean()  

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)
        
        return loss.item()