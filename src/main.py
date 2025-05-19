import os
import random
from collections import deque
import gymnasium as gym
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
from DQN import DQNAgent
from PrioritizedDQN import PrioritizedDQNAgent
from DuelingDQN import DuelingAgent
from DDQN import DoubleDQNAgent

class DQNBenchmark:
    def __init__(self):

        self.env_configs = [
            {"env": "LunarLander-v3", "state": 8, "action": 4},
            {"env": "MountainCar-v0", "state": 2, "action": 3, "reward_shaping": True}
        ]

        self.params = {
            "hidden_layers": [128, 128],
            "max_steps": 500,
            "episodes": 500
        }

        self.agents = {
            "DQN": DQNAgent,
            "DDQN": DoubleDQNAgent,
            "Prioritized": PrioritizedDQNAgent,
            "Dueling": DuelingAgent
        }
        
        
        os.makedirs("benchmark_results", exist_ok=True)

    def _train_agent(self, agent, env, use_shaping):
        rewards = []
        losses = []
        
        for _ in trange(self.params["episodes"], desc="Training"):
            state, _ = env.reset()
            total_reward = 0
            total_loss = 0
            
            for _ in range(self.params["max_steps"]):
                action = agent.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                
                if use_shaping:
                    reward = (
                        10 * (next_state[0] - state[0]) +      
                        0.1 * abs(next_state[1]) +        
                        (-0.01 if action != 1 else 0)  
                    )
                
                agent.memory.push(state, action, reward, next_state, float(done))
                loss = agent.update_model()
                
                state = next_state
                total_reward += reward
                if loss:
                    total_loss += loss
                
                if done or truncated:
                    break
            
            rewards.append(total_reward)
            losses.append(total_loss)
        
        return rewards, losses

    def _plot_results(self, results, env_name):
        for m, i in [('reward', 0), ('loss', 1)]:
            plt.figure(figsize=(10, 6))
            
            for label, data in results.items():
                plt.plot(data[i], label=label)
                
                # Add moving average for rewards
                if m == 'reward' and len(data[i]) >= 100:
                    window = 100
                    cumsum = np.cumsum(np.insert(data[i], 0, 0)) 
                    moving_avg = (cumsum[window:] - cumsum[:-window]) / window
                    plt.plot(np.arange(window-1, len(data[i])), 
                            moving_avg, 
                            linestyle='--',
                            alpha=0.7,
                            label=f'{label} (avg)')
            
            plt.title(f"{env_name} - {m.capitalize()} Comparison")
            plt.xlabel("Episode")
            plt.ylabel(m.capitalize())
            plt.legend()
            plt.grid(True)
            
            filename = f"{env_name.replace('-', '_')}_{m}.png"
            plt.savefig(f"benchmark_results/{filename}", dpi=120)
            plt.close()

    def run(self):
        for env_config in self.env_configs:
            env = gym.make(env_config["env"])
            results = {}
            
            print(f"\nBenchmarking on {env_config['env']}:")
            for agent_name, Agent in self.agents.items():
                print(f"  Testing {agent_name}...")
                
                agent = Agent(
                    env_config["state"],
                    env_config["action"],
                    self.params["hidden_layers"],
                )
                
                rewards, losses = self._train_agent(
                    agent,
                    env,
                    env_config.get("reward_shaping", False)
                )
                results[agent_name] = (rewards, losses)
            
            self._plot_results(results, env_config["env"])
            env.close()

if __name__ == "__main__":
    print("Starting DQN Benchmark...")
    benchmark = DQNBenchmark()
    benchmark.run()
    print("Benchmark completed!")