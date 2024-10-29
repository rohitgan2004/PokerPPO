import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PokerRL.eval.rl_br import rl_br
from PokerRL.game.Poker import Poker
from PokerRL.game._.rl_env import RLEnv
from PokerRL.game._.PokerEnvBuilder import PokerEnvBuilder

class PPOAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(PPOAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, action_size)
        self.value_head = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)
        return policy, value

class PPO:
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99, clip_eps=0.2):
        self.agent = PPOAgent(state_size, action_size)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps

    def compute_advantages(self, rewards, values):
        advantages = []
        advantage = 0
        for reward, value in zip(reversed(rewards), reversed(values)):
            td_error = reward - value + self.gamma * advantage
            advantage = td_error
            advantages.insert(0, advantage)
        return advantages

    def update(self, states, actions, rewards, old_probs, values):
        advantages = self.compute_advantages(rewards, values)
        advantages = torch.tensor(advantages)

        for _ in range(4): 
            policy, value = self.agent(states)
            dist = torch.distributions.Categorical(policy)
            new_probs = dist.log_prob(actions)

            ratio = torch.exp(new_probs - old_probs)
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = F.mse_loss(value, rewards)

            loss = policy_loss + value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def create_poker_env():
    env_builder = PokerEnvBuilder(env_cls=rl_br.RLBR,
                                  env_args={
                                      "n_seats": 2,  # 2-player poker
                                      "starting_stack_sizes_list": [100, 100],
                                      "bet_sizes_list_as_frac_of_pot": [0.5, 1.0],
                                      "max_rounds": 100
                                  },
                                  eval_env_args={
                                      "agent_cls": rl_br.RLBR,
                                      "agent_args": {
                                          "training_agent_cls": "DeepCFR",
                                          "lr": 0.01
                                      }
                                  })
    return RLEnv(env_builder)

def train_ppo_agent():
    env = create_poker_env()
    state_size = env.observation_shape[0]
    action_size = env.action_dim

    ppo_agent = PPO(state_size, action_size)

    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        states, actions, rewards, values, old_probs = [], [], [], [], []

        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            policy, value = ppo_agent.agent(state_tensor)
            dist = torch.distributions.Categorical(policy)
            action = dist.sample()

            old_prob = dist.log_prob(action)
            next_state, reward, done, _ = env.step(action.item())

            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            old_probs.append(old_prob)

            state = next_state

        ppo_agent.update(torch.stack(states),
                         torch.stack(actions),
                         torch.tensor(rewards),
                         torch.stack(old_probs),
                         torch.stack(values))

        if episode % 100 == 0:
            print(f"Episode {episode} completed.")

if __name__ == "__main__":
    train_ppo_agent()
