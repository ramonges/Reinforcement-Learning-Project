from src.ActorCritic import Actor, Critic
import torch
import numpy as np
from torch import optim
import tensorflow as tf


class PPOAgent:
    def __init__(self, state_size, action_size, action_bound, lr_actor=1e-4, lr_critic=1e-3, action_std=0.5 , update_epochs=10 , clip_param=0.2 , entropy_beta=0.01):
        self.actor = Actor(state_size, action_size, action_bound)
        self.critic = Critic(state_size)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.action_std = action_std  # Define the standard deviation for action distribution
        self.gamma = 0.99
        self.tau = 0.95
        self.update_epochs = update_epochs
        self.clip_param = clip_param
        self.entropy_beta = entropy_beta

        # # Define the environment and PPO agent parameters
        # state_size = 100  # Assume 100 features for the state
        # action_size = 1  # Buy/sell quantity as a single action
        # action_bound = 1  # Action limit (e.g., normalized quantity between -1 and 1)

    


    def preprocess_state(self, state):
        return torch.FloatTensor(state).unsqueeze(0)

    def predict_action(self, state, return_log_prob=False):
        state = self.preprocess_state(state)
        with torch.no_grad():
            action_mean = self.actor(state)
            dist = torch.distributions.Normal(action_mean, self.action_std)  # Assuming self.action_std is defined
            action = dist.sample()
            log_prob = dist.log_prob(action) if return_log_prob else None
        return action.cpu().numpy(), log_prob


    def compute_gae(self, next_value, rewards, masks, values):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * masks[step] - values[step]
            gae = delta + self.gamma * self.tau * masks[step] * gae
            next_value = values[step]
            returns.insert(0, gae + values[step])
        return returns


    def update_policy(self, states, actions, rewards, next_states, dones, old_log_probs):
        # Normalize rewards
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-10)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        old_log_probs = torch.stack(old_log_probs)
        

        for _ in range(self.update_epochs):
            log_probs, state_values, dist_entropy = self.evaluate_actions(states, actions)
            advantages, returns = self.calculate_advantages(rewards, states, next_states, dones, state_values)

            # Calculate actor loss
            # print("log_probs.shape:", log_probs.shape)
            old_log_probs = old_log_probs.squeeze()
            old_log_probs = old_log_probs.mean(dim=-1).squeeze()
            # print("old_log_probs.shape:", old_log_probs.shape)
            ratio = torch.exp(log_probs - old_log_probs.detach())

            # print("ratio.shape:", ratio.shape)
            # print("advantages.shape:", advantages.shape)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_beta * dist_entropy.mean()

            # Calculate critic loss
            critic_loss = (returns - state_values).pow(2).mean()

            # Perform backprop
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.optimizer_critic.step()

    def calculate_advantages(self, rewards, states, next_states, dones, state_values):
        with torch.no_grad():
            aggregated_rewards = rewards.mean(dim=-1).squeeze()
            next_state_values = self.critic(next_states).squeeze()
            

            # print("Before deltas computation")

            state_values = state_values.squeeze()

            # print("aggregated_rewards.shape:", aggregated_rewards.shape)
            # print("next_state_values.shape:", next_state_values.shape)
            # print("dones.shape:", dones.shape)
            # print("state_values.shape:", state_values.shape)
            

            deltas = aggregated_rewards + self.gamma * next_state_values * (1 - dones) - state_values


            # print("After deltas computation")




            advantages = torch.zeros_like(aggregated_rewards)
            running_add = 0.0
            for t in reversed(range(len(aggregated_rewards))):
                # print("deltas[t].shape:", deltas[t].shape)
                # print("self.gamma:", self.gamma)
                # print("self.tau:", self.tau)
                # print("dones[t].shape:", dones[t].shape)
                # print("running_add:", running_add)
                # print("advantages[t].shape:", advantages[t].shape)

                running_add = deltas[t] + self.gamma * self.tau * (1 - dones[t]) * running_add
                
                # print("running_add.shape after computation:", running_add.shape)
                advantages[t] = running_add

            returns = advantages + state_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            return advantages, returns




    def evaluate_actions(self, states, actions):
        states_np = np.array(states)  # Convert list to numpy array
        states = torch.FloatTensor(states_np)  # Then convert to tensor

        actions_np = np.array(actions)  # Convert list to numpy array
        actions = torch.FloatTensor(actions_np).squeeze(1)  # Then convert to tensor
        # actions = torch.FloatTensor(actions).squeeze(1)

        action_means = self.actor(states)  # Expected shape: [1616, 3]
        state_values = self.critic(states).squeeze(-1)  # Ensure [1616]
        
        dist = torch.distributions.Normal(action_means, self.action_std)

        # Calculate log probabilities and ensure it's reduced to [1616]
        # print("actions.shape:", actions.shape)
        log_probs = dist.log_prob(actions).sum(dim=-1)  # Summing over action dimensions

        # Calculate entropy and ensure it's already correctly shaped [1616]
        entropy = dist.entropy().sum(dim=-1)

        # print("log_probs.shape:", log_probs.shape)  # Should be [1616]
        # print("state_values.shape:", state_values.shape)  # Should be [1616]
        # print("entropy.shape:", entropy.shape)  # Should be [1616]

        return log_probs, state_values, entropy
    


