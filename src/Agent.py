from src.ActorCritic import Actor, Critic
import torch
import numpy as np
from torch import optim
import tensorflow as tf


class PPOAgent:
    def __init__(self, state_size, action_size, action_bound, lr_actor=1e-4, lr_critic=1e-3, action_std=0.5 , update_epochs=10 , clip_param=0.2 , entropy_beta=0.01):
        """
        Initialize the PPO Agent
        """
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
        """
        Predict the action to take in the current state
        """
        state = self.preprocess_state(state)
        with torch.no_grad():
            action_mean = self.actor(state)
            dist = torch.distributions.Normal(action_mean, self.action_std)  # Assuming self.action_std is defined
            action = dist.sample()
            log_prob = dist.log_prob(action) if return_log_prob else None
        return action.cpu().numpy(), log_prob



    def compute_gae(self, next_value, rewards, masks, values):
        """
        Compute the Generalized Advantage Estimation
        """
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * masks[step] - values[step]
            gae = delta + self.gamma * self.tau * masks[step] * gae
            next_value = values[step]
            returns.insert(0, gae + values[step])
        return returns



    def update_policy(self, states, actions, rewards, next_states, dones, old_log_probs , verbose=False):
        """
        Update the policy using PPO algorithm

        Steps:
        1. Calculate advantages
        2. Calculate returns
        3. Calculate old log probabilities
        4. Calculate actor loss
        5. Calculate critic loss
        6. Perform backpropagation
        7. Repeat for multiple epochs
        """
        # Normalize rewards
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-10)
        
        # Convert everything into PyTorch tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        old_log_probs = torch.stack(old_log_probs)

        print("states.shape:", states.shape) if verbose else None
        print("actions.shape:", actions.shape) if verbose else None
        print("rewards.shape:", rewards.shape) if verbose else None
        print("next_states.shape:", next_states.shape) if verbose else None
        print("dones.shape:", dones.shape) if verbose else None
        print("old_log_probs.shape:", old_log_probs.shape) if verbose else None

        for _ in range(self.update_epochs):
            log_probs, state_values, dist_entropy = self.evaluate_actions(states, actions)
            advantages, returns = self.calculate_advantages(rewards, states, next_states, dones, state_values)

            # Calculate actor loss
            print("\nlog_probs.shape:", log_probs.shape) if verbose else None
            old_log_probs = old_log_probs.squeeze()
            old_log_probs = old_log_probs.mean(dim=-1).squeeze()
            print("old_log_probs.shape:", old_log_probs.shape) if verbose else None
            ratio = torch.exp(log_probs - old_log_probs.detach())

            print("ratio.shape:", ratio.shape) if verbose else None
            print("advantages.shape:", advantages.shape) if verbose else None

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



    def calculate_advantages(self, rewards, states, next_states, dones, state_values , verbose=False):
        """
        Calculate the advantages and returns
        """
        with torch.no_grad():
            aggregated_rewards = rewards.mean(dim=-1).squeeze()
            next_state_values = self.critic(next_states).squeeze()
            
            print("aggregated_rewards.shape:", aggregated_rewards.shape) if verbose else None
            print("next_state_values.shape:", next_state_values.shape) if verbose else None

            state_values = state_values.squeeze()

            print("aggregated_rewards.shape:", aggregated_rewards.shape) if verbose else None
            print("next_state_values.shape:", next_state_values.shape) if verbose else None
            print("dones.shape:", dones.shape) if verbose else None
            print("state_values.shape:", state_values.shape) if verbose else None
            
            deltas = aggregated_rewards + self.gamma * next_state_values * (1 - dones) - state_values

            advantages = torch.zeros_like(aggregated_rewards)
            running_add = 0.0
            for t in reversed(range(len(aggregated_rewards))):

                print("\ndeltas[t].shape:", deltas[t].shape) if verbose else None
                print("self.gamma:", self.gamma) if verbose else None
                print("self.tau:", self.tau) if verbose else None
                print("dones[t].shape:", dones[t].shape) if verbose else None
                print("running_add:", running_add) if verbose else None
                print("advantages[t].shape:", advantages[t].shape) if verbose else None

                running_add = deltas[t] + self.gamma * self.tau * (1 - dones[t]) * running_add
                
                print("running_add.shape after computation:", running_add.shape) if verbose else None

                advantages[t] = running_add

            returns = advantages + state_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            return advantages, returns



    def evaluate_actions(self, states, actions , verbose=False):
        """
        Evaluate the actions taken by the agent

        Returns:
        - log_probs    : Log probabilities of the actions
        - state_values : Estimated state values
        - entropy      : Entropy of the action distribution
        """
        states_np = np.array(states)
        states = torch.FloatTensor(states_np)

        actions_np = np.array(actions)
        actions = torch.FloatTensor(actions_np).squeeze(1)

        action_means = self.actor(states)
        state_values = self.critic(states).squeeze(-1)
        
        dist = torch.distributions.Normal(action_means, self.action_std)

        print("actions.shape:", actions.shape) if verbose else None

        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        print("out: log_probs.shape:", log_probs.shape) if verbose else None
        print("out: state_values.shape:", state_values.shape) if verbose else None
        print("out: entropy.shape:", entropy.shape) if verbose else None

        return log_probs, state_values, entropy
    


