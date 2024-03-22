from src.ActorCritic import Actor, Critic
import torch
import numpy as np
from torch import optim
import tensorflow as tf


class PPOAgent:
    def __init__(self, state_size, action_size, action_bound, lr_actor=1e-4, lr_critic=2e-4):
        self.actor = Actor(state_size, action_size, action_bound)
        self.critic = Critic(state_size, action_size)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def preprocess_state(self, state):
        # Your preprocessing logic here, making sure the state is a float tensor
        return torch.FloatTensor(state).unsqueeze(0)

    def predict_action(self, state):
        state = self.preprocess_state(state)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()

    def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * next_value * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            next_value = values[step]
            returns.insert(0, gae + values[step])
        return returns

    def update_policy(self, states, actions, rewards, next_states, dones):
        advantages, discounted_rewards = self.calculate_advantages(rewards, states, next_states, dones)
        
        # Convert lists to numpy arrays for processing
        states = np.array(states)
        actions = np.array(actions)
        advantages = np.array(advantages)
        discounted_rewards = np.array(discounted_rewards)

        # Update actor
        with tf.GradientTape() as tape:
            # Calculate loss for the actor
            actor_loss = self.calculate_actor_loss(states, actions, advantages)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Update critic
        with tf.GradientTape() as tape:
            # Calculate loss for the critic
            critic_loss = self.calculate_critic_loss(states, discounted_rewards)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * next_value * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            next_value = values[step]
            returns.insert(0, gae + values[step])
        return torch.tensor(returns)
    


