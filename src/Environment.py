import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces


class TradingEnvironment(gym.Env):
    def __init__(self, data, init_balance=10000, init_pos=0, max_steps=360):
        """
        Initialize the trading environment
        """
        super(TradingEnvironment, self).__init__()

        self.data = data
        self.init_balance = init_balance
        self.init_pos = init_pos
        self.max_steps = max_steps

        self.current_step = None
        self.state = None

        self.action_space = spaces.Discrete(3)
        self.action_dic = {0: -1, 1: 0, 2: 1}  # sell  # hold  # buy

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([np.inf, np.inf, np.inf]),
            dtype=np.float32,
        )

        self.reset()

    def reset(self):
        """
        Reset the trading environment
        """
        self.index_step = 0

        self.index_loc = random.randint(0, len(self.data) - self.max_steps - 1)
        self.state = (
            self.init_balance,  # balance
            self.data.iloc[self.index_loc],  # price
            self.init_pos,  # position
        )

        self.done = False

        return self.state

    def reward_function(
        self, balance, new_balance, price, new_price, position, new_position, action
    ):
        """
        Calculate the reward based on the change in balance, position, and trading action.
        """
        # reward = new_balance + new_position * new_price
        reward = new_balance + new_price * new_position - balance - price * position
        return reward

    def step(self, action):
        """
        Take a step in the trading environment: buy, sell, or hold
        state -> action -> reward / new state
        """
        self.index_step += 1

        # State
        balance, price, position = self.state

        if self.action_dic[action] == -1 and position == 0:
            for key, value in self.action_dic.items():
                if value == 0:
                    action = key
        if self.action_dic[action] == 1 and balance < price:
            for key, value in self.action_dic.items():
                if value == 0:
                    action = key

        # New state
        new_balance = balance - price * self.action_dic[action]
        new_price = self.data.iloc[self.index_loc + self.index_step]
        new_position = position + self.action_dic[action]
        self.state = (new_balance, new_price, new_position)

        # Reward
        self.reward = self.reward_function(
            balance, new_balance, price, new_price, position, new_position, action
        )

        # Done
        if self.index_step == self.max_steps:
            self.done = True

        # Info
        self.info = {
            "balance": balance,
            "price": price,
            "position": position,
            "action": self.action_dic[action],
            "new_balance": new_balance,
            "new_price": new_price,
            "new_position": new_position,
            "reward": self.reward,
            "done": self.done,
        }

        return self.state, self.reward, self.done, self.info

    def render(self):
        """
        Print the current state
        """
        balance, price, position = self.state
        print("Step:", self.index_step)
        print("Balance:", balance)
        print(f"Price: {price}")
        print("Position:", position)
        print(f"Reward: {balance + position * price}")

    def run_backtest(self, policy):
        """
        Run a full backtest of the trading environment using the provided policy.
        The policy function should take a state as input and return an action.
        """
        self.reset()
        history = []
        while not self.done:
            action = policy(self.state)
            self.step(action)
            history.append(self.info)
        return history


def random_policy(state):
    return random.choice([0, 1, 2])
