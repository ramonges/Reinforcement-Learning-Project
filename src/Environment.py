import numpy as np


class TradingEnvironment:
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001):
        """
        Initialize the trading environment
        """
        self.data = data
        self.state_space = data.shape[1] # the number of features in the data, could be beneficial to include a window of previous prices
        self.action_space = 3  # For example: buy, sell,
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reset()



    def reset(self):
        """
        Reset the trading environment
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.done = False
        self.position = 0
        self.history = []  # To store trade history
        return self._next_observation()



    def _next_observation(self):
        """
        Get the next observation
        """
        return self.data.iloc[self.current_step]



    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Dernier']
        next_step = min(self.current_step + 1, len(self.data) - 1)
        next_price = self.data.iloc[next_step]['Dernier']

        # Interpret action
        if action == 1:  # Buy
            delta_position = self.balance * 0.1 / current_price  # Buy with 10% of balance
        elif action == 2:  # Sell
            delta_position = -self.position  # Sell all
        else:  # Hold or invalid action
            delta_position = 0

        # Update balance and position based on action
        if delta_position > 0:  # Buying
            cost = delta_position * current_price
            self.balance -= cost + (cost * self.transaction_cost)
            self.position += delta_position
        elif delta_position < 0:  # Selling
            revenue = abs(delta_position) * current_price
            self.balance += revenue - (revenue * self.transaction_cost)
            self.position += delta_position  # delta_position is negative

        # Update portfolio value and reward calculation
        self.portfolio_value = self.balance + (self.position * next_price)
        reward = self.portfolio_value - self.initial_balance - (abs(delta_position) * current_price * self.transaction_cost)

        self.current_step = next_step
        self.done = self.current_step >= len(self.data) - 1

        return self._next_observation(), reward, self.done, {}




    def render(self):
        """
        Print the current state
        """
        print("Step:", self.current_step)
        print("Balance:", self.balance)
        print("Position:", self.position)
        print("Portfolio Value:", self.portfolio_value)



    def run_backtest(self, policy):
        """
        Run a full backtest of the trading environment using the provided policy.
        The policy function should take a state as input and return an action.
        """
        self.reset()
        while not self.done:
            current_state = self._next_observation()
            action = policy(current_state)
            self.step(action)
        return self.history
    


def sample_policy(state):
    # This is a dummy policy that randomly decides to buy, hold, or sell
    return np.random.uniform(-1, 1)