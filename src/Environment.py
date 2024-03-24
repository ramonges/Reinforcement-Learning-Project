import numpy as np


class TradingEnvironment:
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001):
        """
        Initialize the trading environment
        """
        self.data = data
        self.state_space = data.shape[1]  # This should match the number of features used to represent a state
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
        """
        Take a step in the trading environment: buy, sell, or hold
        """      
        # Ensure action is within a valid range
        action = np.clip(action, -1, 1)

        # Calculate the number of shares bought/sold based on the action
        delta_position = action * self.balance  # This assumes all-in on each action

        # Get the current price from the dataset to calculate changes in portfolio value
        current_price = self.data.iloc[self.current_step]['Dernier']  # Assuming 'close' is a column in your dataset
        next_step = min(self.current_step + 1, len(self.data) - 1)  # Ensure we don't go past the end of the dataset
        next_price = self.data.iloc[next_step]['Dernier']

        # Update position and balance
        change_in_value = delta_position * (next_price - current_price) / current_price
        self.balance += change_in_value - (abs(delta_position) * self.transaction_cost)
        self.portfolio_value = self.balance  # This could be more complex if managing multiple positions
        self.position += delta_position

        self.current_step = next_step

        # Check if we're at the end
        if self.current_step >= len(self.data) - 1:
            self.done = True

        # Here, the reward could be the change in portfolio value, or some other metric
        reward = change_in_value - (abs(delta_position) * self.transaction_cost)
        
        # Record this step
        self.history.append((self.current_step, self.position, self.portfolio_value, reward))

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