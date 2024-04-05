## Reinforcement-Learning-Project

# Abstract
This project aims to develop a reinforcement learning strategy for trading through the development of a PPO algorithm on seasonality stocks such as Gaz, Oil, other commodities, and equity seasonal stocks such as Michelin for example. Our hypothesis is that for an RL Agent it could be easier to learn pattern from a seasonal stock than others.

Therefore, our objective is to design and implement an automated trading system capable of navigating the highly volatile and seasonal gas & commodity market to make profitable trades. We leverage the potential of the PPO algorithm, such as its stability and efficiency in policy updates, to maximize the expected cumulative reward while effectively managing market risk.

The architcture will be around three modules: 1. Environment 2. Feature engineering 3. PPO Agent 

The trading environment simulates the gas market, incorporating elements like price vectors (Open, High, Low, Close), technical indicators (MACD, RSI)  to create a realistic and dynamic setting for the agent to interact with. 

As a baseline we will have an epsilon greedy algorithm. 

The PPO agent, designed with a focus on adaptability and learning efficiency, employs neural networks to approximate both policy and value functions, enabling it to make informed trading decisions based on the state of the market.



