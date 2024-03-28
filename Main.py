import pandas as pd
import numpy as np
import ta
from src.DataCleaning import convert_k_m_to_numeric
from src.Features import FeatureEngineer
from src.Environment import TradingEnvironment , sample_policy
from src.Agent import PPOAgent
from src.ActorCritic import Actor, Critic
from src.Training import train_agent , run_simulation
from torchinfo import summary
import wandb



def main():
    # Load and preprocess the data
    gaz = pd.read_csv('Data/Dutch TTF Natural Gas Futures - Données Historiques (1).csv')
    fe_gaz = FeatureEngineer(gaz)
    fe_gaz.apply_preprocessing()
    gaz = fe_gaz.df


    # Initialize the environment
    env = TradingEnvironment(gaz)
    backtest_history = env.run_backtest(sample_policy)



    # Setup wandb for logging
    wandb.init(project="RL Trading",config={
        "episodes"             : 1000,
        "learning_rate_actor"  : 1e-3,
        "learning_rate_critic" : 5e-4
    })
    config = wandb.config


    # Initialise the agent
    agent = PPOAgent(state_size    = env.state_space,             # The number of features
                     action_size   = env.action_space,            # The number of possible actions
                     action_bound  = 1,                           # The bound for the action values
                     lr_actor      = config.learning_rate_actor,  # The learning rate for the actor network
                     lr_critic     = config.learning_rate_critic, # The learning rate for the critic network
                     action_std    = 0.5,                         # The standard deviation for the action distribution
                     update_epochs = 10,                          # The number of epochs for updating the network
                     clip_param    = 0.3,                         # The clip parameter for the PPO algorithm
                     entropy_beta  = 0.05)                        # The entropy beta for the loss function


    # Train the agent
    train_agent(env, agent , episodes=config.episodes)

    # Run the simulation
    run_simulation(env, agent) # not working




if __name__ == '__main__':
    main()