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
    # Load the data
    gaz = pd.read_csv('Data/Dutch TTF Natural Gas Futures - Données Historiques (1).csv')
    gaz['Date']=pd.to_datetime(gaz['Date'])
    lists = ['Dernier', 'Ouv.', ' Plus Haut', 'Plus Bas', 'Vol.','Variation %']
    for i in lists: 
        gaz[i] = gaz[i].str.replace(',', '.')


    # Process the data
    gaz['Dernier'] = pd.to_numeric(gaz['Dernier'], errors='coerce')
    gaz['rsi'] = ta.momentum.rsi(gaz['Dernier'], window=14)
    gaz['macd'] = ta.trend.macd_diff(gaz['Dernier'])
    gaz['ema20'] = ta.trend.ema_indicator(gaz['Dernier'], window=20)
    gaz['ema50'] = ta.trend.ema_indicator(gaz['Dernier'], window=50)
    gaz['Vol.'] = gaz['Vol.'].apply(convert_k_m_to_numeric)
    for column in gaz.columns:
        gaz[column] = pd.to_numeric(gaz[column], errors='coerce')
    assert gaz.applymap(np.isreal).all().all(), "Non-numeric data found in the dataset."


    # Feature Engineering
    feature_engineer = FeatureEngineer(gaz)
    feature_engineer.calculate_technical_indicators()  # Calculate RSI, MACD, EMA20, EMA50
    feature_engineer.normalize_features()  # Normalize the features
    features_vector = feature_engineer.construct_feature_vector()  # Get the feature vector


    # Fill NaN in gaz (not using feature engineering yet)
    gaz_copy = gaz.copy()
    gaz_copy.fillna(0, inplace=True)


    # Initialize the environment
    env = TradingEnvironment(gaz_copy)
    backtest_history = env.run_backtest(sample_policy)



    # Setup wandb for logging
    wandb.init(project="RL Trading", config={
        "episodes": 1000,
        "learning_rate_actor": 1e-4,
        "learning_rate_critic": 1e-3
    })
    config = wandb.config


    # Initialise the agent
    agent = PPOAgent(state_size=env.state_space,
                     action_size=env.action_space,
                     action_bound=1,
                     lr_actor=config.learning_rate_actor,
                     lr_critic=config.learning_rate_critic,
                     action_std=0.5,
                     update_epochs=10,
                     clip_param=0.2,
                     entropy_beta=0.01)


    # Train the agent
    train_agent(env, agent , episodes=config.episodes)

    # Run the simulation
    run_simulation(env, agent)




if __name__ == '__main__':
    main()