import yaml
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





def run_sweeping():
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

    # Fill NaN in gaz (not using feature engineering yet)
    gaz_copy = gaz.copy()
    gaz_copy.fillna(0, inplace=True)

    # Initialize the environment
    env = TradingEnvironment(gaz_copy)



    a = wandb.init()

    agent = PPOAgent(state_size    = env.state_space,
                     action_size   = env.action_space,
                     action_bound  = 1,
                     lr_actor      = wandb.config.lr_actor,
                     lr_critic     = wandb.config.lr_critic,
                     action_std    = wandb.config.action_std,
                     update_epochs = wandb.config.update_epochs,
                     clip_param    = wandb.config.clip_param,
                     entropy_beta  = wandb.config.entropy_beta) 

    train_agent(env, agent, episodes=100) # change to 1000 for better results



def do_sweep():
    with open("src/config/bayes_sweep.yaml") as file:
        sweep_config = yaml.load(file , Loader=yaml.FullLoader)
    
    sweep_id = wandb.sweep(sweep=sweep_config, project="RL Trading")
    wandb.agent(sweep_id, function=run_sweeping , count=100)




if __name__ == '__main__':
    do_sweep()