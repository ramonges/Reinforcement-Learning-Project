import torch
import wandb
from tqdm import tqdm


def train_agent(env, agent, episodes=1000):
    with torch.autograd.set_detect_anomaly(True):
        for episode in tqdm(range(episodes)):
            state = env.reset()
            done = False
            states, actions, rewards, next_states, dones, old_log_probs = [], [], [], [], [], []
            while not done:
                action, log_prob = agent.predict_action(state, return_log_prob=True)  # Updated call


                next_state, reward, done, _ = env.step(action)
                
                # Store experiences
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                old_log_probs.append(log_prob)
                
                state = next_state

            # After collecting experience, update the policy
            agent.update_policy(states, actions, rewards, next_states, dones, old_log_probs)

            # print total rewards for the episode
            total_rewards = sum(sum(rewards)[0])
            wandb.log({"total_rewards": total_rewards})

            # print(f"Episode {episode + 1}: Total Rewards: {total_rewards}")

        print("Training complete")



def run_simulation(env, agent):
    state = env.reset()
    done = False
    total_pnl = 0
    print("Date\t\tAction\tReward\tPortfolio Value")

    while not done:
        action = agent.predict_action(state)
        next_state, reward, done, _ = env.step(action)

        # Action interpretation for logging: -1 (Sell), 1 (Buy), 0 (Hold)
        action_str = "Buy" if action > 0 else "Sell" if action < 0 else "Hold"

        # Assuming 'date' is part of the environment's state
        date = state.index[env.current_step].strftime('%Y-%m-%d')
        print(f"{date}\t{action_str}\t{reward:.2f}\t{env.portfolio_value:.2f}")

        total_pnl += reward
        state = next_state

    print(f"Final PnL: {total_pnl:.2f}")
