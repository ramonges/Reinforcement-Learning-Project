


def train_agent(env, agent, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.predict_action(state)  # Predict the action for the current state
            next_state, reward, done, _ = env.step(action)  # Take the action in the environment
            agent.update_policy(state, action, reward, next_state, done)  # Update the policy
            state = next_state
        print(f"Episode {episode + 1}: Complete")
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
