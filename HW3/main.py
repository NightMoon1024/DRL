import numpy as np
import matplotlib.pyplot as plt
import os

# Set fixed reward values for 10 arms
fixed_rewards = np.array([1.0, 0.8, 0.6, 0.4, 0.2,
                          0.0, -0.2, -0.4, -0.6, -0.8])
n_arms = len(fixed_rewards)
n_steps = 1000
n_runs = 200
epsilon = 0.1
temperature = 0.5
c = 2  # UCB confidence level

# Create shared rewards for all runs
true_rewards = np.tile(fixed_rewards, (n_runs, 1))

def run_bandit(strategy):
    all_rewards = np.zeros((n_runs, n_steps))
    exploration_flags = np.zeros((n_runs, n_steps))
    arm_counts = np.zeros((n_runs, n_arms))

    for run in range(n_runs):
        Q = np.zeros(n_arms)
        N = np.zeros(n_arms)

        for t in range(n_steps):
            explore = False

            if strategy == "epsilon_greedy":
                if np.random.rand() < epsilon:
                    action = np.random.randint(n_arms)
                    explore = True
                else:
                    action = np.argmax(Q)

            elif strategy == "ucb":
                if t < n_arms:
                    action = t
                else:
                    ucb_values = Q + c * np.sqrt(np.log(t + 1) / (N + 1e-5))
                    action = np.argmax(ucb_values)

            elif strategy == "softmax":
                exp_q = np.exp(Q / temperature)
                probs = exp_q / np.sum(exp_q)
                action = np.random.choice(n_arms, p=probs)
                explore = True

            elif strategy == "thompson":
                samples = np.random.normal(Q, 1 / (np.sqrt(N + 1e-5)))
                action = np.argmax(samples)

            else:
                raise ValueError("Unknown strategy")

            reward = np.random.normal(true_rewards[run][action], 1.0)
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]

            all_rewards[run][t] = reward
            exploration_flags[run][t] = int(explore)
            arm_counts[run][action] += 1

    return all_rewards, exploration_flags, arm_counts

strategies = ["epsilon_greedy", "ucb", "softmax", "thompson"]
results = {}

os.makedirs("plots", exist_ok=True)

# Plot true arm values
plt.figure()
plt.bar(range(n_arms), fixed_rewards)
plt.title("True Reward Values for Arms")
plt.xlabel("Arm")
plt.ylabel("Expected Reward")
plt.grid(True)
plt.savefig("plots/true_rewards.png")
plt.close()

# Run each strategy and collect results
for strategy in strategies:
    rewards, explores, counts = run_bandit(strategy)

    avg_cum_rewards = np.mean(np.cumsum(rewards, axis=1), axis=0)
    avg_explore_rate = np.mean(explores, axis=0)
    total_arm_counts = np.sum(counts, axis=0)

    results[strategy] = {
        "rewards": avg_cum_rewards,
        "explore_rate": avg_explore_rate,
        "arm_counts": total_arm_counts,
    }

    # Plot 1: Cumulative reward
    plt.figure()
    plt.plot(avg_cum_rewards)
    plt.title(f"{strategy.replace('_', ' ').title()} - Cumulative Reward")
    plt.xlabel("Steps")
    plt.ylabel("Average Cumulative Reward")
    plt.grid(True)
    plt.savefig(f"plots/{strategy}_reward.png")
    plt.close()

    # Plot 2: Exploration rate
    plt.figure()
    plt.plot(avg_explore_rate)
    plt.title(f"{strategy.replace('_', ' ').title()} - Exploration Rate")
    plt.xlabel("Steps")
    plt.ylabel("Exploration Rate")
    plt.grid(True)
    plt.savefig(f"plots/{strategy}_explore.png")
    plt.close()

    # Plot 3: Arm selection count
    plt.figure()
    plt.bar(range(n_arms), total_arm_counts)
    plt.title(f"{strategy.replace('_', ' ').title()} - Arm Selection Count")
    plt.xlabel("Arm")
    plt.ylabel("Total Selection")
    plt.grid(True)
    plt.savefig(f"plots/{strategy}_armcount.png")
    plt.close()

# Final comparison plot
plt.figure(figsize=(10, 6))
for strategy in strategies:
    plt.plot(results[strategy]["rewards"], label=strategy.replace('_', ' ').title())
plt.title("Cumulative Reward Comparison")
plt.xlabel("Steps")
plt.ylabel("Average Cumulative Reward")
plt.legend()
plt.grid(True)
plt.savefig("plots/mab_comparison.png")
plt.close()
