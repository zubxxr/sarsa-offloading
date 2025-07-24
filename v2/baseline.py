from simulator import simulate_task_and_state
import numpy as np
from collections import defaultdict

def random_strategy():
    return np.random.choice(["local", "edge1", "edge2", "cloud"])

def threshold_strategy(state):
    if state.battery < 30 or state.cpu_usage > 75:
        if state.bandwidth > 20:
            return "edge1"
        else:
            return "cloud"
    else:
        return "local"

def estimate_latency(task, bandwidth):
    return task.data_size / (bandwidth / 8)  # seconds

def estimate_energy(task, action):
    if action == "local":
        return task.cpu_cycles * 0.005  # arbitrary unit
    elif action == "cloud":
        return task.cpu_cycles * 0.001 + 1.0  # network + transmission
    else:  # edge1 or edge2
        return task.cpu_cycles * 0.002 + 0.5

def run_baseline(strategy_name, episodes=100):
    energy_results = []
    latency_results = []
    action_counts = defaultdict(int)

    for _ in range(episodes):
        task, state = simulate_task_and_state()

        if strategy_name == "random":
            action = random_strategy()
        elif strategy_name == "threshold":
            action = threshold_strategy(state)
        else:
            raise ValueError("Invalid strategy")

        action_counts[action] += 1

        print(f"{strategy_name.upper()} | Action: {action} | Battery: {state.battery:.1f} | CPU: {state.cpu_usage:.1f} | Bandwidth: {state.bandwidth:.1f}")

        latency = estimate_latency(task, state.bandwidth)
        energy = estimate_energy(task, action)

        energy_results.append(energy)
        latency_results.append(latency)

    # Summary of actions
    print(f"\n{strategy_name.upper()} ACTION COUNTS:")
    for act, count in action_counts.items():
        percent = (count / episodes) * 100
        print(f"  {act:<6}: {count:3d} times ({percent:.1f}%)")

    return energy_results, latency_results

def reward_function(latency, energy):
    # You can tune the weights to your liking
    return - (0.7 * latency + 0.3 * energy)

def run_sarsa(agent, episodes=200):
    energy_results = []
    latency_results = []
    q_updates = []
    action_counts = defaultdict(int)

    for ep in range(episodes):
        task, state = simulate_task_and_state()
        current_state = agent.discretize_state(state.battery, state.cpu_usage, state.bandwidth)
        action = agent.choose_action(current_state)

        latency = estimate_latency(task, state.bandwidth)
        energy = estimate_energy(task, action)
        reward = reward_function(latency, energy)

        # Next state
        next_task, next_state = simulate_task_and_state()
        next_state_discrete = agent.discretize_state(next_state.battery, next_state.cpu_usage, next_state.bandwidth)
        next_action = agent.choose_action(next_state_discrete)

        agent.update(current_state, action, reward, next_state_discrete, next_action)

        energy_results.append(energy)
        latency_results.append(latency)
        q_updates.append((ep, current_state, action, reward))
        action_counts[action] += 1

        print(f"[EP {ep+1}] Action: {action} | Reward: {reward:.2f} | Battery: {state.battery:.1f} | CPU: {state.cpu_usage:.1f} | BW: {state.bandwidth:.1f}")

    # Print SARSA action summary
    print(f"\nSARSA ACTION COUNTS:")
    for act, count in action_counts.items():
        percent = (count / episodes) * 100
        print(f"  {act:<6}: {count:3d} times ({percent:.1f}%)")

    return energy_results, latency_results, q_updates