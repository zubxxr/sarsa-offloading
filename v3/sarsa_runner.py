from simulator import simulate_task_and_state
from utils import estimate_latency, estimate_energy
from collections import defaultdict

def reward_function(latency, energy):
    return - (0.85 * latency + 0.15 * energy)  # prioritize latency

def run_sarsa(agent, episodes=200):
    energy_results = []
    latency_results = []
    action_counts = defaultdict(int)

    for ep in range(episodes):
        task, state = simulate_task_and_state()
        current_state = agent.discretize_state(state.battery, state.cpu_usage, state.bandwidth)
        action = agent.choose_action(current_state)

        latency = estimate_latency(task, state.bandwidth)
        energy = estimate_energy(task, action)
        reward = reward_function(latency, energy)

        next_task, next_state = simulate_task_and_state()
        next_state_disc = agent.discretize_state(next_state.battery, next_state.cpu_usage, next_state.bandwidth)
        next_action = agent.choose_action(next_state_disc)

        agent.update(current_state, action, reward, next_state_disc, next_action)
        agent.decay_epsilon()

        energy_results.append(energy)
        latency_results.append(latency)
        action_counts[action] += 1

        # print(f"[EP {ep+1}] Battery: {state.battery:.1f}%, CPU: {state.cpu_usage:.1f}%, Bandwidth: {state.bandwidth:.1f}Mbps")
        print(f"   → Action: {action.upper()} | Reward: {reward:.2f} | Energy: {energy:.2f}J | Latency: {latency:.2f}s\n")


    print("\nSARSA ACTION COUNTS:")
    total = sum(action_counts.values())
    for act in agent.actions:
        print(f"  {act:6s} : {action_counts[act]:3d} times ({100 * action_counts[act]/total:.1f}%)")

    return energy_results, latency_results
