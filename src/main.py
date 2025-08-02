from baseline import run_baseline
from sarsa_agent import SARSAAgent
from sarsa_runner import run_sarsa
import matplotlib.pyplot as plt
from collections import defaultdict
import os, json, csv

def plot_results(strategy_names, energy_all, latency_all, sarsa_updates, sarsa_action_counts):
    avg_energy = [sum(e) / len(e) for e in energy_all]
    avg_latency = [sum(l) / len(l) for l in latency_all]

    # Reward curve
    rewards = [r for _, _, _, r in sarsa_updates]
    avg_rewards = [sum(rewards[:i+1]) / (i+1) for i in range(len(rewards))]

    # Pie chart values from SARSA
    labels = list(sarsa_action_counts.keys())
    sizes = [sarsa_action_counts[label] for label in labels]

    # Plot 3 charts in one figure
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Bar Chart – Energy & Latency
    width = 0.35
    x = range(len(strategy_names))
    axs[0].bar([i - width/2 for i in x], avg_energy, width=width, label='Energy')
    axs[0].bar([i + width/2 for i in x], avg_latency, width=width, color='orange', label='Latency')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(strategy_names)
    axs[0].set_title("Avg Energy & Latency")
    axs[0].legend()

    # 2. Line Chart – Reward Curve
    axs[1].plot(avg_rewards, color='green')
    axs[1].set_title("SARSA Avg Reward Over Time")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Average Reward")

    # 3. Pie Chart – SARSA Decision Distribution
    axs[2].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    axs[2].axis('equal')  # Equal aspect ratio for circle
    axs[2].set_title("SARSA Action Distribution")

    plt.tight_layout()
    plt.show()

def main():
    strategy_names = ["random", "threshold"]
    energy_all, latency_all = [], []

    for strategy in strategy_names:
        e, l = run_baseline(strategy, episodes=2000)
        energy_all.append(e)
        latency_all.append(l)

    print("\nTraining SARSA agent...")
    agent = SARSAAgent(
        epsilon=1.0,
        min_epsilon=0.01,
        decay_rate=0.999,
        alpha=0.25,
        gamma=0.9
    )

    sarsa_energy, sarsa_latency, sarsa_updates = run_sarsa(agent, episodes=2000)

    strategy_names.append("sarsa")
    energy_all.append(sarsa_energy)
    latency_all.append(sarsa_latency)

    # --- Table ---
    print("\nFINAL AVERAGE METRICS\n")
    print(f"{'Strategy':<12} | {'Energy (J)':<12} | {'Latency (s)':<12}")
    print("-" * 40)

    # Collect for saving
    run_results = {}
    for name, energy, latency in zip(strategy_names, energy_all, latency_all):
        e = sum(energy) / len(energy)
        l = sum(latency) / len(latency)
        print(f"{name:<12} | {e:<12.2f} | {l:<12.2f}")
        run_results[name] = {"energy": round(e, 2), "latency": round(l, 2)}

    log_file = "run_results.json"

    # Prepare compact result per run
    compact_result = {
        name: [round(sum(energy) / len(energy), 2), round(sum(latency) / len(latency), 2)]
        for name, energy, latency in zip(strategy_names, energy_all, latency_all)
    }

    history = {}
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                content = f.read().strip()
                if content:
                    history = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            print("⚠️ Warning: Failed to decode existing JSON. Starting fresh.")
            history = {}

    # Add new run as run_N
    run_id = f"run_{len(history)+1}"
    history[run_id] = compact_result

    # Save back
    with open(log_file, "w") as f:
        json.dump(history, f, indent=4)

    # --- CSV Export ---
    csv_file = "run_results.csv"
    csv_exists = os.path.exists(csv_file)

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Write header if file is new
        if not csv_exists:
            writer.writerow([
                "Run ID",
                "Random Energy", "Random Latency",
                "Threshold Energy", "Threshold Latency",
                "SARSA Energy", "SARSA Latency"
            ])

        # Write a single row with all strategy results
        row = [run_id]
        for strategy in ["random", "threshold", "sarsa"]:
            avg_e, avg_l = compact_result[strategy]
            row.extend([avg_e, avg_l])
        writer.writerow(row)

    # --- Plotting ---
    # plot_results(strategy_names, energy_all, latency_all, sarsa_updates, agent.get_action_counts())


if __name__ == "__main__":
    main()