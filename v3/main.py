from baseline import run_baseline
from sarsa_agent import SARSAAgent
from sarsa_runner import run_sarsa
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_results(strategy_names, energy_all, latency_all):
    x = range(len(strategy_names))
    avg_energy = [sum(e) / len(e) for e in energy_all]
    avg_latency = [sum(l) / len(l) for l in latency_all]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].bar(x, avg_energy, tick_label=strategy_names)
    axs[0].set_title("Average Energy Usage")
    axs[0].set_ylabel("Energy Units")

    axs[1].bar(x, avg_latency, tick_label=strategy_names, color='orange')
    axs[1].set_title("Average Latency")
    axs[1].set_ylabel("Seconds")

    plt.tight_layout()
    plt.show()

def main():
    strategy_names = ["random", "threshold"]
    energy_all, latency_all = [], []

    for strategy in strategy_names:
        # print(f"\nRunning {strategy.upper()} strategy...")
        e, l = run_baseline(strategy, episodes=200)
        energy_all.append(e)
        latency_all.append(l)

    print("\nTraining SARSA agent...")
    agent = SARSAAgent()
    sarsa_energy, sarsa_latency = run_sarsa(agent, episodes=200)

    strategy_names.append("sarsa")
    energy_all.append(sarsa_energy)
    latency_all.append(sarsa_latency)

    plot_results(strategy_names, energy_all, latency_all)


if __name__ == "__main__":
    main()