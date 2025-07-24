from baseline import run_baseline
from baseline import run_sarsa
import matplotlib.pyplot as plt
from sarsa_agent import SARSAAgent

def plot_results(strategy_names, energy_all, latency_all):
    x = range(len(strategy_names))

    avg_energy = [sum(e) / len(e) for e in energy_all]
    avg_latency = [sum(l) / len(l) for l in latency_all]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Energy Bar Chart
    axs[0].bar(x, avg_energy, tick_label=strategy_names)
    axs[0].set_title("Average Energy Usage")
    axs[0].set_ylabel("Energy Units")

    # Latency Bar Chart
    axs[1].bar(x, avg_latency, tick_label=strategy_names, color='orange')
    axs[1].set_title("Average Latency")
    axs[1].set_ylabel("Seconds")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    strategy_names = ["random", "threshold"]
    energy_all = []
    latency_all = []

    for strategy in strategy_names:
        print(f"Running: {strategy}")
        energy, latency = run_baseline(strategy, episodes=200)
        energy_all.append(energy)
        latency_all.append(latency)

    print("Running: SARSA")
    agent = SARSAAgent()
    sarsa_energy, sarsa_latency, sarsa_log = run_sarsa(agent, episodes=200)
    energy_all.append(sarsa_energy)
    latency_all.append(sarsa_latency)
    strategy_names.append("sarsa")


    plot_results(strategy_names, energy_all, latency_all)



