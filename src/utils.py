def estimate_latency(task, bandwidth):
    return task.data_size / (bandwidth / 8)  # seconds

def estimate_energy(task, action):
    if action == "local":
        return task.cpu_cycles * 0.005  # arbitrary unit
    elif action == "cloud":
        return task.cpu_cycles * 0.001 + 1.0  # network + transmission
    else:  # edge1 or edge2
        return task.cpu_cycles * 0.002 + 0.5
