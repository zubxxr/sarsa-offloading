import numpy as np

class Task:
    def __init__(self):
        self.cpu_cycles = np.random.randint(500, 2000)  # in million cycles
        self.data_size = np.random.uniform(1, 10)       # in MB

class DeviceState:
    def __init__(self):
        self.battery = np.random.uniform(10, 100)       # in percentage
        self.cpu_usage = np.random.uniform(10, 90)      # in percentage
        self.bandwidth = np.random.uniform(1, 100)      # in Mbps

def simulate_task_and_state():
    task = Task()
    state = DeviceState()
    return task, state