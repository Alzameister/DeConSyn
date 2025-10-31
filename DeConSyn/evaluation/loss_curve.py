import pandas as pd
import matplotlib.pyplot as plt

agents = ['agent_00', 'agent_01', 'agent_02', 'agent_03']

def get_avg_loss(run_path, baseline=False):
    dfs = []

    if baseline:
        df = pd.read_csv(f'{run_path}/loss.csv')
        return df.set_index('step')['loss']

    for agent in agents:
        df = pd.read_csv(f'{run_path}/{agent}/loss.csv')
        dfs.append(df.set_index('step')['loss'])
    # Align steps and average
    avg_loss = pd.concat(dfs, axis=1).mean(axis=1)
    return avg_loss

run1_path = '../../runs/run-20251015-181504-4Agents-1Epochs-3000Iterations-full-tabddpm'
run2_path = '../../runs/tabddpm/tabddpm_baseline'
run3_path = '../../runs/run-20251016-064516-4Agents-1Epochs-1000Iterations-full-tabddpm'

avg_loss1 = get_avg_loss(run1_path)
avg_loss2 = get_avg_loss(run2_path, baseline=True)
avg_loss3 = get_avg_loss(run3_path)

plt.figure(figsize=(10, 5))
plt.plot(avg_loss1.index, avg_loss1.values, label='Run 1 Avg Loss')
plt.plot(avg_loss2.index, avg_loss2.values, label='Run 2 Avg Loss')
plt.plot(avg_loss3.index, avg_loss3.values, label='Run 3 Avg Loss')
plt.xlabel('Step')
plt.ylabel('Average Loss')
plt.title('Average Loss vs Step (All Agents)')
plt.ylim(1, 3)
plt.legend()
plt.grid(True)
plt.show()

def plot_agents(run_path, run_label):
    plt.figure(figsize=(10, 5))
    for agent in agents:
        df = pd.read_csv(f'{run_path}/{agent}/loss.csv')
        plt.plot(df['step'], df['loss'], label=agent)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Loss vs Step ({run_label})')
    plt.ylim(1, 3)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_agents(run1_path, 'Run 1')
plot_agents(run3_path, 'Run 3')