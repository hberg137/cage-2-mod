import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
plt.rcParams['font.size'] = 20


# cardiff = np.load("results/Scenario2_RED_MB_MODEL_MEANDER.npy")
# oracle_switch = np.load("results/Scenario2_RED_MB_MODEL_MB_FULL.npy")
# learned_switch = np.load("results/Scenario2_RED_MB_MODEL_MB_LSTM.npy")

lstm_results = np.load("results/lstm_results.npy")

print(lstm_results.shape)

# Calculate the mean and standard deviation along the episodes
# cardiff_df = pd.DataFrame(cardiff).cumsum(axis=1).ewm(span=10, adjust=False).mean()
# cardiff_mean = np.mean(cardiff_df, axis=0)
# cardiff_std = np.std(cardiff_df, axis=0)

# oracle_switch_df = pd.DataFrame(oracle_switch).cumsum(axis=1).ewm(span=10, adjust=False).mean()
# oracle_switch_mean = np.mean(oracle_switch_df, axis=0)
# oracle_switch_std = np.std(oracle_switch_df, axis=0)

# learned_switch_df = pd.DataFrame(learned_switch).cumsum(axis=1).ewm(span=10, adjust=False).mean()
# learned_switch_mean = np.mean(learned_switch_df, axis=0)
# learned_switch_std = np.std(learned_switch_df, axis=0)

lstm_df = pd.DataFrame(lstm_results).ewm(span=10, adjust=False).mean()
lstm_mean = np.mean(lstm_results, axis=0)
lstm_std = np.std(lstm_results, axis=0)


# Plot the EWMA and standard deviation
plt.figure(figsize=(8, 8))

# Plot EWMA
# plt.plot(cardiff_mean, label='CardiffUni', color='red')
# plt.fill_between(range(len(cardiff_mean)), cardiff_mean - cardiff_std, cardiff_mean + cardiff_std, color='red', alpha=0.1)

# plt.plot(oracle_switch_mean, label='OracleSwitch', color='blue')
# plt.fill_between(range(len(oracle_switch_mean)), oracle_switch_mean - oracle_switch_std, oracle_switch_mean + oracle_switch_std, color='orange', alpha=0.1)

# plt.plot(learned_switch_mean, label='LearnedSwitch (Ours)', color='green')
# plt.fill_between(range(len(learned_switch_mean)), learned_switch_mean - learned_switch_std, learned_switch_mean + learned_switch_std, color='green', alpha=0.1)

plt.plot(lstm_mean, label='CardiffUni', color='blue')
plt.fill_between(range(len(lstm_std)), lstm_mean - lstm_std, lstm_mean + lstm_std, color='blue', alpha=0.1)

# plt.title('EWMA with Std Dev')
plt.xlabel('Iteration')
plt.ylabel('BCE Loss')
# plt.legend()
plt.savefig("figs/lstm_training.pdf")
plt.show()