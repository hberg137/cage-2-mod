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

# cardiff_original = np.load("results/Scenario2_RED_MB_MODEL_MB_MEANDER.npy")
cardiff_abstract = np.load("results/Scenario2_RED_MB_MODEL_MB_MEANDER_ABSTRACT.npy")
oracle_switch_abstract = np.load("results/Scenario2_RED_MB_MODEL_MB_ORACLE_ABSTRACT.npy")
learned_switch = np.load("results/Scenario2_RED_MB_MODEL_MB_LEARNED.npy")
learned_switch_abstract = np.load("results/Scenario2_RED_MB_MODEL_MB_LEARNED_ABSTRACT.npy")

# lstm_results = np.load("results/lstm_results.npy")

# print(lstm_results.shape)

# Calculate the mean and standard deviation along the episodes
# cardiff_original_df = pd.DataFrame(cardiff_original).cumsum(axis=1).ewm(span=10, adjust=False).mean()
# cardiff_original_mean = np.mean(cardiff_original_df, axis=0)
# cardiff_original_std = np.std(cardiff_original_df, axis=0)

cardiff_abstract_df = pd.DataFrame(cardiff_abstract).cumsum(axis=1).ewm(span=10, adjust=False).mean()
cardiff_abstract_mean = np.mean(cardiff_abstract_df, axis=0)
cardiff_abstract_std = np.std(cardiff_abstract_df, axis=0)

# oracle_switch_df = pd.DataFrame(oracle_switch).cumsum(axis=1).ewm(span=10, adjust=False).mean()
# oracle_switch_mean = np.mean(oracle_switch_df, axis=0)
# oracle_switch_std = np.std(oracle_switch_df, axis=0)

learned_switch_df = pd.DataFrame(learned_switch).cumsum(axis=1).ewm(span=10, adjust=False).mean()
learned_switch_mean = np.mean(learned_switch_df, axis=0)
learned_switch_std = np.std(learned_switch_df, axis=0)

oracle_switch_abstract_df = pd.DataFrame(oracle_switch_abstract).cumsum(axis=1).ewm(span=10, adjust=False).mean()
oracle_switch_abstract_mean = np.mean(oracle_switch_abstract_df, axis=0)
oracle_switch_abstract_std = np.std(oracle_switch_abstract_df, axis=0)

learned_switch_abstract_df = pd.DataFrame(learned_switch_abstract).cumsum(axis=1).ewm(span=10, adjust=False).mean()
learned_switch_abstract_mean = np.mean(learned_switch_abstract_df, axis=0)
learned_switch_abstract_std = np.std(learned_switch_abstract_df, axis=0)

# lstm_df = pd.DataFrame(lstm_results).ewm(span=10, adjust=False).mean()
# lstm_mean = np.mean(lstm_results, axis=0)
# lstm_std = np.std(lstm_results, axis=0)


# Plot the EWMA and standard deviation
plt.figure(figsize=(8, 8))

# Plot EWMA
# plt.plot(cardiff_original_mean, label='CardiffUni Original BT', color='red')
# plt.fill_between(range(len(cardiff_original_mean)),
#                  cardiff_original_mean - cardiff_original_std,
#                  cardiff_original_mean + cardiff_original_std, color='red', alpha=0.1)

# plt.plot(cardiff_abstract_mean, label='CardiffUni GPBT', color='red')
# plt.fill_between(range(len(cardiff_abstract_mean)),
#                  cardiff_abstract_mean - cardiff_abstract_std,
#                  cardiff_abstract_mean + cardiff_abstract_std, color='red', alpha=0.1)

# plt.plot(oracle_switch_mean, label='OracleSwitch', color='blue')
# plt.fill_between(range(len(oracle_switch_mean)), oracle_switch_mean - oracle_switch_std, oracle_switch_mean + oracle_switch_std, color='orange', alpha=0.1)

plt.plot(learned_switch_abstract_mean, label='LearnedSwitch GPBT', color='red')
plt.fill_between(range(len(learned_switch_abstract_mean)),
                 learned_switch_abstract_mean - learned_switch_abstract_std,
                 learned_switch_abstract_mean + learned_switch_abstract_std, color='red', alpha=0.1)

plt.plot(learned_switch_mean, label='LearnedSwitch Expert', color='blue', linestyle='dashed', dashes=(10,10))
plt.fill_between(range(len(learned_switch_mean)),
                 learned_switch_mean - learned_switch_std,
                 learned_switch_mean + learned_switch_std, color='blue', alpha=0.1)

# plt.plot(oracle_switch_abstract_mean, label='OracleSwitch GPBT', color='blue')
# plt.fill_between(range(len(oracle_switch_abstract_mean)),
#                  oracle_switch_abstract_mean - oracle_switch_abstract_std,
#                  oracle_switch_abstract_mean + oracle_switch_abstract_std, color='blue', alpha=0.1)




# plt.plot(lstm_mean, label='CardiffUni', color='blue')
# plt.fill_between(range(len(lstm_std)), lstm_mean - lstm_std, lstm_mean + lstm_std, color='blue', alpha=0.1)

# plt.title('EWMA with Std Dev')
plt.xlabel('Timestep')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.savefig("figs/abstract/learned_comparison.jpeg")
plt.savefig("figs/abstract/learned_comparison.pdf")
plt.show()