import pandas as pd

df = pd.read_csv("../market_data_and_move.csv")

target_columns = ['up', 'down']
feature_columns = [col for col in df.columns if col not in target_columns]

dataset_mean = df[feature_columns].mean()
dataset_std = df[feature_columns].std()

dataset_std[dataset_std == 0] = 1e-9

df_normalized = df.copy()

for col in feature_columns:
    df_normalized[col] = (df_normalized[col] - dataset_mean[col]) / dataset_std[col]

df_normalized.to_csv("normalized_with_true_full_dataset.csv", index=False)
