import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import os

dataset_name = 'OnlineNewsPopularity'

# Function to read CSV and compute z-score normalization
def normalize_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Select only numeric columns for normalization
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    
    # Fill NA values with the mean of each column
    for col in numeric_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    # Apply z-score normalization to numeric columns
    df[numeric_cols] = df[numeric_cols].apply(zscore)

    return df

def cal_corrcoef(data_list, rk1, rk2):
    rk1_data = [item[rk1] for item in data_list]
    rk2_data = [item[rk2] for item in data_list]
    return np.corrcoef(np.array(rk1_data), np.array(rk2_data))[0, 1]

file_path = f'{dataset_name}.csv'
normalized_df = normalize_csv(file_path)
normalized_df.to_csv(f"{dataset_name}-normalized.csv")
print(normalized_df)


header_list, data_list, target_list = [], [], []
with open(f"{dataset_name}-normalized.csv") as f:
    lines = [x.strip() for x in f]
    header_list = [x for x in lines[0].split(",")[1:-1]] # ignore the id and prediction columns

for line in lines[1:]:
    items = [float(x) for x in line.split(",")]
    data_list.append(items[:-1])
    target_list.append(items[-1])
        
print(header_list)
num_features = len(data_list[0])
heat_array = np.zeros((num_features, num_features))

deleted_header = []

for i in range(num_features):
    for j in range(num_features):
        heat_array[i][j] = cal_corrcoef(data_list, i, j)
        if abs(heat_array[i][j]) > 0.9 and i < j:
            print(header_list[i], "&", header_list[j], "  corrcoef =",  heat_array[i][j])
            if header_list[i] not in deleted_header:
                deleted_header.append(header_list[i])
                normalized_df.drop(header_list[i], axis=1, inplace=True)
                print(f'delete feature {header_list[i]}')

# Save new normalized data without highly correlated features
normalized_df.to_csv(f"{dataset_name}-normalized.csv")
            
sns.set()
ax = sns.heatmap(heat_array)
plt.savefig(f"{dataset_name}-corr-heatmap.png")

# Split for train, dev, test
if not os.path.exists(f"./split/{dataset_name}"):
    os.mkdir(f"./split/{dataset_name}")
train_df = normalized_df.sample(frac=0.6, random_state=0)
dev_df = normalized_df.drop(train_df.index)
dev_df = dev_df.sample(frac=0.5, random_state=0)
test_df = normalized_df.drop(train_df.index).drop(dev_df.index)
train_df.to_csv(f"./split/{dataset_name}/train.csv")
dev_df.to_csv(f"./split/{dataset_name}/dev.csv")
test_df.to_csv(f"./split/{dataset_name}/test.csv")