import pandas as pd
import os

def get_dataset(dataset):
    base_root = "data/split"
    train_path = os.path.join(base_root, dataset, "train.csv")
    dev_path = os.path.join(base_root, dataset, "dev.csv")
    test_path = os.path.join(base_root, dataset, "test.csv")
    train = pd.read_csv(train_path)
    dev = pd.read_csv(dev_path)
    test = pd.read_csv(test_path)
    return train, dev, test

def _load_data(dataframe):
    X = dataframe.iloc[:, :-1].values  # 除了最后一列，其余都是特征
    y = dataframe.iloc[:, -1].values  # 最后一列是目标变量
    return X, y

def load_data(dataset = "WineQualityRed"):
    train, dev, test = get_dataset(dataset)
    X_train,y_train = _load_data(train)
    X_test,y_test = _load_data(test)
    return X_train, X_test, y_train, y_test

datasets = ["OnlineNewsPopularity", "WineQualityRed", "HousingData"]

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_features(X, dataset_name):
    df = pd.DataFrame(X)
    num_features = df.shape[1]

    # 计算接近正方形的行列数
    ncols = int(np.ceil(np.sqrt(num_features)))
    nrows = int(np.ceil(num_features / ncols))

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))

    # 检查是否有多个子图
    if nrows * ncols > 1:
        ax = ax.flatten()
    else:
        ax = [ax]

    for i in range(num_features):
        unique_values = df.iloc[:, i].nunique()

        # 分类变量使用条形图
        if unique_values <= 10:
            sns.countplot(x=df.iloc[:, i], ax=ax[i])
            ax[i].set_title(f"Categorical Feature {i} in {dataset_name}")
        else:
            # 连续变量使用直方图
            sns.histplot(df.iloc[:, i], ax=ax[i], kde=True)
            ax[i].set_title(f"Continuous Feature {i} in {dataset_name}")

        ax[i].set_xlabel(f"Feature {i} Value")
        ax[i].set_ylabel("Count")

    # 移除多余的子图
    for j in range(num_features, nrows * ncols):
        fig.delaxes(ax[j])

    plt.tight_layout()
    plt.show()
    plt.savefig(dataset_name + ".png")

# 加载数据并绘制每个数据集的特征
for data in datasets:
    X_train, X_test, y_train, y_test = load_data(data)
    plot_features(X_train, data)
