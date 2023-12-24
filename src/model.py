from sklearn.svm import SVR  # 导入支持向量回归类
from sklearn.tree import DecisionTreeRegressor  # 导入决策树回归类
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林回归类
from sklearn.metrics import mean_squared_error  # 导入MSE计算函数
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归类
import numpy as np  # 导入numpy库
from src.dataset import load_data  # 假设这个函数用于获取数据集
import pandas as pd  # 导入pandas库

class SimpleLinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        # 增加一个截距项，即全1的列
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # 使用最小二乘法的解析解来计算权重
        self.coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        # 进行预测
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 同样增加一个截距项
        return X_b.dot(self.coefficients)


def train_linear_regression_model(X_train, y_train, X_test, y_test):
    model = SimpleLinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def train_model(X_train, y_train, X_test, y_test, model_class):
    """
    通用模型训练函数。
    """
    model = model_class()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


def train_all(datasets):
    results = {"Dataset": [], "SVM": [], "Decision Tree": [], "Random Forest": [],
               "Simple Linear Regression": []}
    for dataset in datasets:
        X_train, X_test, y_train, y_test = load_data(dataset)

        # SVM
        svm_mse = train_model(X_train, y_train, X_test, y_test, SVR)
        results["Dataset"].append(dataset)
        results["SVM"].append(svm_mse)

        # Decision Tree
        dt_mse = train_model(X_train, y_train, X_test, y_test, DecisionTreeRegressor)
        results["Decision Tree"].append(dt_mse)

        # Random Forest
        rf_mse = train_model(X_train, y_train, X_test, y_test, RandomForestRegressor)
        results["Random Forest"].append(rf_mse)

        lr_mse = train_linear_regression_model(X_train, y_train, X_test, y_test)
        results["Simple Linear Regression"].append(lr_mse)

    # 将结果转换为DataFrame并打印
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df

