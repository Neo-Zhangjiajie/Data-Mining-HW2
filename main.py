from src.model import train_all
import pandas as pd

# 调用train_all函数，传入想要训练的数据集名称列表
train_all(["OnlineNewsPopularity", "WineQualityRed", "HousingData"]).to_excel("results-xyf.xlsx")