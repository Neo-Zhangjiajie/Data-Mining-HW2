import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def cal_corrcoef(data_list, rk1, rk2):
    rk1_data = [item[rk1] for item in data_list]
    rk2_data = [item[rk2] for item in data_list]
    return np.corrcoef(np.array(rk1_data), np.array(rk2_data))[0, 1]

header_list, data_list, target_list = [], [], []
with open("./winequality-red.csv") as f:
    lines = [x.strip() for x in f]
header_list = [x[1:-1] for x in lines[0].split(";")]
with open("../../rrl_regression/dataset/red-wine-quality.info", "w") as fout:
    for head in header_list:
        fout.write(head.replace(' ', '_')+' '+'continuous\n')
    fout.write('LABEL_POS -1\n')
with open("../../rrl_regression/dataset/red-wine-quality.data", "w") as fout:
    for line in lines[1:]:
        fout.write(line.replace(";", ',')+'\n')
        items = [float(x) for x in line.split(";")]
        data_list.append(items[:-1])
        target_list.append(items[-1])
    
# print(header_list)
# num_features = len(data_list[0])
# heat_array = np.zeros((num_features, num_features))

# for i in range(num_features):
#     for j in range(num_features):
#         heat_array[i][j] = cal_corrcoef(data_list, i, j)
#         if abs(heat_array[i][j]) > 0.75 and i < j:
#             print(header_list[i], "&", header_list[j], "  corrcoef =",  heat_array[i][j])
# sns.set()
# ax = sns.heatmap(heat_array)
# plt.savefig("corr-heatmap.png")

