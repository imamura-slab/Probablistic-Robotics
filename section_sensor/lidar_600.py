#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



data = pd.read_csv("sensor_data_600.txt", delimiter=" ",
                   header=None, names=("date","time","ir","lidar"))

# data["lidar"].hist(bins=max(data["lidar"]) - min(data["lidar"]), align='left')
# plt.show()

# data.lidar.plot()
# plt.show()


### センサ値を時間ごとにグループ分けして各グループの平均値をグラフ化
data["hour"] = [e//10000 for e in data.time]
d = data.groupby("hour")
# d.lidar.mean().plot()
# plt.show()


### 6時台, 14時台のヒストグラム
# d.lidar.get_group(6).hist()
# d.lidar.get_group(14).hist()
# plt.show()


### 同時確率分布, 結合確率分布
each_hour = {i:d.lidar.get_group(i).value_counts().sort_index() for i in range(24)}
freqs = pd.concat(each_hour, axis=1)
freqs = freqs.fillna(0)
probs = freqs/len(data)

## 表示1
# sns.heatmap(probs)
# plt.show()

## 表示2    表示できない!?
# sns.jointplot(data["hour"], data["lidar"], data, kind="kde")
# plt.show()





