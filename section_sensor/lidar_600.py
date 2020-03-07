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


### 同時確率の表 probs から P(z), P(t) の値を計算
p_t = pd.DataFrame(probs.sum())
# p_t.plot()
# print(p_t.transpose())
# plt.show()

p_z = pd.DataFrame(probs.transpose().sum())
# p_z.plot()
# print(p_z.transpose())
# plt.show()


### P(z,t) (probs) から P(z|t) を作成
cond_z_t = probs/p_t[0]
# print(cond_z_t)


### P(z | t=6), P(z | t=14)
# (cond_z_t[6]).plot.bar(color="blue", alpha=0.5)
# (cond_z_t[14]).plot.bar(color="orange", alpha=0.5)
# plt.show()


### P(z=630 | t=13)
cond_t_z = probs.transpose()/probs.transpose().sum()
# print("             P(z=630) =", p_z[0][630])
# print("             P(t= 13) =", p_t[0][13])
# print("       P(t= 13|z=630) =", cond_t_z[630][13])
# print("Bayes  P(z=630|t= 13) =", cond_t_z[630][13]*p_z[0][630]/p_t[0][13])
# print("answer P(z=630|t= 13) =", cond_z_t[13][630])


### ベイズ推定
def bayes_estimation(sensor_value, current_estimation):
    new_estimation = []
    for i in range(24):
        new_estimation.append(cond_z_t[i][sensor_value]*current_estimation[i])

    return new_estimation/sum(new_estimation) #正規化


## センサ値 630 が得られたとき
# estimation = bayes_estimation(630, p_t[0])
# plt.plot(estimation)
# plt.show()

# ## センサ値 630, 632, 636 が連続で得られたとき (5時台のデータ)
# values_5 = [630, 632, 636]
# estimation = p_t[0]
# for v in values_5:
#     estimation = bayes_estimation(v, estimation)
# plt.plot(estimation)
# plt.show()

## センサ値 617, 624, 619 が連続で得られたとき (11時台のデータ)
# values_11 = [617, 624, 619]
# estimation = p_t[0]
# for v in values_11:
#     estimation = bayes_estimation(v, estimation)
# plt.plot(estimation)
# plt.show()










