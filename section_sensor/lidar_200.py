#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


### データ読み込み
data = pd.read_csv("sensor_data_200.txt", delimiter=" ",
                   header=None, names=("date","time","ir","lidar"))
#print(data["lidar"][0:5])


### 平均値
mean1 = sum(data["lidar"])/len(data["lidar"])
mean2 = data["lidar"].mean() #Pandas


### ヒストグラム作成
# data["lidar"].hist(bins=max(data["lidar"])-min(data["lidar"]), color="orange", align='left')
# plt.vlines(mean1, ymin=0, ymax=5000, color="red")
# plt.show()


### 分散計算
zs = data["lidar"]
mean = sum(zs)/len(zs)
diff_square = [(z-mean)**2 for z in zs]

sampling_var = sum(diff_square)/(len(zs))   # 標本分散
unbiased_var = sum(diff_square)/(len(zs)-1) # 不変分散


### 標準偏差計算
stddev1 = math.sqrt(sampling_var)
stddev2 = math.sqrt(unbiased_var)


### 素朴な確率
freqs = pd.DataFrame(data["lidar"].value_counts())
freqs["probs"] = freqs["lidar"]/len(data["lidar"])
#print(freqs.transpose())


### 確率質量関数
# freqs["probs"].sort_index().plot.bar()
# plt.show()


### 分布からサンプル生成
def drawing():
    return freqs.sample(n=1, weights="probs").index[0]

#samples = [drawing() for i in range(1000)]
samples = [drawing() for i in range(len(data))]
simulated = pd.DataFrame(samples, columns=["lidar"])
p = simulated["lidar"]
p.hist(bins=max(p)-min(p), color="orange", align='left')
plt.show()





