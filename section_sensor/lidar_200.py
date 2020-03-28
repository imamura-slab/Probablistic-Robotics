#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import random


### データ読み込み
data = pd.read_csv("sensor_data_200.txt", delimiter=" ",  #データの区切り文字は " " (SPACE)
                   header=None, names=("date","time","ir","lidar"))
#print(data["lidar"][0:5])


### 平均値
mean1 = sum(data["lidar"])/len(data["lidar"]) #素直に
mean2 = data["lidar"].mean()                  #Pandas


### ヒストグラム作成
## bins: 横軸の各区間の数. 今回は区間の幅を1にしている.
## align='left' : 中央の値を整数にしている. (今回用いたセンサ値は整数だから)
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

# #samples = [drawing() for i in range(1000)]
# samples = [drawing() for i in range(len(data))]  #じかんかかる
# simulated = pd.DataFrame(samples, columns=["lidar"])
# p = simulated["lidar"]
# p.hist(bins=max(p)-min(p), color="orange", align='left')
# plt.show()


### 正規分布作成
def p(z, mu=209.7, dev=23.4):
    return math.exp(-(z-mu)**2/(2*dev))/math.sqrt(2*math.pi*dev)
# zs = range(190, 230)
# ys = [p(z) for z in zs]
# plt.plot(zs, ys)
# plt.show()


### 区間[x-0.5, x+0.5)の範囲で積分
def prob(z, width=0.5):
    return width*(p(z-width)+p(z+width))
# zs = range(190, 230)
# ys = [prob(z) for z in zs]
# plt.bar(zs, ys, color="red", alpha=0.3)
# f = freqs["probs"].sort_index()
# plt.bar(f.index, f.values, color="blue", alpha=0.3)
# plt.show()


### scipyを使用してpdf(確率密度関数)
# zs = range(190, 230)
# ys = [norm.pdf(z, mean1, stddev1) for z in zs]
# plt.plot(zs, ys)
# plt.show()


### cdf (累積分布関数)
# zs = range(190, 230)
# ys = [norm.cdf(z, mean1, stddev1) for z in zs]
# plt.plot(zs, ys, color="red")
# plt.show()


### cdfから確率分布を描く
# zs = range(190, 230)
# ys = [norm.cdf(z+0.5, mean1, stddev1) - norm.cdf(z-0.5, mean1, stddev1) for z in zs]
# plt.bar(zs, ys)
# plt.show()


### さいころを10000回振って疑似的に期待値を求める (さいころの出目の期待値は3.5)
samples = [random.choice([1,2,3,4,5,6]) for i in range(10000)]
print(sum(samples)/len(samples))







