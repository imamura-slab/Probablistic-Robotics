#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal


data = pd.read_csv("sensor_data_700.txt", delimiter=" ",
                   header=None, names=("date","time","ir","lidar"))

# data["lidar"].hist(bins=max(data["lidar"]) - min(data["lidar"]), align='left')
# plt.show()

d = data[(data["time"]<160000) & (data["time"]>=120000)] #12時から16時までのデータのみ抽出
d = d.loc[:, ["ir", "lidar"]] #loc[:,[列ラベル]], loc[[行ラベル]], loc[[行ラベル],[列ラベル]] で指定

# sns.jointplot(d["ir"], d["lidar"], kind='kde')
# plt.show()



### 分散, 共分散

## 方法1
# print("光センサの計測値の分散:", d.ir.var())
# print("LiDARの計測値の分散:", d.lidar.var())
# diff_ir = d.ir - d.ir.mean()
# diff_lidar = d.lidar - d.lidar.mean()
# a = diff_ir * diff_lidar
# print("共分散:", sum(a)/(len(d)-1))
# print(d.mean())

## 方法2
print(d.cov())



### 2次元ガウス分布の描画
irilidar = multivariate_normal(mean=d.mean().values.T, cov=d.cov().values)
x, y = np.mgrid[0:40, 710:750] #2次元平面に均等にX座標Y座標を作る
pos = np.empty(x.shape + (2,)) #40x40x2のリストを作成
pos[:, :, 0] = x
pos[:, :, 1] = y
##----- contour: 等高線 -----##
# cont = plt.contour(x,y,irilidar.pdf(pos)) #x,y座標とそれに対応する密度を算出
# cont.clabel(fmt='%1.1e') #等高線に値を書き込むためのフォーマットを指定
# plt.show()


c = d.cov().values + np.array([[0,20],[20,0]])
tmp = multivariate_normal(mean=d.mean().values.T, cov=c)
cont = plt.contour(x, y, tmp.pdf(pos))
plt.show()



