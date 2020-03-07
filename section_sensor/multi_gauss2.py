#!/opt/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
import math


data = pd.read_csv("sensor_data_200.txt", delimiter=" ",
                   header=None, names=("date","time","ir","lidar"))

# data["lidar"].hist(bins=max(data["lidar"]) - min(data["lidar"]), align='left')
# plt.show()

d = data.loc[:, ["ir", "lidar"]]

# sns.jointplot(d["ir"], d["lidar"], kind='kde')
# print(d.cov())
# plt.show()


x, y = np.mgrid[280:340, 190:230]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y
irlidar = multivariate_normal(mean=d.mean().values.T, cov=d.cov().values)
cont = plt.contour(x, y, irlidar.pdf(pos))
cont.clabel(fmt='%1.1e')
plt.show()






