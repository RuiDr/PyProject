# -*- coding: utf-8 -*-
# @Author  : dr
# @Time    : 2022/3/14 10:35
# @Function:使用scikit-learn训练并运行线性模型
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


def prepare_country_stats(oecd_bli_pre, gdp_per_capita):
    oecd_bli_pre = oecd_bli_pre[oecd_bli_pre["INEQUALITY"] == "TOT"]
    oecd_bli_pre = oecd_bli_pre.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli_pre, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


# 加载数据
oecd_bli = pd.read_csv("E:\\projects\\python\\PyProject\\Ml_In_Action\\handsOnMlwithSklearnAndTf\\datasets\\lifesat\\"
                       "oecd_bli_2015.csv", thousands=',')
ddp_per_capita = pd.read_csv("E:\\projects\\python\\PyProject\\Ml_In_Action\\handsOnMlwithSklearnAndTf\\datasets\\"
                             "lifesat\\gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1',
                             na_values='n/a')

# 准备数据
country_stats = prepare_country_stats(oecd_bli, ddp_per_capita)
x = np.c_[country_stats['GDP per capita']]
y = np.c_[country_stats['Life satisfaction']]
print(x)
print(y)
# 选择线性模型
# lin_reg_model = sklearn.linear_model.LinearRegression()
# 选择k近邻模型
lin_reg_model = KNeighborsRegressor(n_neighbors=3)

# 训练模型
lin_reg_model.fit(x, y)

# 对塞浦路斯进行预测
# 塞浦路斯的人均GDP
x_new = [[22587]]
print(lin_reg_model.predict(x_new))

# 可视化数据
country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
plt.show()

