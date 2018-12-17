import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

data = pd.read_csv('longincome.csv')

print(data.columns)

clo_list = ['第三产业生产总值',
        '城镇居民消费支出', '城镇居民消费支出.1', '农村居民消费支出', '农村消费支出',
        '固定电话年末用户数', '移动电话年末用户数', '互联网宽带接入用户数']

target_df = data['江西移动收入']

data_df = data[clo_list]

target_train = target_df.values[:]
target_test = target_df.values[2]

features_train = data_df.values[:]
features_test = data_df.values[2]

print(features_train.shape)
print(target_train.shape)

lr = sklearn.linear_model.LinearRegression()

lr.fit(features_train, target_train)


#https://blog.csdn.net/little_bobo/article/details/78861578
y = lr.predict(features_test.reshape(1, -1))

print(y)

print(lr.coef_)

