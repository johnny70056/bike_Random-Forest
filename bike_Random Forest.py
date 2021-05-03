# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 09:30:03 2021

@author: johnn
"""

from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import calendar
pd.options.mode.chained_assignment = None

df = pd.read_csv('./4 bike-sharing-demand\\train.csv')
df_test = pd.read_csv('./4 bike-sharing-demand\\test.csv')
#%%
#檢查資料分布
check = df.describe()
print(check)
#%%
print('整理前資料量',df.shape)
df = df[np.abs(df['count']-df['count'].mean())<=(3*df['count'].std())] #因count可能有錯誤的數值，所以將大於標準差3倍的值剃除
print('整理後資料量',df.shape)


#%%
#將train和test資料合併，因為要處理時間日期格式
data = df.append(df_test)
data.reset_index(inplace=True)
data.drop('index',inplace=True,axis=1)

#%%
#將時間拆成日期、小時、年、星期幾、月份的欄位（日期之後不會用到主要是用來算星期幾以及月份）
data["date"] = data.datetime.apply(lambda x : x.split()[0])
data["hour"] = data.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")
data["year"] = data.datetime.apply(lambda x : x.split()[0].split("-")[0])
data["weekday"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())
data["month"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)

print(data)

#%%

#檢查風速、溫度、體感溫度、濕度的常態變化
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
sns.distplot(data["temp"],ax=axes[0][0])
sns.distplot(data["atemp"],ax=axes[0][1])
sns.distplot(data["humidity"],ax=axes[1][0])
sns.distplot(data["windspeed"],ax=axes[1][1])

axes[0][0].set(xlabel='temp',title="distribution of temp")
axes[0][1].set(xlabel='atemp',title="distribution of atemp")
axes[1][0].set(xlabel='humidity',title="distribution of humidity")
axes[1][1].set(xlabel='windspeed',title="distribution of windspeed")

#可以發現風速不為常態變化，且風速為0值太多(可認定為儀器檢測不了or缺值就補上0)

#%%
"""
我們先將資料分成風速為0以及風速不為0的資料，
並且用風速不為0的資料來訓練random forest的模型，
將訓練好的模型來預估風速為0的風速到底是多少。
"""
dataWind0 = data[data["windspeed"]==0]
dataWindNot0 = data[data["windspeed"]!=0]
rfModel_wind = RandomForestRegressor(n_estimators=1000,random_state=42)
windColumns = ["season","weather","humidity","month","temp","year","atemp"]
rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])

wind0Values = rfModel_wind.predict(X= dataWind0[windColumns])
dataWind0.loc[:,"windspeed"] = wind0Values
data = dataWindNot0.append(dataWind0)
data.reset_index(inplace=True)
data.drop('index',inplace=True,axis=1)

sns.distplot(data["windspeed"])


#%%
"""
觀察腳踏車出借數量(count)的資料分佈，
可以發現原本的資料非常歪斜(Skew)，
也就是不符合常態分佈。透過取Log的方式，
來讓資料分布較為接近常態分佈，這樣的技巧也可以讓預估上更準確。
"""
dataTrain = data[pd.notnull(data['count'])].sort_values(by=["datetime"])
dataTest = data[~pd.notnull(data['count'])].sort_values(by=["datetime"])
datetimecol = dataTest["datetime"]
yLabels = dataTrain["count"]
yLabelsLog = np.log(yLabels)

dropFeatures = ['casual',"count","datetime","date","registered"]
dataTrain  = dataTrain.drop(dropFeatures,axis=1)
dataTest  = dataTest.drop(dropFeatures,axis=1)

sns.distplot(yLabels)

#%%

#可以看到取Log後的資料更接近常態分佈了。

sns.distplot(yLabelsLog)

#%%

rfModel = RandomForestRegressor(n_estimators=1000,random_state=42)
yLabelsLog = np.log(yLabels)
rfModel.fit(dataTrain,yLabelsLog)
preds = rfModel.predict(X= dataTrain)


predsTest = rfModel.predict(X= dataTest)
submission = pd.DataFrame({
        "datetime": datetimecol,
        "count": [max(0, x) for x in np.exp(predsTest)]
    })
submission.to_csv('bike_kaggle.csv', index=False,header=True)



