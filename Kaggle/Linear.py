import gc,joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# read dataset
data = pd.read_feather('/data/xxl/dataset/train.feather')
data = data.astype('float16')
print(data.head(10))
data_columns = data.columns
print(data_columns)
# get feature columns
feature_columns = data.drop(['row_id','time_id','investment_id','target'],axis=1).columns
print(feature_columns)

# 利用sklearn中的StandardScaler对数据进行归一化处理
scaler = StandardScaler()
# 计算均值和方差
scaler.fit(pd.DataFrame(data['investment_id']))   # <class 'pandas.core.frame.DataFrame'>


# 定义一个函数，用来制作数据集
def make_dataset(df):
    investment_df = df['investment_id']    # <class 'pandas.core.series.Series'>
    feature_df= df[feature_columns]
    # 对 investment_id 数据进行转换，转换成标准正态分布
    scaled_investment_id = scaler.transform(pd.DataFrame(investment_df)) 
    df['investment_id'] = scaled_investment_id
    # 数据拼接，将想要的两列数据拼接在一起
    data_x = pd.concat([df['investment_id'], feature_df], axis=1)
    return data_x

x_data = make_dataset(data)
# x_data = x_data.values
# x_data = np.array(x_data)

y_data = data['target']
# y_data = y_data.values
# y_data = np.array(y_data)

# make model
kf  = KFold(n_splits=5)
models = []
scores = []
for i,(train_index,val_index) in enumerate(kf.split(x_data)):
    print('-'*50)
    print(f'round{i}')

    x_train,y_train = x_data.iloc[train_index],y_data.iloc[train_index]
    x_val,y_val = x_data.iloc[val_index],y_data.iloc[val_index]

    model = LinearRegression()
    model.fit(x_train,y_train)
    models.append(model)
    joblib.dump(model,f'round_{i}.pkl')

    y_pred = model.predict(x_val)
    rmse = np.sqrt(mean_squared_error(y_pred,y_val))
    corr = pearsonr(y_pred,y_val)[0]

    print(f'RMSE: {rmse},\t Pearson correlation score: {corr}')

print(f'相关系数的均值: {np.mean(scores, axis=0)}')    




