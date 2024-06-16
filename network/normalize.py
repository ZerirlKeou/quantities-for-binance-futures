import sys, os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
sys.path.append(os.pardir)
import sqlite3
import pandas as pd
import pickle
import numpy as np

def save_to_pickle(df, filename):
    with open(filename, 'wb') as file:
        pickle.dump(df, file)


def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


class DataNormalizer:
    def __init__(self, db_path):
        self.db_path = db_path

    def normalize(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT Open, High, Close, Low, williams_r, CCI, macd, dif, dea, Volume, Return_1 FROM 'BTCUSDT';", conn,
                               dtype='float')
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df.drop('Return_1', axis=1))
        df_scaled = pd.DataFrame(df_scaled, columns=df.columns[:-1])

        # 添加涨跌幅列并处理分类标签
        df_scaled['Return_1'] = df['Return_1']
        df_return = df_scaled['Return_1'].dropna()
        lower_third = np.percentile(df_return, 33)
        upper_third = np.percentile(df_return, 66)
        print(lower_third)
        print(upper_third)
        df_scaled['Label'] = df_scaled['Return_1'].apply(lambda x: 2 if x > upper_third else (1 if x < lower_third else 0))
        df_scaled = df_scaled.dropna()
        return df_scaled


normalizer = DataNormalizer('D:\\quant\\binance1.5.0\\data\\data_base\\1h\\BTCUSDT.db')
df_normalized = normalizer.normalize()

# 保存到 pickle 文件
pickle_filename = 'normalized_data.pkl'
save_to_pickle(df_normalized, pickle_filename)

# # 从 pickle 文件加载数据
# loaded_df = load_from_pickle(pickle_filename)
#
# df_return = loaded_df['Label'].copy()
# # 绘制 Return_1 的分布图
# sns.displot(df_return, bins=30)
# plt.title('Distribution of Return_1')
# plt.xlabel('Return_1')
# plt.ylabel('Density')
# plt.show()

