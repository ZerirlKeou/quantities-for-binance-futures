import pandas as pd
import numpy as np
import os


class SingleInterval(object):
    """
    : params interval : 输入级别
    : params size : 合并前size天的数据
    """

    def __init__(self, interval, size):
        self.interval = interval
        self.size = size

    def _clean_data(self):
        data_file = os.path.join("data", f"{self.interval}_data.csv")
        key_columns = ['Open time', 'open-close', 'open-close', 'high-open', 'Close', 'Open', 'High', 'Low', 'dea',
                       'dif', 'macd', 'Return_1', 'Return_2', 'volume_change']
        data = pd.read_csv(data_file, usecols=key_columns)
        columns_to_check = ['Close', 'dif', 'dea', 'macd', 'Return_1', 'Return_2']
        data.dropna(subset=columns_to_check, inplace=True)
        return data.reset_index(drop=True)

    def _merge_data(self):
        # 准备合并数据
        df = self._clean_data()
        columns = df.columns
        df_tail = df.drop(['Open time', 'Return_1', 'Return_2'], axis=1)
        for i in range(1, self.size):
            df_tail.index = df_tail.index + 1
            df = pd.concat([df, df_tail], axis=1)
        df = df.dropna().reset_index(drop=True).to_numpy()

        return pd.DataFrame(df, columns=list(columns) + [f"subdf_{i + 1}" for i in
                                                         range((self.size - 1) * df_tail.shape[1])])

    def new_data(self):
        df_merged = self._merge_data()
        df_merged.to_csv('test.csv', index=False)


class GetTwoData(object):
    """ 用于得到两个级别数据，并返回将数据时间戳对其的dataframe类型
    : params interval1 : 第一个级别
    : params interval2 : 第二个级别
    """

    def __init__(self, interval1, interval2):
        self.df = self._clean_data(interval1).reset_index(drop=True)
        self.subdf = self._clean_data(interval2).reset_index(drop=True)
        self.subdf_times = set(self.subdf["Open time"])
        index_df, index_subdf = self._find_beginning_position()
        if index_df == 0:
            pass
        else:
            self.df = self.df[-(len(self.df) - index_df):].reset_index(drop=True)
        if index_subdf == 0:
            pass
        else:
            self.subdf = self.subdf[-(len(self.subdf) - index_subdf):].reset_index(drop=True)

    def _clean_data(self, interval):
        data_file = os.path.join("data", f"{interval}_data.csv")
        key_columns = ['Open time', "Close", "Open", "High", "Low", 'high-open', "open-close", "low-open", "high+dif",
                       'Return_1', 'Return_2', "macd", "dea", "dif"]
        # key_columns = ['Open time', 'macd_back',
        #                'Return_1', 'Return_2']
        data = pd.read_csv(data_file, usecols=key_columns)
        # data["dea"].replace(0, pd.NA, inplace=True)
        data.dropna(subset=["high-open", "open-close", "low-open", "high+dif", "macd", "dea", "dif",
                            'Return_1', 'Return_2'], inplace=True)
        # data.dropna(subset=['macd_back',
        #                     'Return_1', 'Return_2'], inplace=True)
        return data

    def _find_beginning_position(self, start_row=0):
        df_subset = self.df.iloc[start_row:]

        for index_df, row_df in df_subset.iterrows():
            if row_df["Open time"] in self.subdf_times:
                subdf_row = self.subdf.loc[self.subdf["Open time"] == row_df["Open time"]]
                return index_df, subdf_row.index[0] + start_row

        return None


class Connection2(GetTwoData):
    """ 自动替换小级别中的预测值
    : params interval1 : 第一个级别 大级别
    : params interval2 : 第二个级别 小级别
    """

    def __init__(self, interval1, interval2):
        super().__init__(interval1, interval2)
        if interval1 == "15m" and interval2 == "5m":
            self.size = 3
        elif interval1 == "5m" and interval2 == "1m":
            self.size = 5
        else:
            raise ValueError("Check the Input interval whether legal")

    def _merge_data(self, df_data, subdf_data):
        length = subdf_data.shape[0] // self.size
        if len(df_data) < length + 1:
            length = len(df_data) - 1
        else:
            df_data = df_data[:length]
        if len(subdf_data) == length * self.size:
            pass
        else:
            subdf_data = subdf_data[:length * 3]

        k = 0
        for j in range(length):
            for i in range(1, self.size + 1):
                subdf_data.loc[k, "Return_1"] = df_data.loc[j + 1, "Return_1"]
                k += 1

        columns = subdf_data.columns
        df_tail = subdf_data.drop(['Open time', 'Return_1', 'Return_2'], axis=1)
        for i in range(1, self.size):
            df_tail.index = df_tail.index + 1
            subdf_data = pd.concat([subdf_data, df_tail], axis=1)
        subdf_data = subdf_data.dropna().reset_index(drop=True).to_numpy()

        return pd.DataFrame(subdf_data, columns=list(columns) + [f"subdf_{i + 1}" for i in
                                                                 range((self.size - 1) * df_tail.shape[1])])

    def new_data(self):
        df_merged = self._merge_data(self.df, self.subdf)
        df_merged.to_csv('test.csv', index=False)


class Connection(GetTwoData):
    def __init__(self, interval1, interval2):
        super().__init__(interval1, interval2)

    def _merge_data(self, df_data, subdf_data, size, columns):
        subdf_data = subdf_data.reindex(columns=columns)
        length = subdf_data.shape[0] // (size * 3)
        if len(df_data) < length:
            df_array = df_data.to_numpy()
            length = len(df_array)
        else:
            df_array = df_data.to_numpy()[:length]

        if len(subdf_data) == length * 3:
            subdf_array = subdf_data.to_numpy()
        else:
            subdf_array = subdf_data.to_numpy()[:length * 3]
        new_subdf = subdf_array.reshape(length, -1)
        new_df = np.concatenate((df_array, new_subdf), axis=1)

        return pd.DataFrame(new_df, columns=list(df_data.columns) + [f"subdf_{i + 1}" for i in
                                                                     range(size * 3 * len(columns))])

    def new_data(self):
        df_merged = self._merge_data(self.df, self.subdf, 1,
                                     ['high-open', "open-close", "low-open", "high+dif"])  # 按需求连接五分钟行情和五个一分钟行情
        df_merged.to_csv('test.csv', index=False)
