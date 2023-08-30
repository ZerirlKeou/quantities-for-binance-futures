import numpy as np


class AddBuyPoint:
    def __init__(self):
        pass

    def macd_back(self, df):
        # df['normal_macd_back'] = np.where(df['macd_condition'] & df['shifted_condition1'] & df[
        #     'shifted_condition2'], -1,
        #                                   np.where(~df['macd_condition'] & ~df['shifted_condition1'] & ~df[
        #                                       'shifted_condition2'], 1, 0))
        # df['macd_sell_point'] = np.where((df['normal_macd_back'] == -1) | (
        #         ~df['macd_condition'] & df['shifted_condition1'] & df['shifted_condition2']), -1, np.where(
        #     (df['normal_macd_back'] == 1) | (
        #                 df['macd_condition'] & ~df['shifted_condition1'] & ~df['shifted_condition2']), 1, 0))
        # df['macd_back'] = np.where(
        #     df['deaRewrite'] & df['difRewrite'] & (df['normal_macd_back'] == -1), -1,
        #     np.where(~df['deaRewrite'] & ~df['difRewrite'] & (df['normal_macd_back'] == 1), 1, 0))
        df['macd_condition'] = df['macd'] > 0
        df['shifted_condition1'] = df['macd'] < df['macd'].shift(1)
        df['shifted_condition2'] = df['macd'].shift(1) > df['macd'].shift(2)
        df['deaRewrite'] = df['dea'] > df['macd']
        df['difRewrite'] = df['dif'] > df['macd']

        df['macd_back'] = np.where(
            df['deaRewrite'] & df['difRewrite'] & (
                    df['macd_condition'] & df['shifted_condition1'] & df['shifted_condition2']),
            -1,
            np.where(~df['deaRewrite'] & ~df['difRewrite'] & (
                    ~df['macd_condition'] & ~df['shifted_condition1'] & ~df['shifted_condition2']),
                     1, 0))
        return df

    def williams_point1(self, df):
        df['williams_condition1'] = df['williams_r'].shift(1) < -80
        df['williams_condition2'] = df['williams_r'] > -80
        df['williams_condition3'] = df['williams_r'].shift(1) > -20
        df['williams_condition4'] = df['williams_r'] < -20
        df['williams_points'] = np.where(df['williams_condition1'] & df['williams_condition2'], 1,
                                         np.where(df['williams_condition3'] & df['williams_condition4'], -1, 0))
        return df

    def basic_buy_point(self, df):
        df['basic_Open_point'] = np.where((df['macd_back'] == 1) & (df['williams_points'] == 1), 1,
                                          np.where((df['macd_back'] == -1) & (df['williams_points'] == -1), -1, 0))
        return df

    def williams_point(self, df):
        df['williams_1'] = (-df['williams_r'].shift(1) - 80) / (-df['williams_r'] - 80)
        df['low-open'] = df['Low'] - df['Open'] / df['Close']
        df['high-open'] = df['High'] - df['Open'] / df["Close"]
        """2023.07.16目前为止最好的一个指标"""
        df['open-close'] = (df['Open'] - df['Close']) / df["Close"]
        df['CCI_HHV'] = np.where(df["CCI"] >= 0, df['CCI'].rolling(3).max() - df['CCI'],
                                 -df['CCI'].rolling(3).max() + df['CCI'])
        df["chazhi"] = df["CCI_HHV"].shift(1) - df["CCI_HHV"]
        df["high+dif"] = df["High"] + df["dif"]
        return df
