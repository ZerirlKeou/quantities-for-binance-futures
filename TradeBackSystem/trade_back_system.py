import pandas as pd
from visible import draw_klines as dk
import joblib
from grid_trader import grid_static


class LoadModel:
    def __init__(self, order, type='rf'):
        if type == 'rf':
            self.model = joblib.load('model\\randomforest\\random_forest_model' + str(order) + '.pkl')
        elif type == 'svm':
            self.model = joblib.load('model\\svm\\svm_model' + str(order) + '.pkl')

    def predict_rf(self, data):
        try:
            features = data.drop(["Open time", "Return_1", "Return_2"], axis=1)
            new_predictions = self.model.predict(features)
            return new_predictions
        except:
            print("Input False!!")
            return [0]


# 交易回测父类
class TradeBackSystem:
    def __init__(self, money, premium, interval, lever):
        self.original_money = money
        self.money = money
        self.lever = lever
        self.premium = premium
        self.interval = interval

    def _read_clean_data(self):
        self.path = "data/" + str(self.interval) + "_data.csv"
        df = pd.read_csv(self.path)
        df['change'] = (df['Close'] - df['Open']) / df['Close'].shift(1)
        return df

    def calculate_sharpe_ratio(self):
        df = self.df.copy()
        df.at[0, 'money'] = self.original_money

        for i in range(1, len(df)):
            df.at[i, 'money'] = df.at[i - 1, 'money'] + df.at[i, 'Profit']
            df.at[i, 'Daily Return'] = df.at[i, 'Profit'] / df.at[i - 1, 'money']
        mean_return = df['Daily Return'].mean()
        std_return = df['Daily Return'].std()

        risk_free_rate = 0  # Assuming risk-free rate is 0 for simplicity
        sharpe_ratio = (mean_return - risk_free_rate) / std_return
        return sharpe_ratio

    def calculate_maximum_drawdown(self):
        df = self.df.copy()
        interval = 0
        max_drawdown = 0
        begin = 0
        end = 0
        for index in range(len(df)):
            if index > 0:
                if df.loc[index, 'Cumulative Profit'] > df.loc[index - 1, "Cumulative Profit"]:
                    temp = df.loc[index, 'Cumulative Profit'] - df.loc[
                        index - 1 - interval, "Cumulative Profit"]
                    if max_drawdown < temp:
                        max_drawdown = temp
                        begin = index
                        end = index - 1 - interval
                    interval += 1
                else:
                    interval = 0
        Peak = df['Cumulative Profit'].cummax().tolist()
        return max_drawdown, Peak[-1:], begin, end

    def plot_profit(self):
        raise NotImplementedError("Subclasses must implement the 'plot_profit' method.")

    def run_backtest(self):
        self.plot_profit()
        sharpe_ratio = self.calculate_sharpe_ratio()
        max_drawdown, peak, begin, end = self.calculate_maximum_drawdown()
        print("Sharpe Ratio:", sharpe_ratio)
        print("Maximum Drawdown:", max_drawdown, "begin", begin, "end", end)
        print("Maximum Profit:", peak)


class TradeBackSystemV5(TradeBackSystem):
    """
    : params money : 初始投入资金
    : params premium : 交易手续费
    : params lever : 杠杆
    """

    def __init__(self, money, premium, lever):
        super().__init__(money, premium, '5m', lever)
        self.df = pd.read_csv('test.csv')
        """分类器：4"""
        self.lm = LoadModel(4, type='rf')

    def distinct(self, df):
        split_index = int(len(df) * 0.8)
        df = df[split_index:].reset_index(drop=True)
        return df

    def plot_profit1(self, stop_loss_percentage=0.072):
        """计算带有止损的回测
        : params : 亏损触发止损的百分比, 默认为7.2%
        """
        df = self.df
        df['Profit'] = 0.0
        df['prediction'] = 0
        # df['change'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['change'] = df['Return_1'].shift(1)
        df = self.distinct(df)
        bought = False
        situation = "none"
        for i in range(len(df)):
            row_df = pd.DataFrame([self.df.iloc[i]])
            prediction = self.lm.predict_rf(row_df)[0]
            df.loc[i, 'prediction'] = prediction
            if not bought:
                if prediction == 1:
                    bought = True
                    situation = "more"
                    df.loc[i, 'Profit'] = -self.money * self.premium * 0.01 * self.lever
                    self.money += df.loc[i, 'Profit']
                elif prediction == -1:
                    bought = True
                    situation = 'less'
                    df.loc[i, 'Profit'] = -self.money * self.premium * 0.01 * self.lever
                    self.money += df.loc[i, 'Profit']
            else:
                if situation == 'more':
                    if prediction == -1:
                        situation = "less"
                        df.loc[i, 'Profit'] += -self.money * self.premium * 0.01 * self.lever + self.money * df.loc[
                            i, 'change'] * self.lever
                        self.money += df.loc[i, 'Profit']
                    else:
                        df.loc[i, 'Profit'] += self.money * df.loc[i, 'change'] * self.lever
                        if df.loc[i, 'Profit'] <= -self.money * stop_loss_percentage:
                            df.loc[i, 'Profit'] = -self.money * stop_loss_percentage  # 卖出，设置亏损10.2%
                            self.money += df.loc[i, 'Profit']
                            bought = False
                        else:
                            self.money += df.loc[i, 'Profit']
                elif situation == 'less':
                    if prediction == 1:
                        situation = "more"
                        df.loc[i, 'Profit'] += -self.money * self.premium * 0.01 * self.lever - self.money * df.loc[
                            i, 'change'] * self.lever
                        self.money += df.loc[i, 'Profit']
                    else:
                        df.loc[i, 'Profit'] += -self.money * df.loc[i, 'change'] * self.lever
                        if df.loc[i, 'Profit'] <= -self.money * stop_loss_percentage:
                            df.loc[i, 'Profit'] = -self.money * stop_loss_percentage  # 卖出，设置亏损10.2%
                            self.money += df.loc[i, 'Profit']
                            bought = False
                        else:
                            self.money += df.loc[i, 'Profit']
        df['Cumulative Profit'] = df['Profit'].cumsum()
        print(self.money)
        df.to_csv("NEW_test.csv", index=None)
        chart = dk.StockChart2(
            'NEW_test.csv',
            "macd",
            ['Cumulative Profit', 'prediction'],
            themeplot=True,
            draw_number=200
        )
        chart.show_chart()

    def plot_profit(self):
        """计算普通地回测"""
        df = self.distinct(self.df)
        df['Profit'] = 0.0
        df['prediction'] = 0
        df['change'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
        bought = False
        situation = "none"
        for i in range(len(df)):
            # print(self.lm.predict_rf(self.df.iloc[i]))
            row_df = pd.DataFrame([self.df.iloc[i]])
            prediction = self.lm.predict_rf(row_df)[0]
            df.loc[i, 'prediction'] = prediction
            # 做多
            if not bought:
                # 开仓做多
                if prediction == 1:
                    bought = True
                    situation = "more"
                    df.loc[i, 'Profit'] = -self.money * self.premium * 0.01 * self.lever
                    self.money += df.loc[i, 'Profit']
                # 开仓做空
                elif prediction == -1:
                    bought = True
                    situation = 'less'
                    df.loc[i, 'Profit'] = -self.money * self.premium * 0.01 * self.lever
                    self.money += df.loc[i, 'Profit']
            else:
                # 做多情况
                if situation == 'more':
                    # 卖出
                    if prediction == -1:
                        # 反手做空
                        situation = "less"
                        df.loc[i, 'Profit'] += -self.money * self.premium * 0.01 * self.lever + self.money * df.loc[
                            i, 'change'] * self.lever
                        self.money += df.loc[i, 'Profit']
                    # 持有
                    else:
                        df.loc[i, 'Profit'] += self.money * df.loc[i, 'change'] * self.lever
                        self.money += df.loc[i, 'Profit']
                elif situation == 'less':
                    # 卖出
                    if prediction == 1:
                        situation = "more"
                        df.loc[i, 'Profit'] += -self.money * self.premium * 0.01 * self.lever - self.money * df.loc[
                            i, 'change'] * self.lever
                        self.money += df.loc[i, 'Profit']
                    # 持有
                    else:
                        df.loc[i, 'Profit'] += -self.money * df.loc[i, 'change'] * self.lever
                        self.money += df.loc[i, 'Profit']
        df['Cumulative Profit'] = df['Profit'].cumsum()
        print(self.money)
        df.to_csv("NEW_test.csv", index=None)
        chart = dk.StockChart2(
            'NEW_test.csv',
            "macd",
            ['Cumulative Profit', 'prediction'],
            themeplot=True,
            draw_number=900
        )
        chart.show_chart()


class GridTradeMulti(TradeBackSystem):
    """
        : params money : 初始投入资金
        : params premium : 交易手续费
        : params lever : 杠杆
        """

    def __init__(self, money, premium, lever):
        super().__init__(money, premium, '5m', lever)
        # self.df = pd.read_csv('test.csv')
        self.df = self._read_clean_data()
        """分类器：4"""
        # self.lm = LoadModel(4, type='rf')

    def distinct(self, df):
        split_index = int(len(df) * 0.8)
        df = df[split_index:].reset_index(drop=True)
        return df

    def plot_profit(self):
        """计算普通地回测"""
        # df = self.distinct(self.df)
        df = self.df
        per_money = self.money / 10
        buy_stack = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        sell_stack = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        buy_pointer = 0
        sell_pointer = 0
        df['Profit'] = 0.0
        df['prediction'] = 0
        for i in range(len(df)):
            # row_df = pd.DataFrame([self.df.iloc[i]])
            # prediction = self.lm.predict_rf(row_df)[0]
            # df.loc[i, 'prediction'] = prediction
            if df.loc[i, "macd_back"] == 1:
                if sell_pointer == 0:
                    if buy_pointer <= 14:
                        buy_stack[buy_pointer] = (per_money - per_money * self.premium * 0.01 * self.lever) / df.loc[
                            i, "Close"]
                        buy_pointer += 1
                else:
                    sell_pointer -= 1
                    df.loc[i, 'Profit'] = per_money - sell_stack[sell_pointer] * df.loc[i, "Close"]
                    self.money += df.loc[i, 'Profit']
            # 开仓做空
            elif df.loc[i, "macd_back"] == -1:
                if buy_pointer == 0:
                    if sell_pointer <= 14:
                        sell_stack[sell_pointer] = (per_money - per_money * self.premium * 0.01 * self.lever) / df.loc[
                            i, "Close"]
                        sell_pointer += 1
                else:
                    buy_pointer -= 1
                    df.loc[i, 'Profit'] = buy_stack[buy_pointer] * df.loc[i, "Close"] - per_money
                    self.money += df.loc[i, 'Profit']
        df['Cumulative Profit'] = df['Profit'].cumsum()
        print(self.money)
        df.to_csv("NEW_test.csv", index=None)
        chart = dk.StockChart2(
            'NEW_test.csv',
            "macd",
            ['Cumulative Profit', 'macd_back'],
            themeplot=True,
            draw_number=20000
        )
        chart.show_chart()

# 回测用法
# backtest_system = TradeBackSystem(money=41.24, premium=0.02, interval='5m', lever=8)
# backtest_system.run_backtest()


# class Trade:
#     def __init__(self, lever, premium):
#         self.lever = lever
#         self.premium = premium
#
#     def fees(self, money):
#         return money * self.premium * 0.01 * self.lever
#
#     def maker(self, money, df, i):
#         position = (money * self.lever) / df.loc[i, "Close"]
#         df.loc[i, 'Profit'] = -self.fees(money)
#         money += df.loc[i, 'Profit']
#         return position, money, df
#
#     def position(self, maker_type, position, money, df, i):
#         if maker_type == "long":
#             df.loc[i, 'Profit'] += position * df.loc[
#                 i, 'Close'] / self.lever - money
#             money += df.loc[i, 'Profit']
#         else:
#             df.loc[i, 'Profit'] += money - position * df.loc[
#                 i, 'Close'] / self.lever
#             money += df.loc[i, 'Profit']
#         return money, df


# money:usdt,premium:% now for the maker
# class TradeBackSystemV1(TradeBackSystem):
#     def __init__(self, money, premium, interval, lever):
#         super().__init__(money, premium, interval, lever)
#         self.trade = Trade(lever, premium)
#         self.df = super()._read_clean_data()
#
#     def plot_profit(self):
#         df = self.df.copy()
#         self.position = 0
#         count = 0
#         df['Profit'] = 0.0
#         bought = False
#         situation = "none"
#         for i in range(len(df)):
#             # 做多
#             if not bought:
#                 # 开仓做多
#                 if df.loc[i, 'basic_Open_point'] == 1:
#                     bought = True
#                     situation = "more"
#                     self.position, self.money, df = self.trade.maker(money=self.money, df=df, i=i)
#                 # 开仓做空
#                 elif df.loc[i, 'basic_Open_point'] == -1:
#                     bought = True
#                     situation = 'less'
#                     self.position, self.money, df = self.trade.maker(money=self.money, df=df, i=i)
#             else:
#                 # 做多情况
#                 if situation == 'more':
#                     # 先持有更新
#                     self.money, df = self.trade.position(maker_type='short', position=self.position,
#                                                          money=self.money, df=df, i=i)
#                     # 卖出
#                     if df.loc[i, 'macd_sell_point'] == -1:
#                         bought = False
#                         self.position = 0
#                 elif situation == 'less':
#                     # 先持有更新
#                     self.money, df = self.trade.position(maker_type='long', position=self.position,
#                                                          money=self.money, df=df, i=i)
#                     # 卖出
#                     if df.loc[i, 'macd_sell_point'] == 1:
#                         bought = False
#                         self.position = 0
#         df['Profit'].to_csv("test")
#         df['Cumulative Profit'] = df['Profit'].cumsum()
#         print(self.money)
#         df.to_csv(self.path, index=None)
#         chart = dk.StockChart(
#             self.interval,
#             "macd",
#             ["Cumulative Profit", 'basic_Open_point', "macd_sell_point"],
#             themeplot=True,
#             draw_number=-1700
#         )
#         chart.show_chart()
#         return count
#
#
# def Two_kline_strategy(bought, situation, df, i, count, money, lever, premium):
#     # 做多情况
#     if situation == 'more':
#         # 卖出
#         if i - count == 2:
#             bought = False
#             df.loc[i, 'Profit'] = -money * premium * 0.01 * lever + money * df.loc[
#                 i, 'change'] * lever
#             money += df.loc[i, 'Profit']
#         # 持有
#         else:
#             df.loc[i, 'Profit'] = money * df.loc[i, 'change'] * lever
#             money += df.loc[i, 'Profit']
#     elif situation == 'less':
#         # 卖出
#         if i - count == 2:
#             bought = False
#             df.loc[i, 'Profit'] = -money * premium * 0.01 * lever - money * df.loc[
#                 i, 'change'] * lever
#             money += df.loc[i, 'Profit']
#         # 持有
#         else:
#             df.loc[i, 'Profit'] = -money * df.loc[i, 'change'] * lever
#             money += df.loc[i, 'Profit']
#     return bought, money, df
#
#
# class TradeBackSystemV2(TradeBackSystem):
#     def __init__(self, money, premium, interval, lever):
#         super().__init__(money, premium, interval, lever)
#         self.df = super()._read_clean_data()
#
#     def plot_profit(self):
#         df = self.df.copy()
#         df['Profit'] = 0.0
#         bought = False
#         situation = "none"
#         count = 0
#         for i in range(len(df)):
#             # 做多
#             if not bought:
#                 # 开仓做多
#                 if df.loc[i, 'basic_Open_point'] == 1:
#                     bought = True
#                     situation = "more"
#                     count = i
#                     df.loc[i, 'Profit'] = -self.money * self.premium * 0.01 * self.lever
#                     self.money += df.loc[i, 'Profit']
#                 # 开仓做空
#                 elif df.loc[i, 'basic_Open_point'] == -1:
#                     bought = True
#                     situation = 'less'
#                     count = i
#                     df.loc[i, 'Profit'] = -self.money * self.premium * 0.01 * self.lever
#                     self.money += df.loc[i, 'Profit']
#             else:
#                 bought, self.money, df = Two_kline_strategy(bought, situation=situation, df=df, i=i, count=count,
#                                                             money=self.money,
#                                                             lever=self.lever,
#                                                             premium=self.premium)
#                 # # 做多情况
#                 # if situation == 'more':
#                 #     # 卖出
#                 #     if df.loc[i, 'macd_sell_point'] == -1:
#                 #         # 反手做空
#                 #         situation = "less"
#                 #         df.loc[i, 'Profit'] += -self.money * self.premium * 0.01 * self.lever + self.money * df.loc[
#                 #             i, 'change'] * self.lever
#                 #         self.money += df.loc[i, 'Profit']
#                 #     # 持有
#                 #     else:
#                 #         df.loc[i, 'Profit'] += self.money * df.loc[i, 'change'] * self.lever
#                 #         self.money += df.loc[i, 'Profit']
#                 # elif situation == 'less':
#                 #     # 卖出
#                 #     if df.loc[i, 'macd_sell_point'] == 1:
#                 #         situation = "more"
#                 #         df.loc[i, 'Profit'] += -self.money * self.premium * 0.01 * self.lever - self.money * df.loc[
#                 #             i, 'change'] * self.lever
#                 #         self.money += df.loc[i, 'Profit']
#                 #     # 持有
#                 #     else:
#                 #         df.loc[i, 'Profit'] += -self.money * df.loc[i, 'change'] * self.lever
#                 #         self.money += df.loc[i, 'Profit']
#
#         df['Cumulative Profit'] = df['Profit'].cumsum()
#         print("Final Money:", self.money)
#         df.to_csv(self.path, index=None)
#         chart = dk.StockChart(
#             self.interval,
#             "macd",
#             ["Cumulative Profit", 'basic_Open_point', "macd_sell_point"],
#             themeplot=True,
#             draw_number=200
#         )
#         chart.show_chart()
#         return count
#
#
# class TradeBackSystemV3(TradeBackSystem):
#     def __init__(self, money, premium, interval1, interval2, lever):
#         super().__init__(money, premium, interval1, lever)
#         self.interval2 = interval2
#         self.df, self.subdf = self._read_clean_data()
#
#     def _read_clean_data(self):
#         df = super()._read_clean_data()
#         self.path2 = "data/" + str(self.interval2) + "_data.csv"
#         subdf = pd.read_csv(self.path2)
#         subdf['change'] = (subdf['Close'] - subdf['Open']) / subdf['Close'].shift(1)
#         return df, subdf
#
#     def _find_kline(self, subdf, df, i):
#         interrupt = True
#         index = None
#         try:
#             time = df.loc[i, "Open time"]
#             index = subdf.loc[subdf['Open time'] == time].index[0]
#             interrupt = False
#         except:
#             pass
#         return index, interrupt
#
#     def _find_beginning_position(self, df, subdf, start_row=0):
#         subdf_times = set(subdf["Open time"])
#         df_subset = df.iloc[start_row:]  # 从指定行开始创建 df 的子集
#
#         for index_df, row_df in df_subset.iterrows():
#             if row_df["Open time"] in subdf_times:
#                 for index, row in subdf.iterrows():
#                     if row["Open time"] == row_df["Open time"]:
#                         return index_df, index + start_row  # 返回实际索引位置，加上起始行的偏移量
#
#         return None
#
#     def plot_profit(self):
#         df = self.df.copy()
#         subdf = self.subdf.copy()
#         count = 0
#         df['Profit'] = 0.0
#         bought = False
#         situation = "none"
#         index_df, j = self._find_beginning_position(df, subdf)
#
#         for i in range(index_df, len(df)):
#             if not bought:
#                 # 开仓做多
#                 if df.loc[i, 'basic_Open_point'] == 1:
#                     print("ok")
#                     bought = True
#                     situation = "more"
#                     df.loc[i, 'Profit'] = -self.money * self.premium * 0.01 * self.lever
#                     self.money += df.loc[i, 'Profit']
#                 # 开仓做空
#                 elif df.loc[i, 'basic_Open_point'] == -1:
#                     bought = True
#                     situation = 'less'
#                     df.loc[i, 'Profit'] = -self.money * self.premium * 0.01 * self.lever
#                     self.money += df.loc[i, 'Profit']
#                     print(self.money)
#
#             else:
#                 # 做多情况
#                 if situation == 'more':
#                     print(self.money)
#                     k, skip = self._find_kline(subdf, df, i)
#                     if not skip:
#                         try:
#                             for j in range(5):
#                                 # 卖出
#                                 if subdf.loc[k + j, 'macd_sell_point'] == -1:
#                                     bought = False
#                                     # # 反手做空
#                                     # situation = "less"
#                                     subdf.loc[
#                                         k + j, 'Profit'] += -self.money * self.premium * 0.01 * self.lever + self.money * \
#                                                             subdf.loc[
#                                                                 k + j, 'change'] * self.lever
#                                     self.money += subdf.loc[k + j, 'Profit']
#                                     df.loc[i, 'Profit'] += subdf.loc[k + j, 'Profit']
#                                 # 持有
#                                 else:
#                                     subdf.loc[k + j, 'Profit'] += self.money * subdf.loc[k + j, 'change'] * self.lever
#                                     self.money += subdf.loc[k + j, 'Profit']
#                                     df.loc[i, 'Profit'] += subdf.loc[k + j, 'Profit']
#                         except:
#                             pass
#                     else:
#                         bought = False
#                         df.loc[i, 'Profit'] += -self.money * self.premium * 0.01 * self.lever + self.money * \
#                                                df.loc[
#                                                    i, 'change'] * self.lever
#                         self.money += df.loc[i, 'Profit']
#                         df.loc[i, 'Profit'] += df.loc[i, 'Profit']
#
#                 elif situation == 'less':
#                     k, skip = self._find_kline(subdf, df, i)
#                     if not skip:
#                         try:
#                             for j in range(5):
#                                 # 卖出
#                                 if subdf.loc[k + j, 'macd_sell_point'] == 1:
#                                     # situation = "more"
#                                     subdf.loc[
#                                         k + j, 'Profit'] += -self.money * self.premium * 0.01 * self.lever - self.money * \
#                                                             subdf.loc[
#                                                                 k + j, 'change'] * self.lever
#                                     self.money += subdf.loc[k + j, 'Profit']
#                                     df.loc[i, 'Profit'] += subdf.loc[k + j, 'Profit']
#                                 # 持有
#                                 else:
#                                     subdf.loc[k + j, 'Profit'] += -self.money * subdf.loc[k + j, 'change'] * self.lever
#                                     self.money += subdf.loc[k + j, 'Profit']
#                                     df.loc[i, 'Profit'] += subdf.loc[k + j, 'Profit']
#                         except:
#                             pass
#                     else:
#                         df.loc[i, 'Profit'] += -self.money * self.premium * 0.01 * self.lever - self.money * \
#                                                df.loc[
#                                                    i, 'change'] * self.lever
#                         self.money += df.loc[i, 'Profit']
#                         df.loc[i, 'Profit'] += df.loc[i, 'Profit']
#         df['Cumulative Profit'] = df['Profit'].cumsum()
#         print("Final Money:", self.money)
#         df.to_csv(self.path, index=None)
#         chart = dk.StockChart(
#             self.interval,
#             "macd",
#             ["Cumulative Profit", 'basic_Open_point', "macd_sell_point"],
#             themeplot=True,
#             draw_number=-1500
#         )
#         chart.show_chart()
#         return count
#
#
# # model
# class TradeBackSystemV4(TradeBackSystem):
#     def __init__(self, money, premium, interval, lever):
#         super().__init__(money, premium, interval, lever)
#         self.df = super()._read_clean_data()
#         self.ml = LoadModel(4)
#
#     def plot_profit(self):
#         df = self.df.copy()
#         df['Profit'] = 0.0
#         bought = False
#         situation = "none"
#         for i in range(1500, len(df)):
#             row_df = pd.DataFrame([self.df.iloc[i]])
#             prediction = self.ml.predict_rf(row_df)[0]
#             # 做多
#             if not bought:
#                 # 开仓做多
#                 if prediction == 1:
#                     bought = True
#                     situation = "more"
#                     df.loc[i, 'Profit'] = -self.money * self.premium * 0.01 * self.lever
#                     self.money += df.loc[i, 'Profit']
#                 # 开仓做空
#                 elif prediction == -1:
#                     bought = True
#                     situation = 'less'
#                     df.loc[i, 'Profit'] = -self.money * self.premium * 0.01 * self.lever
#                     self.money += df.loc[i, 'Profit']
#             else:
#                 # 做多情况
#                 if situation == 'more':
#                     # 卖出
#                     if prediction == -1:
#                         # 反手做空
#                         situation = "less"
#                         df.loc[i, 'Profit'] += -self.money * self.premium * 0.01 * self.lever + self.money * df.loc[
#                             i, 'change'] * self.lever
#                         self.money += df.loc[i, 'Profit']
#                     # 持有
#                     else:
#                         df.loc[i, 'Profit'] += self.money * df.loc[i, 'change'] * self.lever
#                         self.money += df.loc[i, 'Profit']
#                 elif situation == 'less':
#                     # 卖出
#                     if prediction == 1:
#                         situation = "more"
#                         df.loc[i, 'Profit'] += -self.money * self.premium * 0.01 * self.lever - self.money * df.loc[
#                             i, 'change'] * self.lever
#                         self.money += df.loc[i, 'Profit']
#                     # 持有
#                     else:
#                         df.loc[i, 'Profit'] += -self.money * df.loc[i, 'change'] * self.lever
#                         self.money += df.loc[i, 'Profit']
#         df['Profit'].to_csv("test")
#         df['Cumulative Profit'] = df['Profit'].cumsum()
#         print(self.money)
#         df.to_csv(self.path, index=None)
#         chart = dk.StockChart(
#             self.interval,
#             "macd",
#             ["Cumulative Profit", 'basic_Open_point', "macd_sell_point"],
#             themeplot=True,
#             draw_number=2000
#         )
#         chart.show_chart()
