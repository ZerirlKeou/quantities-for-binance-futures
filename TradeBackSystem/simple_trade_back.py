import pandas as pd
from visible import drawScatter as dk
import os
import sqlite3


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
        max_drawdown, peak, begin, end = self.calculate_maximum_drawdown()
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

    def plot_profit(self, stop_loss_percentage=0.072):
        """计算带有止损的回测
        : params : 亏损触发止损的百分比, 默认为7.2%
        """
        df = self.df
        df['Profit'] = 0.0
        df['change'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
        bought = False
        situation = "none"
        for i in range(len(df)):
            if not bought:
                if df["macd_back"] == 1:
                    bought = True
                    situation = "more"
                    df.loc[i, 'Profit'] = -self.money * self.premium * 0.01 * self.lever
                    self.money += df.loc[i, 'Profit']
                elif df["macd_back"] == -1:
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

def test():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_folder = "data\\data_base\\ETHUSDT.db"
    data_folder_path = os.path.join(parent_dir, data_folder)
    conn = sqlite3.connect(data_folder_path)

    df = pd.read_sql_query("select * from 'ETHUSDT';", conn, dtype='float',index_col='Open time')