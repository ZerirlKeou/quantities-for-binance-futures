"""
建立纯指标的回测系统，不可用于机器学习这种具有时间序列要求的回测框架
更趋近于策略的设计
"""
import pandas as pd
from visible import drawMainFigure
import os
import sqlite3
import numpy as np

app = drawMainFigure.MplVisualIf()


def _read_clean_data(df, step):
    """
    截取并清洗数据，计算其涨跌幅值
    """
    total_rows = len(df)
    df = df.iloc[total_rows - step - 1:total_rows]
    df['change'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['Profit'] = 0.0
    return df.iloc[1:]


def shift_right(lst, fill_value):
    return [fill_value] + lst[:-1]


class IndicatorTradeBackSystem:
    """
    计算带有止损的回测, 纯指标形回测
    :params money : 初始投入资金
    :params premium : 交易手续费
    :params lever : 杠杆
    :params position : 持仓方向，可选long, short, both
    :params step: 回测多少步，默认从最新的数据往前数开始进行回测
    :params stop_loss_percentage : 止损比例，默认7.2%，可接受传参0.072 float
    """

    def __init__(self, df, money, premium, lever, position, step, stop_loss_percentage=0.1):
        self.count = None
        self.open_money = None
        self.money = money
        df['money'] = money
        self.lever = lever
        self.premium = premium
        self.position = position
        self.df = _read_clean_data(df, step)
        self.fee_multiplier = self.premium * 0.01 * self.lever
        self.plot_profit(indicator_name="macd_sell_time_series", stop_loss_percentage=stop_loss_percentage)

    def open_position(self, df, i, profit_column_index):
        df.iloc[i, profit_column_index] = -self.money * self.fee_multiplier
        self.open_money = self.money
        self.money += df.iloc[i]['Profit']
        return True, df

    def _open_position(self, df, i, profit_column_index, times):
        times -= 1
        df.iloc[i, profit_column_index] = -self.money[times] * self.fee_multiplier
        self.open_money[times] = self.money[times]
        self.money += df.iloc[i]['Profit']
        return times, df

    def plot_profit(self, indicator_name="basic_Open_point", stop_loss_percentage=0.1):
        """
        顺序执行算法，将时间步考虑其中
        Args:
            indicator_name: 需要测试的指标名称
            stop_loss_percentage: 止损线，默认为7.2%
        """
        df = self.df
        profit_column_index = df.columns.get_loc('Profit')
        money_column_index = df.columns.get_loc('money')

        # 开仓策略函数
        # df['macd_sell_time_series'] = -df['macd_sell_time_series']
        # df = self.make_double(df, profit_column_index, money_column_index, indicator_name, stop_loss_percentage)
        # df = self.make_double_position_for_double_indicator(df, profit_column_index, money_column_index, indicator_name, indicator_two='macd_sell', stop_loss_percentage=stop_loss_percentage)
        # df = self.make_single(df, profit_column_index, money_column_index, indicator_name, stop_loss_percentage,position='long')
        # df = self.make_single_position_for_double_indicator(df, profit_column_index, money_column_index, indicator_name,
        #                                                     indicator_two='macd_sell',
        #                                                     stop_loss_percentage=stop_loss_percentage, position='long')
        # df = self.make_grid_positions(df, profit_column_index, money_column_index, indicator_name, stop_loss_percentage, times=3)
        df = self.fixed_time_sell(df, profit_column_index, money_column_index, indicator_name, stop_loss_percentage,
                                  times=3)

        # 可视化
        df['Open time'] = (df.index.astype(np.int64) // 10 ** 3).astype(int)

        layout_dict = {'df': df,
                       'draw_kind': ['kline', 'position', 'macd_sell_time_series', 'money'],
                       'title': f"{indicator_name}指标策略回测"}
        app.fig_output(**layout_dict)

    def make_double(self, df, profit_column_index, money_column_index, indicator_name, stop_loss_percentage):
        df['position'] = 0.0
        position_column_index = df.columns.get_loc('position')
        bought = False
        situation = "none"
        for i in range(len(df)):
            if not bought:
                # 开仓
                if df.iloc[i][indicator_name] == 1:
                    situation = "more"
                    bought, df = self.open_position(df, i, profit_column_index)
                elif df.iloc[i][indicator_name] == -1:
                    situation = 'less'
                    bought, df = self.open_position(df, i, profit_column_index)
            else:
                if situation == 'more':
                    df.iloc[i, profit_column_index] += self.money * df.iloc[i]['change'] * self.lever
                    if df.iloc[i][indicator_name] == -1:
                        situation = "none"
                        df.iloc[i, profit_column_index] += -self.money * self.fee_multiplier
                        bought = False
                    elif self.money + df.iloc[i]['Profit'] <= self.open_money * (1 - stop_loss_percentage):
                        df.iloc[i, profit_column_index] = -self.money * stop_loss_percentage  # 达到止损线卖出
                        situation = "none"
                        bought = False
                    self.money += df.iloc[i]['Profit']
                elif situation == 'less':
                    df.iloc[i, profit_column_index] += -self.money * df.iloc[i]['change'] * self.lever
                    if df.iloc[i][indicator_name] == 1:
                        situation = "none"
                        df.iloc[i, profit_column_index] += -self.money * self.fee_multiplier
                        bought = False
                    elif self.money + df.iloc[i]['Profit'] <= self.open_money * (1 - stop_loss_percentage):
                        df.iloc[i, profit_column_index] = -self.money * stop_loss_percentage  # 达到止损线卖出
                        situation = "none"
                        bought = False
                    self.money += df.iloc[i]['Profit']
            df.iloc[i, money_column_index] = self.money
        print('资金余额：', self.money)
        df.to_csv("双向开仓回测结果.csv", index=None)
        return df

    def make_double_position_for_double_indicator(self, df, profit_column_index, money_column_index, indicator_one,
                                                  indicator_two, stop_loss_percentage):
        bought = False
        situation = "none"
        for i in range(len(df)):
            if not bought:
                # 开仓
                if df.iloc[i][indicator_one] == 1:
                    situation = "more"
                    bought = self.open_position(df, i, profit_column_index)
                elif df.iloc[i][indicator_one] == -1:
                    situation = 'less'
                    bought = self.open_position(df, i, profit_column_index)
            else:
                if situation == 'more':
                    df.iloc[i, profit_column_index] += self.money * df.iloc[i]['change'] * self.lever
                    if df.iloc[i][indicator_two] == -1:
                        situation = "none"
                        df.iloc[i, profit_column_index] += -self.money * self.fee_multiplier
                        bought = False
                    elif self.money + df.iloc[i]['Profit'] <= self.open_money * (1 - stop_loss_percentage):
                        df.iloc[i, profit_column_index] = -self.money * stop_loss_percentage  # 达到止损线卖出
                        situation = "none"
                        bought = False
                    self.money += df.iloc[i]['Profit']
                elif situation == 'less':
                    df.iloc[i, profit_column_index] += -self.money * df.iloc[i]['change'] * self.lever
                    if df.iloc[i][indicator_two] == 1:
                        situation = "none"
                        df.iloc[i, profit_column_index] += -self.money * self.fee_multiplier
                        bought = False
                    elif self.money + df.iloc[i]['Profit'] <= self.open_money * (1 - stop_loss_percentage):
                        df.iloc[i, profit_column_index] = -self.money * stop_loss_percentage  # 达到止损线卖出
                        situation = "none"
                        bought = False
                    self.money += df.iloc[i]['Profit']
            df.iloc[i, money_column_index] = self.money
        print('资金余额：', self.money)
        df.to_csv("双指标双向开仓回测结果.csv", index=None)
        return df

    def make_single(self, df, profit_column_index, money_column_index, indicator_name, stop_loss_percentage,
                    position='long'):
        if position == 'long':
            indicator_number = 1
        else:
            indicator_number = -1
        bought = False
        for i in range(len(df)):
            if not bought:
                # 开仓
                if df.iloc[i][indicator_name] == indicator_number:
                    bought = self.open_position(df, i, profit_column_index)
            else:
                df.iloc[i, profit_column_index] += indicator_number * self.money * df.iloc[i]['change'] * self.lever
                if df.iloc[i][indicator_name] == -indicator_number:
                    df.iloc[i, profit_column_index] += -self.money * self.fee_multiplier
                    bought = False
                elif self.money + df.iloc[i]['Profit'] <= self.open_money * (1 - stop_loss_percentage):
                    df.iloc[i, profit_column_index] = -self.money * stop_loss_percentage  # 达到止损线卖出
                    bought = False
                self.money += df.iloc[i]['Profit']
            df.iloc[i, money_column_index] = self.money
        print('资金余额：', self.money)
        df.to_csv(f"单向{position}开仓回测结果.csv", index=None)
        return df

    def make_single_position_for_double_indicator(self, df, profit_column_index, money_column_index, indicator_one,
                                                  indicator_two, stop_loss_percentage, position='long'):
        if position == 'long':
            indicator_number = 1
        else:
            indicator_number = -1
        bought = False
        for i in range(len(df)):
            if not bought:
                # 开仓
                if df.iloc[i][indicator_one] == indicator_number:
                    bought = self.open_position(df, i, profit_column_index)
            else:
                df.iloc[i, profit_column_index] += indicator_number * self.money * df.iloc[i]['change'] * self.lever
                if df.iloc[i][indicator_two] == -indicator_number:
                    df.iloc[i, profit_column_index] += -self.money * self.fee_multiplier
                    bought = False
                elif self.money + df.iloc[i]['Profit'] <= self.open_money * (1 - stop_loss_percentage):
                    df.iloc[i, profit_column_index] = -self.money * stop_loss_percentage  # 达到止损线卖出
                    bought = False
                self.money += df.iloc[i]['Profit']
            df.iloc[i, money_column_index] = self.money
        print('资金余额：', self.money)
        df.to_csv(f"单向{position}双指标开仓回测结果.csv", index=None)
        return df

    def make_grid_positions(self, df, profit_column_index, money_column_index, indicator_name, stop_loss_percentage,
                            times=30):
        df['position'] = times
        position_column_index = df.columns.get_loc('position')
        situation = "none"
        origin_times = times
        one_pit = self.money / times
        total_money = self.money
        self.money = [one_pit for _ in range(times)]
        self.open_money = self.money

        for i in range(len(df)):
            if times == origin_times:
                # 开仓
                if df.iloc[i][indicator_name] == 1:
                    situation = "more"
                    times, df = self._open_position(df, i, profit_column_index, times)
                elif df.iloc[i][indicator_name] == -1:
                    situation = 'less'
                    times, df = self._open_position(df, i, profit_column_index, times)
            else:
                if situation == 'more':
                    for j in range(origin_times - times):
                        df.iloc[i, profit_column_index] += self.money[times - j] * df.iloc[i]['change'] * self.lever
                    if df.iloc[i][indicator_name] == -1:
                        df.iloc[i, profit_column_index] += -self.money[times] * self.fee_multiplier
                        times += 1
                        if times == origin_times:
                            situation = "none"
                    elif self.money[times] + df.iloc[i]['Profit'] <= self.open_money[times] * (
                            1 - stop_loss_percentage):
                        df.iloc[i, profit_column_index] += -self.money[times] * (
                                    stop_loss_percentage + self.fee_multiplier)  # 达到止损线卖出
                        times += 1
                        if times == origin_times:
                            situation = "none"
                    # if df.iloc[i][indicator_name] == 1:
                    #     times -= 1
                    #     df.iloc[i, profit_column_index] += -self.money[times] * self.fee_multiplier
                    total_money += df.iloc[i]['Profit']
                elif situation == 'less':
                    for j in range(origin_times - times):
                        df.iloc[i, profit_column_index] += -self.money[times - j] * df.iloc[i]['change'] * self.lever
                    if df.iloc[i][indicator_name] == 1:
                        df.iloc[i, profit_column_index] += -self.money[times] * self.fee_multiplier
                        times += 1
                        if times == origin_times:
                            situation = "none"
                    elif self.money[times] + df.iloc[i]['Profit'] <= self.open_money[times] * (
                            1 - stop_loss_percentage):
                        df.iloc[i, profit_column_index] += -self.money[times] * (
                                    stop_loss_percentage + self.fee_multiplier)  # 达到止损线卖出
                        times += 1
                        if times == origin_times:
                            situation = "none"
                    # if df.iloc[i][indicator_name] == -1:
                    #     times -= 1
                    #     df.iloc[i, profit_column_index] += -self.money[times] * self.fee_multiplier
                    total_money += df.iloc[i]['Profit']
            df.iloc[i, position_column_index] = times
            df.iloc[i, money_column_index] = total_money
        print('资金余额：', total_money)
        df.to_csv("老鼠仓回测结果.csv", index=None)
        return df

    def fixed_time_sell(self, df, profit_column_index, money_column_index, indicator_name, stop_loss_percentage,
                        times=3):
        df['position'] = times
        position_column_index = df.columns.get_loc('position')
        situation = "none"
        origin_times = times
        one_pit = self.money / times
        total_money = self.money
        self.money = [one_pit for _ in range(times)]
        print(self.money)
        self.count = [0 for _ in range(times)]
        self.open_money = self.money
        sell = False

        for i in range(len(df)):
            if times == origin_times:
                # 开仓
                if df.iloc[i][indicator_name] == 1:
                    situation = "more"
                    times, df = self._open_position(df, i, profit_column_index, times)
                elif df.iloc[i][indicator_name] == -1:
                    situation = 'less'
                    times, df = self._open_position(df, i, profit_column_index, times)
            else:
                if situation == 'more':
                    for j in range(origin_times - times):
                        order = origin_times - j - 1
                        self.money[order] += self.money[order] * df.iloc[i][
                            'change'] * self.lever
                        # 先计算止损
                        if self.money[order] <= self.open_money[order] * (
                                1 - stop_loss_percentage):
                            df.iloc[i, profit_column_index] += -self.money[order] * (
                                    stop_loss_percentage + self.fee_multiplier)  # 达到止损线卖出
                            sell = True
                        else:
                            df.iloc[i, profit_column_index] += self.money[order] - self.open_money[order]
                        # 再计算当前开仓的order有没有已经过了5个时间
                        if self.count[order] == 3:
                            df.iloc[i, profit_column_index] += -self.money[order] * self.fee_multiplier
                            sell = True
                        if sell:
                            sell = False
                            times += 1
                            self.count = shift_right(self.count, 0)
                            self.money = np.insert(self.money, 0, self.open_money[order])[:-1]
                    if df.iloc[i][indicator_name] == 1 and times > 0:
                        times -= 1
                        df.iloc[i, profit_column_index] += -self.money[times] * self.fee_multiplier
                    total_money += df.iloc[i]['Profit']

                elif situation == 'less':
                    for j in range(origin_times - times):
                        order = origin_times - j - 1
                        self.money[order] -= self.money[order] * df.iloc[i]['change'] * self.lever
                        if self.money[order] <= self.open_money[order] * (
                                1 - stop_loss_percentage):
                            df.iloc[i, profit_column_index] += -self.money[times] * (
                                    stop_loss_percentage + self.fee_multiplier)  # 达到止损线卖出
                            sell = True
                        else:
                            df.iloc[i, profit_column_index] += self.money[order] - self.open_money[order]
                        if self.count[order] == 3:
                            df.iloc[i, profit_column_index] += -self.money[times] * self.fee_multiplier
                            sell = True
                        if sell:
                            sell = False
                            times += 1
                            self.count = shift_right(self.count, 0)
                            self.money = np.insert(self.money, 0, self.open_money[order])[:-1]
                    if df.iloc[i][indicator_name] == -1 and times > 0:
                        times -= 1
                        df.iloc[i, profit_column_index] += -self.money[times] * self.fee_multiplier
                    total_money += df.iloc[i]['Profit']

            for k in range(origin_times - times):
                self.count[origin_times - k - 1] += 1
            df.iloc[i, position_column_index] = times
            df.iloc[i, money_column_index] = total_money
        print('资金余额：', total_money)
        df.to_csv("固定天数回测结果.csv", index=None)
        return df


def test():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_folder = "data\\data_base\\1m\\BTCUSDT.db"
    data_folder_path = os.path.join(parent_dir, data_folder)
    conn = sqlite3.connect(data_folder_path)

    df = pd.read_sql_query("select * from 'BTCUSDT';", conn, dtype='float', index_col='Open time')
    IndicatorTradeBackSystem(df, money=50, premium=0.018, lever=6, position='long', step=800000,
                             stop_loss_percentage=0.072)
