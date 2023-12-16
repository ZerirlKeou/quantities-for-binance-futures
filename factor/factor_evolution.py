import json
import sqlite3
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
pool_path = os.path.join(parent_dir, "data\\future_pool.json")

def json_to_str():
    """
    获取合约池信息
    Returns:合约列表
    """
    with open(pool_path, 'r') as load_f:
        pair_index = json.load(load_f)
    return pair_index

def calculate_performance(df, conn, pair, indicator_col='macd_back', periods=None):
    if periods is None:
        periods = [1, 2, 3, 4, 5]
    results = {'pair':[], 'Period': [], 'Long Wins': [], 'Short Wins': [], 'Total Wins': [],
               'Long Win Percentage': [], 'Short Win Percentage': [], 'Total Win Percentage': [],
               'Long Total Return': [], 'Short Total Return': [], 'Total Return': []}

    for period in periods:
        long_condition = df[indicator_col] > 0
        short_condition = df[indicator_col] < 0
        long_condition_len = len(df[df[indicator_col] > 0])
        short_condition_len = len(df[df[indicator_col] < 0])

        long_wins = df[long_condition][f'Return_{period}'].gt(0).sum()
        short_wins = df[short_condition][f'Return_{period}'].lt(0).sum()
        total_wins = long_wins + short_wins

        long_win_percentage = long_wins / max(long_condition_len, 1) * 100
        short_win_percentage = short_wins / max(short_condition_len, 1) * 100
        total_win_percentage = total_wins / max((long_condition_len+short_condition_len),1) * 100

        long_total_return = df[long_condition][f'Return_{period}'].sum()
        short_total_return = -df[short_condition][f'Return_{period}'].sum()
        total_return = long_total_return + short_total_return

        results['pair'].append(pair)
        results['Period'].append(period)
        results['Long Wins'].append(long_wins)
        results['Short Wins'].append(short_wins)
        results['Total Wins'].append(total_wins)
        results['Long Win Percentage'].append(long_win_percentage)
        results['Short Win Percentage'].append(short_win_percentage)
        results['Total Win Percentage'].append(total_win_percentage)
        results['Long Total Return'].append(long_total_return)
        results['Short Total Return'].append(short_total_return)
        results['Total Return'].append(total_return)

    results_df = pd.DataFrame(results)

    results_df.to_sql(indicator_col, conn, index=False, if_exists='append')

def calculate_rate(interval, conn2, pair, name):
    """
    Args:
        conn2: factor_(interval).db
        interval: 时间级别
        pair: 交易对
        name: 测试的指标名字
    Returns: 返回测试胜率的平均值
    """
    data_folder = "data\\data_base\\{}\\{}.db".format(interval,pair)
    data_folder_path = os.path.join(parent_dir, data_folder)
    conn = sqlite3.connect(data_folder_path)
    try:
        df = pd.read_sql_query("select * from {};".format(pair), conn, dtype='float')
        calculate_performance(df,conn2,pair,indicator_col=name)
    except:
        print(u'{} database has not create, and it will be created soon'.format(pair))


def test():
    intervals = ['1m', '5m', '15m', '1h', '1d']
    names = ['basic_Open_point', 'macd_back', 'cci_open_point']
    pairs = json_to_str()
    for interval in intervals:
        factor_data_path = os.path.join(parent_dir, "data\\factor_base\\{}.db".format(interval))
        conn = sqlite3.connect(factor_data_path)
        cursor = conn.cursor()
        for name in names:
            try:
                cursor.execute(f"DELETE FROM {name};")
                conn.commit()
            except:
                pass
            for pair in pairs:
                calculate_rate(interval=interval, conn2=conn,pair=pair, name=name)
        conn.close()


