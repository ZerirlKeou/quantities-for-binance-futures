import json
import sqlite3
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

def json_to_str():
    """
    获取合约池信息
    Returns:合约列表
    """
    pool_path = os.path.join(parent_dir, "data\\future_pool.json")
    with open(pool_path, 'r') as load_f:
        pair_index = json.load(load_f)
    return pair_index

def calculate_rate(interval, pair, name, returns_columns=None, statistic='mean'):
    """
    Args:
        interval: 时间级别
        pair: 交易对
        name: 测试的指标名字
        returns_columns: 回报率表格
        statistic: 测试指标选择（后期删除，都要测试）
    Returns: 返回测试胜率的平均值
    """
    if returns_columns is None:
        returns_columns = ['Return_1', 'Return_2', 'Return_3', 'Return_4', 'Return_5']
    rates_mean= []
    rates_median = []
    data_folder = "data\\data_base\\{}\\{}.db".format(interval,pair)
    data_folder_path = os.path.join(parent_dir, data_folder)
    conn = sqlite3.connect(data_folder_path)
    try:
        df = pd.read_sql_query("select * from {};".format(pair), conn, dtype='float')
    except:
        print(u'{} database has not create, and it will be created soon'.format(pair))
        return None

    for col in returns_columns:
        if statistic == 'mean':
            rates_mean.append(df[df[name] > 0][col].mean())
        elif statistic == 'median':
            rates_median.append(df[df[name] > 0][col].median())
        # 添加其他可能的统计量

    return rates_mean, rates_median

def test():
    intervals = ['1m', '5m', '15m', '1h', '1d']
    pairs = json_to_str()
    for interval in intervals:
        for pair in pairs:
            rates_mean, rates_median = calculate_rate(interval, pair, 'basic_Open_point')
            print('{} with interval {}: rates for mean is {}, rates for median is {}'.format(pair,interval,rates_mean,rates_median))

