import sys

from factor import basic_factors
from factor import derived_factors
from factor import time_series_factor
from visible import drawMainFigure
from visible import drawScatter as ds
import pandas as pd
import time
from memory_profiler import profile
import json
import sqlite3
import numpy as np
import os
import site

app = drawMainFigure.MplVisualIf()

def timeit_test(func):
    """
    Args: 用于计算指标计算运行时间
        func: 测试函数
    """
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = (time.perf_counter() - start)
        print('Time used: {}'.format(elapsed))
    return wrapper

def json_to_str():
    """
    获取合约池信息
    Returns:合约列表
    """
    with open("data\\future_pool.json", 'r') as load_f:
        pair_index = json.load(load_f)
    return pair_index


def factor_calculate(df):
    """
    Args:计算所有指标，包括基本指标以及衍生指标
    """
    bf = basic_factors.WriteFactorData()
    dbf = derived_factors.DerivedFactorData()
    df = bf.calculate_factors(df)
    df = dbf.calculate_factors(df)
    return df

def factor_time_series(df):
    tsf = time_series_factor.TimeSeriesFactorData()
    df = tsf.calculate_factors(df)
    return df


def get_df(interval,pair):
    name = 'data\\data_base\\{}\\{}.db'.format(interval,pair)
    conn = sqlite3.connect(name)
    try:
        df = pd.read_sql_query("select * from {};".format(pair), conn, dtype='float')
    except:
        print(u'{} database has not create'.format(pair))
        return None

    df = df.drop_duplicates(subset='Open time')
    print('开始计算', pair, '基础指标及其衍生指标')
    df = factor_calculate(df)

    print('开始计算', pair, '时间序列指标')
    df = factor_time_series(df)

    df.to_sql(name=pair, con=conn, index=False, if_exists='replace')
    conn.close()


@profile()
def insert_data():
    # intervals = ['1m','5m','15m','1h','1d']
    # pair_index = json_to_str()
    # for interval in intervals:
    #     for pair in pair_index.values():
    #         get_df(interval, pair)

    # get_df('1m','BTCUSDT')



    name = 'data\\data_base\\1m\\BTCUSDT.db'
    conn = sqlite3.connect(name)
    # df = pd.read_sql_query("select * from 'BTCUSDT';", conn, dtype='float',index_col='Open time')
    df = pd.read_sql_query("SELECT * FROM 'BTCUSDT' ORDER BY 'Open time' DESC;", conn, dtype='float',
                           index_col='Open time')

    df['Open time'] = (df.index.astype(np.int64) // 10 ** 3).astype(int)

    # 散点图绘制
    # chart = ds.show_feature_img(df, "low-open", "Return_1", draw_number=200)
    # chart.show_chart()
    # 主图绘制
    layout_dict = {'df': df,
                   'draw_kind': ['kline','macd','macd_back', 'macd_back_time_series'],
                   'title': u"BTCUSDT"}
    app.fig_output(**layout_dict)


def test_calculate_indicator():
    # TODO 在爬取行情信息后使用本测试函数，减少从main函数中进行索引所浪费的时间，此方式运行较为简单
    # 更改工作目录到父目录
    os.chdir(os.pardir)
    insert_data()
