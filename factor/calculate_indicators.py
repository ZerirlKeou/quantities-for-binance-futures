from factor import basic_factors
from factor import derived_factors
from visible import drawMainFigure
from visible import drawScatter as ds
import pandas as pd
import time
from memory_profiler import profile
import json
import sqlite3

file_names = ['data\\1d_data.csv', 'data\\1h_data.csv', 'data\\15m_data.csv', 'data\\5m_data.csv', 'data\\1m_data.csv']
app = drawMainFigure.MplVisualIf()


def json_to_str():
    """
    获取合约池信息
    Returns:合约列表
    """
    with open("data\\future_pool.json", 'r') as load_f:
        pair_index = json.load(load_f)
    return pair_index


def factor_calculate(df):
    bf = basic_factors.WriteFactorData()
    derived_calculator = derived_factors.AddBuyPoint()

    df = bf.calculate_factors(df)
    df = derived_calculator.williams_point(df)
    df = derived_calculator.macd_back(df)
    df = derived_calculator.williams_point1(df)
    df = derived_calculator.basic_buy_point(df)

    return df


def get_df(pair):
    name = 'data\\data_base\\{}.db'.format(pair)
    conn = sqlite3.connect(name)
    try:
        df = pd.read_sql_query("select * from {};".format(pair), conn, dtype='float')
    except:
        print(u'{} database has not create'.format(pair))
        return None
    print(df)
    # layout_dict = {'df': df,
    #                'draw_kind': ['kline'],
    #                'title': pair}
    # app.fig_output(**layout_dict)
    # df = factor_calculate(df)
    # df.to_sql(name=pair, con=conn, index=False, if_exists='replaced')


@profile()
def insert_data():
    pair_index = json_to_str()
    for pair in pair_index.values():
        get_df(pair)
    # StartTime = time.time()
    # bf = basic_factors.WriteFactorData()
    # derived_calculator = derived_factors.AddBuyPoint()
    # for file_name in file_names:
    #     df = pd.read_csv(file_name)
    #     df = bf.calculate_factors(df)
    #     df = derived_calculator.williams_point(df)
    #     df = derived_calculator.macd_back(df)
    #     df = derived_calculator.williams_point1(df)
    #     df = derived_calculator.basic_buy_point(df)
    #     df.to_csv(file_name, index=False)
    # EndTime = time.time()
    # print(EndTime - StartTime)

    # 散点图绘制
    # chart = ds.StockChart6("data\\1d_data.csv", "VAR8", "Return_1", draw_number=200)
    # chart.show_chart()

    # 主图绘制
    # layout_dict = {'path': "data\\1d_data.csv",
    #                'draw_kind': ['kline', 'volume', 'macd'],
    #                'title': u"BNBUSTD"}
    # app.fig_output(**layout_dict)
