import sqlite3
import pandas as pd
import json
from getData import binance_future
import time


def json_to_str():
    """
    获取合约池信息
    Returns:合约列表
    """
    with open("future_pool.json", 'r') as load_f:
        pair_index = json.load(load_f)
    return pair_index


def write_to_database(pair, start_val=1681956800000, end_val=int(time.time() * 1000)):
    """
    创建1min级别合约数据库
    Args:
        pair: 交易对
        start_val: 开始时间
        end_val: 结束时间
    """
    name = 'data_base\\{}.db'.format(pair)
    conn = sqlite3.connect(name)

    try:
        sql_df = pd.read_sql_query("select * from {};".format(pair), conn)
        start_val = sql_df["Open time"].values[-1]+1
        print(u"Updating the {} database, starting at time {}".format(pair, start_val))
    except:
        print(u"Creating the {} database".format(pair))

    while start_val < end_val:
        df = binance_future(pair, start_val=start_val, end_val=end_val, interval='1m')
        if df.empty:
            break
        try:
            start_val = int(df['Open time'].values[-1]) + 1
        except:
            pass
        df.to_sql(name=pair, con=conn, index=False, if_exists='append')
    conn.close()


def create_database():
    pair_index = json_to_str()
    for pair in pair_index.values():
        write_to_database(pair)


create_database()