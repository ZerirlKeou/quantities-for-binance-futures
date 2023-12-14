import pandas as pd
import sqlite3
import os


def ffill(df, name):
    df[name] = df[name].replace(0, pd.NA)
    df[name] = df[name].ffill()
    return df


class HandlingData(object):
    def __init__(self, df, name):
        ffill(df, name)
        df['strategy'] = df[name] * df['Return_1']
        print(df[['Return_1', 'strategy']].sum())


def test():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_folder = "data\\data_base\\ETHUSDT.db"
    data_folder_path = os.path.join(parent_dir, data_folder)
    conn = sqlite3.connect(data_folder_path)

    df = pd.read_sql_query("select * from 'ETHUSDT';", conn, dtype='float',index_col='Open time')
    HandlingData(df,'macd_back')

