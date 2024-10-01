import os
import sqlite3
import pandas as pd
from visible import drawMainFigure


def draw_picture(df, index1, index2, windows=20):
    """
    通过输入dataframe格式的数据类型，来生成特定位置，特定窗口的比较图案
    Args:
        df: dataframe格式的数据
        index1: 对比pair中的第一个的index
        index2: 对比pair中的第二个index
        windows: 观测时间窗口

    Returns: 返回的为int类型，表示哪一个好，0代表第一个好，1代表第二个好

    """
    fig1 = drawMainFigure.MplTypesDraw()


def write_csv():
    pass


def test():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_folder = "data\\data_base\\15m\\BTCUSDT.db"
    data_folder_path = os.path.join(parent_dir, data_folder)
    conn = sqlite3.connect(data_folder_path)

    df = pd.read_sql_query("select * from 'BTCUSDT';", conn, dtype='float', index_col='Open time')
    draw_picture(df,20,40,windows=20)
