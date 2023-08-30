from factor import basic_factors
from factor import derived_factors
from visible import draw_klines as dk
import pandas as pd
import time
from memory_profiler import profile

file_names = ['data\\1d_data.csv', 'data\\1h_data.csv', 'data\\15m_data.csv', 'data\\5m_data.csv', 'data\\1m_data.csv']


@profile()
def upgrade_data():
    # StartTime = time.time()
    # factor_calculator = basic_factors.FactorCalculator()
    # derived_calculator = derived_factors.AddBuyPoint()
    # for file_name in file_names:
    #     df = pd.read_csv(file_name)
    #     df = factor_calculator.calculate_factors(df)
    #     df = derived_calculator.williams_point(df)
    #     df = derived_calculator.macd_back(df)
    #     df = derived_calculator.williams_point1(df)
    #     df = derived_calculator.basic_buy_point(df)
    #     df.to_csv(file_name, index=False)
    # EndTime = time.time()
    # print(EndTime - StartTime)

    # chart = dk.StockChart5("NEW_test.csv", "prediction", "Return_1", draw_number=20000)
    # chart = dk.StockChart6("data\\5m_data.csv", "open-close", "Return_1", draw_number=20000)
    chart = dk.StockChart2("data\\5m_data.csv", "macd", ["basic_Open_point"], draw_number=-200)
    chart.show_chart()
