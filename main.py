# 策略回测，至少每25小时回测一次
import os
import average.average_cal
from binance.um_futures import UMFutures
import pandas as pd
from factor import calculate_indicators
from GeneticPrograming import genetic_programing
from TradeBackSystem import trade_back_system as tb
from machine_learning import data_preparing
from machine_learning import random_forest
from machine_learning import SVM_nodel
from machine_learning import lstm_model
from machine_learning import arima
from machine_learning import model_predict_write
from average import average_cal
import time
import csv
import joblib

client = UMFutures()


class MarketDataChecker:
    def __init__(self, client):
        self.client = client

    def completion_history_data(self):
        server_time = int(self.client.time()['serverTime'])
        self.check_interval("1d", 86400000, "data/1d_data.csv", server_time)
        self.check_interval("1h", 3600000, "data/1h_data.csv", server_time)
        self.check_interval("15m", 900000, "data/15m_data.csv", server_time)
        self.check_interval("5m", 300000, "data/5m_data.csv", server_time)
        self.check_interval("1m", 60000, "data/1m_data.csv", server_time)

    def write_data_to_csv(self, starttime, interval, times, file_handle):
        completion = int(self.client.time()['serverTime']) // times - int(starttime) // times
        if completion != 0:
            if completion > 1500:
                data = self.client.continuous_klines(pair="BNBUSDT", contractType='PERPETUAL', interval=interval,
                                                     limit=1500)
            else:
                data = self.client.continuous_klines(pair="BNBUSDT", contractType='PERPETUAL', interval=interval,
                                                     limit=completion)
            df = pd.DataFrame(data)
            df.to_csv(file_handle, mode='a', index=False, header=False)
        return completion

    def read_last_time(self, interval):
        path = os.path.join("data", f"{interval}_data.csv")
        df = pd.read_csv(path)
        time_temp = int(df["Open time"].iloc[-1])
        return time_temp

    def check_interval(self, interval, times, file_path, server_time):
        last_time = self.read_last_time(interval)
        if last_time + times < server_time:
            with open(file_path, "a") as file_handle:
                completion = self.write_data_to_csv(last_time, interval, times, file_handle)
            print(f"{completion} data has been repaired")


if __name__ == '__main__':
    # 机器学习模块
    # 清洗连接数据
    # StartTime = time.time()
    # test = data_preparing.Connection("15m", "5m")
    # test = data_preparing.SingleInterval("5m", 3)
    # test.new_data()
    # EndTime = time.time()
    # print(EndTime - StartTime)

    # 训练随机森林模型
    # 分类器
    # classification_report = random_forest.train_random_forest_classifier()
    # 回归器
    # classification_report = random_forest.train_random_forest_regressor()
    # 孤立森林
    # random_forest.train_isolation_random_forest_regressor()
    # 贝叶斯搜索
    # classification_report = random_forest.search_best_paramsbyes(rf_type_classifier=True)
    # classification_report = random_forest.search_best_parameters_regressor()

    # 训练支持向量机
    # 分类任务
    # classification_report = SVM_nodel.train_support_vector_machine_classifier()
    # 回归任务
    # classification_report = SVM_nodel.train_support_vector_machine_regression()
    # 网格搜索
    # classification_report = SVM_nodel.train_svm_with_grid_search()

    # 训练lstm模型
    # classification_report = lstm_model.train_and_plot_arima()

    # classification_report = arima.train_and_plot_arima()
    # random_forest.draw_heat_picture()

    # print(classification_report)

    # 写入模型预测值
    # mp = model_predict_write.PredictionWrite(4, 'randomforest', "random_forest")
    # mp.write_csv()

    # aver = average_cal.MultiAverageLine("5m")

    # 遗传规划
    # gp = genetic_programing.GeneticProgramming()
    # gp.gep_train()

    # 检查行情
    # checker = MarketDataChecker(client)
    # checker.completion_history_data()
    # 计算指标
    # calculate_indicators.upgrade_data()
    # 回测系统
    # backtest_system = tb.TradeBackSystemV5(money=41.24, premium=0.02, lever=8)
    backtest_system = tb.GridTradeMulti(money=41.24, premium=0.02, lever=8)
    backtest_system.run_backtest()
    # 测试账号客户端连通性
    # print(client.exchange_info())
