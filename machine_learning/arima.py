import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import joblib


class deal_with_df:
    def __init__(self):
        pass

    def prepare_data(self, threshold=0.0008, split_ratio=0.8):
        df = pd.read_csv('test.csv')
        features = df.drop(["Open time", "Return_1", "Return_2"], axis=1)[:-1]
        df['target'] = df['Close'].shift(-1)
        df = df[:-1]
        targets = df['target']  # 连续的目标变量

        # 根据时间顺序划分训练集和测试集
        split_index = int(len(df) * 0.35)  # 根据前80%的索引位置进行划分
        X_train = features[:split_index]
        y_train = targets[:split_index]
        X_test = features[split_index:]
        y_test = targets[split_index:]

        return X_train, y_train, X_test, y_test


def plot_predictions(actual, predicted):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Random Forest Model - Actual vs Predicted')
    plt.legend()
    plt.savefig('arima_predictions.png')


def train_random_forest_regression():
    dl = deal_with_df()
    X_train, y_train, X_test, y_test = dl.prepare_data()

    # 训练随机森林回归模型
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)

    # 在测试集上进行预测
    joblib.dump(rf_model, 'model/randomforest/rf_model.pkl')

    # 输出回归报告
    report = "Regression Report:\n"
    report += f"R2 Score: {rf_model.score(X_test, y_test)}\n"
    print(report)

    return rf_model


def train_and_plot_arima():
    model = train_random_forest_regression()
    dl = deal_with_df()
    _, _, X_test, y_test = dl.prepare_data()
    y_pred = model.predict(X_test)
    plot_predictions(y_test.reset_index(drop=True), y_pred)


# def train_and_plot_arima():
#     arima_model = train_arima_model()
#
#     # 获取测试集数据
#     _, X_test = deal_with_df().connect_data1()
#     test_data = X_test.squeeze()
#     print(test_data)
#
#     # 进行模型预测
#     predictions = arima_model.predict(len(_), len(_) + len(test_data))
#     print(predictions)
#
#     # 绘制预测结果图
#     plot_predictions(test_data, predictions)
