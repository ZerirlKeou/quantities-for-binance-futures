import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import classification_report
import joblib
import statsmodels.api as sm
import matplotlib.pyplot as plt


class deal_with_df:
    def __init__(self):
        pass

    # 准备数据
    def connect_data1(self, threshold=0.0008):
        df = pd.read_csv('data\\5m_data.csv')
        key_columns = ['Open time', 'Close', 'Return_1', 'Return_2']
        df = df.reindex(columns=key_columns)
        features = df.drop(
            ["Open time", "Return_1", "Return_2"], axis=1)
        print(features)
        targets = np.where(df['Return_1'] > threshold, 1, np.where(df['Return_1'] < -threshold, -1, 0))

        # 根据时间顺序划分训练集和测试集
        split_index = int(len(df) * 0.8)  # 根据前80%的索引位置进行划分
        X_train = features[:split_index]
        y_train = targets[:split_index]
        X_test = features[split_index:]
        y_test = targets[split_index:]

        return X_train, y_train, X_test, y_test


def train_lstm_classifier():
    dl = deal_with_df()
    X_train, y_train, X_test, y_test = dl.connect_data1()

    # 数据归一化
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 将数据重塑为LSTM输入的三维形状 (样本数量, 时间步长, 特征数量)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 模型训练
    model.fit(X_train, y_train, epochs=20, batch_size=32)

    probabilities = model.predict(X_test)
    predictions = np.where(probabilities > 0.5, 1, 0)

    joblib.dump(model, 'model/lstm/lstm_model.pkl')

    # Output classification report
    report = classification_report(y_test, predictions)

    return report


def train_arima_model():
    dl = deal_with_df()
    X_train, y_train, X_test, y_test = dl.connect_data1()

    # 将训练集转换为一维时间序列数据
    train_data = X_train.squeeze()

    # 训练ARIMA模型
    arima_model = sm.tsa.ARIMA(train_data, order=(1, 0, 1))
    arima_model = arima_model.fit()

    return arima_model


def plot_predictions(actual, predicted):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('ARIMA Model - Actual vs Predicted')
    plt.legend()
    plt.savefig('arima_predictions.png')


def train_and_plot_arima():
    arima_model = train_arima_model()

    # 获取测试集数据
    _, _, X_test, y_test = deal_with_df().connect_data1()
    test_data = X_test.squeeze()

    # 进行模型预测
    predictions = arima_model.predict(start=len(X_test), end=len(X_test) + len(y_test) - 1)

    # 绘制预测结果图
    plot_predictions(test_data, predictions)