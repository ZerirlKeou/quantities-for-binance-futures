import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

# 读取数据
# 获取当前脚本所在的目录
current_dir = os.path.dirname(__file__)

# 构建CSV文件的完整路径
csv_file_path = os.path.join(current_dir, '..', 'data', '5m_data.csv')

# 读取数据
data = pd.read_csv(csv_file_path)

# 选择需要的列
selected_columns = ['Open', 'High', 'Low', 'Close']
data = data[selected_columns]

# 归一化数据
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# 创建时间窗口
sequence_length = 10
sequences = []
labels = []

for i in range(len(data_normalized) - sequence_length):
    sequence = data_normalized[i:i + sequence_length]
    label = 0
    # 判断下一天涨跌幅是否满足条件
    if data['Close'][i + sequence_length] > data['Close'][i + sequence_length - 1] * 1.0010:
        label = 1
    elif data['Close'][i + sequence_length] < data['Close'][i + sequence_length - 1] * 0.9990:
        label = -1
    sequences.append(sequence)
    labels.append(label)

# 转换为NumPy数组
X = np.array(sequences)
y = np.array(labels)
# 划分训练集和测试集
# 定义训练集和测试集的分割比例
split_ratio = 0.8

# 计算分割点
split_point = int(len(X) * split_ratio)

# 划分训练集和测试集
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

# 构建RNN模型
model = Sequential()
model.add(SimpleRNN(units=500, activation='relu', input_shape=(sequence_length, len(selected_columns))))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# 在测试集上评估模型
accuracy = model.evaluate(X_test, y_test)[1]
print('Test Accuracy:', accuracy)
# 在测试集上进行预测
predictions = model.predict(X_test)

# 反归一化预测值和真实值
predictions_denormalized = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_denormalized = scaler.inverse_transform(y_test.reshape(-1, 1))

# 绘制两个子图，一个是收盘价，一个是预测值
plt.figure(figsize=(12, 8))

# 子图1：真实值
plt.subplot(2, 1, 1)
plt.plot(y_test_denormalized, label='True Values', color='blue')
plt.title('True Values - Close Price')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()

# 子图2：预测值
plt.subplot(2, 1, 2)
plt.plot(predictions_denormalized, label='Predictions', color='red')
plt.title('Predictions - Close Price')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()

plt.tight_layout()
plt.show()