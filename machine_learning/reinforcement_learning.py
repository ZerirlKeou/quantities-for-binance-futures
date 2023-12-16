import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
# 获取当前脚本所在的目录
current_dir = os.path.dirname(__file__)

# 构建CSV文件的完整路径
csv_file_path = os.path.join(current_dir, '..', 'data', '5m_data.csv')

# 读取数据
data = pd.read_csv(csv_file_path)

# 数据预处理
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

# 定义状态空间和动作空间
state_size = data_scaled.shape[1]
action_size = 3  # 买入、卖出、持有

# 建立神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(state_size,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(action_size, activation='softmax')
])

# 定义奖励函数
def reward_function(action, next_price, current_price):
    if action == 0:  # 买入
        return next_price - current_price
    elif action == 1:  # 卖出
        return current_price - next_price
    else:  # 持有
        return 0

# 定义强化学习参数
learning_rate = 0.001
discount_factor = 0.95
optimizer = tf.keras.optimizers.Adam(learning_rate)
huber_loss = tf.keras.losses.Huber()

# 定义训练参数
num_episodes = 1000
num_steps = len(data) - 1  # 一个episode中的步数
# 编译模型
model.compile(optimizer=optimizer, loss=huber_loss)

# 定义初始状态
initial_state = data_scaled[0]
# 训练神经网络
for episode in range(num_episodes):
    state = initial_state
    episode_reward = 0

    for t in range(num_steps):
        # 使用神经网络选择动作
        action_probs = model.predict(state.reshape(1, -1))[0]
        action = np.random.choice(action_size, p=action_probs)

        # 执行动作并观察下一个状态和奖励
        next_state = data_scaled[t + 1]
        reward = reward_function(action, next_state[-1], state[-1])

        # 计算预期未来奖励
        future_rewards = model.predict(next_state.reshape(1, -1))[0]
        expected_future_reward = np.max(future_rewards)

        # 计算目标奖励
        target = reward + discount_factor * expected_future_reward

        # 计算损失并进行反向传播
        with tf.GradientTape() as tape:
            current_rewards = model(state.reshape(1, -1))
            loss = huber_loss(target, current_rewards[0][action])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        episode_reward += reward
        state = next_state

    print(f"Episode {episode + 1}, Total Reward: {episode_reward}")
