import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 定义模型
class BitcoinPredictor(nn.Module):
    def __init__(self):
        super(BitcoinPredictor, self).__init__()
        self.fc1 = nn.Linear(8, 128)  # 8 个输入特征
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 3)   # 3 个分类（上涨、下跌、不变）

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# 创建模型实例
model = BitcoinPredictor()

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


with open('normalized_data.pkl', 'rb') as file:
    df_scaled = pickle.load(file)
# 准备数据
X = torch.tensor(df_scaled.drop(['Return_1', 'Label'], axis=1).values).float()
y = torch.tensor(df_scaled['Label'].values).long()
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

train_data = TensorDataset(train_X, train_y)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

losses = []

# 训练模型
for epoch in range(500):  # 进行 1000 个训练周期
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 记录损失
        losses.append(loss.item())

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')


# 使用测试集评估模型
test_X, test_y = test_X.to(device), test_y.to(device)
with torch.no_grad():
    outputs = model(test_X)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == test_y).sum().item() / test_y.size(0)


print(f'Accuracy: {accuracy:.2f}')
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_plot.png')  # 保存图像
plt.show()

