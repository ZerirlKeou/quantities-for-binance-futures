import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# cnn模型
class CNNBitcoinPredictor(nn.Module):
    def __init__(self):
        super(CNNBitcoinPredictor, self).__init__()
        # 1D convolutional layers
        # Assuming each input data is 1-dimensional with 10 numbers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # Adjust in_channels to 1
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layers
        # The input features of the first fully connected layer need to be adjusted
        # based on the output of the last pooling layer, which depends on the length of your sequence
        self.fc1 = nn.Linear(32 * 4, 128)  # Adjust based on the output dimension
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Apply conv follow by ReLU, then pooling
        x = self.pool(F.relu(self.conv1(x)))  # Convolution layer 1
        x = self.pool(F.relu(self.conv2(x)))  # Convolution layer 2

        # Flatten the sequence data for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten all dimensions except batch

        # Fully connected layers with ReLU activation function
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# 定义模型
class BitcoinPredictor(nn.Module):
    def __init__(self):
        super(BitcoinPredictor, self).__init__()
        self.fc1 = nn.Linear(10, 64)  # 8 个输入特征
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512,1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc5 = nn.Linear(1024,512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc6 = nn.Linear(512, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.fc7 = nn.Linear(128,3)# 3 个分类（上涨、下跌、不变）

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = self.fc7(x)
        return x

# 创建模型实例
# model = BitcoinPredictor()
model = CNNBitcoinPredictor()

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
for epoch in range(50):  # 进行 1000 个训练周期
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

