import torch
import torch.nn as nn
import config
import torch.nn.functional as F

# 参数
args = config.args
torch.manual_seed(args.seed)
num_inputs, num_outputs, num_hiddens = args.input_size, args.output_size, args.hidden_size

def relu(X):
    return torch.max(X, torch.tensor(0.0))

def accuracy(labels, preds):
    correct = sum(1 for true_label, pred_label in zip(labels, preds) if true_label == pred_label)  # 计算预测正确的样本数
    return correct / len(labels) # 计算准确率

class FCNNapi(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, initializer=None):
        super(FCNNapi, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_inputs, num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, num_outputs)
        
        if initializer is None:
            initializer = self.init_weights
        self.apply(initializer)  # 使用自定义的初始化方法

    # 初始化网络权重和偏置的函数
    def init_weights(self, m):
        if isinstance(m, nn.Linear):  # 检查是否为线性层
            nn.init.normal_(m.weight, mean=0.0, std=0.01)  # 使用正态分布初始化权重
            nn.init.constant_(m.bias, 0)  # 将偏置初始化为0

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        # x = F.sigmoid(x)
        # x = F.tanh(x)
        # x = F.softmax(x, dim=1)
        x = relu(x)
        x = self.fc2(x)
        return x


class FCNNhands:
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 0.1):
        self.W1 = torch.randn(input_size, hidden_size, requires_grad=True)
        self.b1 = torch.randn(1, hidden_size, requires_grad=True)
        self.W2 = torch.randn(hidden_size, output_size, requires_grad=True)
        self.b2 = torch.randn(1, output_size, requires_grad=True)
        self.learning_rate = learning_rate
    
    def forward(self, X):
        # 前向传播
        X = X.view(X.size(0), -1)
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = torch.matmul(self.a1, self.W2) + self.b2
        return self.z2

    def backward(self, X, y):
        # 后向传播
        loss = F.cross_entropy(self.z2, y)
        loss.backward()

        # 更新参数
        with torch.no_grad():
            self.W1 -= self.learning_rate * self.W1.grad
            self.b1 -= self.learning_rate * self.b1.grad
            self.W2 -= self.learning_rate * self.W2.grad
            self.b2 -= self.learning_rate * self.b2.grad
            
            # 梯度清零
            self.W1.grad.zero_()
            self.b1.grad.zero_()
            self.W2.grad.zero_()
            self.b2.grad.zero_()
            
    # 返回参数用于保存模型
    def state_dict(self):
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }

  