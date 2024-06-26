# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import config
from FCNN import FCNNapi, FCNNhands
from predict import evaluate_model


args = config.args
torch.manual_seed(args.seed)

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

# 下载和加载训练集
train_dataset = datasets.FashionMNIST(
    root=args.dataset_path, train=True, transform=transform, download=True)
# 下载和加载测试集
test_dataset = datasets.FashionMNIST(
    root=args.dataset_path, train=False, transform=transform, download=True)

# 批训练循环api
def train_loop(dataloader, model, loss_fn, optimizer, device, num_epochs,test_loader):
    train_losses = []  # 用于保存每个 epoch 的训练损失
    test_losses = []   # 用于保存每个 epoch 的测试损失
    test_accuracy = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0  # 用于累计每个 epoch 的总损失
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        average_loss = total_loss / len(dataloader.dataset)
        train_losses.append(average_loss)
        print(f'api简洁实现全连接神经网络 Epoch [{epoch+1}/{num_epochs}] 训练完成')
        print(f'Training Loss: {average_loss:.4f}')

        # 在每个epoch结束后调用评估函数
        test_loss, accuracy = evaluate_model(model, test_loader, device, loss_fn)
        print(f"Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

# 批训练循环hands
def train_hands(train_loader, model, epochs,test_loader):
    train_losses = []  # 用于保存每个 epoch 的训练损失
    test_losses = []   # 用于保存每个 epoch 的测试损失
    test_accuracy = []
    for epoch in range(epochs):
        total_loss = 0  # 用于累计每个 epoch 的总损失
        for images, labels in train_loader:
            outputs = model.forward(images)
            model.backward(images, labels)
            total_loss += F.cross_entropy(outputs, labels, reduction='sum').item()
        average_loss = total_loss / len(train_loader.dataset)
        train_losses.append(average_loss)

        print(f'手动全连接神经网络 Epoch [{epoch+1}/{epochs}] 训练完成')
        print(f'Training Loss: {average_loss:.4f}')

        # 评估模型
        correct = 0
        total = 0
        test_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model.forward(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loss += F.cross_entropy(outputs, labels, reduction='sum').item()

        accuracy = 100 * correct / total
        average_loss = test_loss / total
        test_losses.append(average_loss)
        test_accuracy.append(accuracy)
        print(f'Accuracy: {accuracy:.2f}%')
        print(f'Test Loss: {average_loss:.4f}')

       

def main():
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # 加载模型
    modelapi = FCNNapi(args.input_size, args.hidden_size, args.output_size).to(device)
    # num_hiddens = [32,64,128,256,512,1024,2048]
    modelhands = FCNNhands(args.input_size, args.hidden_size, args.output_size)
    
    # 加载pytorch优化器
    # lr=[0.001,0.005,0.01,0.025,0.05,0.1,1.0]
    optimizerapi = optim.SGD(modelapi.parameters(), lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    
    # 训练循环
    train_loop(train_loader, modelapi, torch.nn.CrossEntropyLoss(), optimizerapi, device, args.epochs,test_loader)
    train_hands(train_loader, modelhands, args.epochs,test_loader)

    # 保存模型
    if not os.path.exists(os.path.dirname('./modelapi.pth')):
        os.makedirs(os.path.dirname('./modelapi.pth'))
    torch.save(modelapi.state_dict(), './modelapi.pth')
    print(f"Training done. Model saved to modelapi.pth")

    if not os.path.exists(os.path.dirname('./modelhands.pth')):
        os.makedirs(os.path.dirname('./modelhands.pth'))
    torch.save(modelhands.state_dict(), './modelhands.pth')
    print(f"Training done. Model saved to modelhands.pth")

if __name__ == '__main__':
    main()
