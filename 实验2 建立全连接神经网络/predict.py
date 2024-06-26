import torch

def evaluate_model(model, test_loader, device, loss_fn):
    model.eval()  # 设置模型为评估模式
    test_loss, correct = 0, 0
    with torch.no_grad():  # 在评估阶段，不需要计算梯度
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)
    accuracy = 100 * correct
    return test_loss, accuracy

