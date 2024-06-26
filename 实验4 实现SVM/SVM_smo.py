import numpy as np
import pandas as pd
import random


class SVM:
    def __init__(self, C=1.0, kernel='linear', gamma=0.1, tol=0.001, max_iter=40):
        self.C = C  # 正则化参数
        self.kernel = kernel  # 核函数类型
        self.gamma = gamma  # 高斯核函数的带宽参数
        self.tol = tol  # 迭代停止的阈值
        self.max_iter = max_iter  # 最大迭代次数
        self.b = 0  # 偏置项
        self.alpha = None  # 拉格朗日乘子
        self.support_vectors_ = None  # 支持向量
        self.dual_coef_ = None  # 决策函数的系数
        self.support_ = None  # 支持向量的索引
        
    # 随机选择 j
    def _select_j(self, i, m):
        j = i
        while j == i:
            j = random.randint(0, m - 1)
        return j
    # 限制 alpha 的范围
    def _clip_alpha(self, alpha, H, L):
        return max(L, min(alpha, H))
    # 核函数计算
    def _kernel(self, X1, X2):
        if self.kernel == 'rbf':
            if X2 is None:
                X2 = X1
            K = np.zeros((X1.shape[0], X2.shape[0]))
            for i in range(X1.shape[0]):
                for j in range(X2.shape[0]):
                    K[i, j] = np.exp(-self.gamma * np.linalg.norm(X1[i] - X2[j]) ** 2)
            return K
        
    # 使用SMO算法来拟合SVM模型，通过迭代更新拉格朗日乘子来优化模型参数。
    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.alpha = np.zeros(self.n_samples) # 初始化拉格朗日乘子为零向量
        iter_num = 0 # 初始化迭代次数
        K = self._kernel(X, None) # 计算核矩阵K
        # 开始迭代训练模型
        while iter_num < self.max_iter:
            alpha_changed = 0 # 用于记录在当前迭代中是否有拉格朗日乘子更新
            for i in range(self.n_samples):
                # 计算第i个样本的预测误差
                E_i = self._E(i, K, y)
                # 检查是否满足KKT条件
                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or (y[i] * E_i > self.tol and self.alpha[i] > 0):
                     # 选择第二个乘子j
                    j = self._select_j(i, self.n_samples)
                    # 计算第j个样本的预测误差
                    E_j = self._E(j, K, y)
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    L, H = self._compute_L_H(alpha_i_old, alpha_j_old, y[i], y[j])
                     # 如果L和H相等，则跳过本次迭代
                    if L == H:
                        continue
                    # 计算 eta，如果 eta 大于等于零，则跳过本次迭代            
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    # 更新第二个乘子 alpha_j        
                    self.alpha[j] -= y[j] * (E_i - E_j) / eta
                    self.alpha[j] = self._clip_alpha(self.alpha[j], H, L)
                    # 如果 alpha_j 的变化量小于给定阈值，则跳过本次迭代
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    # 更新参数              
                    self.alpha[i] += y[j] * y[i] * (alpha_j_old - self.alpha[j])
                    b1, b2 = self._compute_b(E_i, E_j, K, i, j, alpha_i_old, alpha_j_old, y)
                    self.b = self._update_b(self.alpha[i], self.alpha[j], b1, b2)
                    # 记录有乘子更新
                    alpha_changed += 1

            iter_num = iter_num + 1 if alpha_changed == 0 else 0
        # 计算支持向量和决策函数系数
        self.support_ = np.where(self.alpha > 0)[0]
        self.support_vectors_ = X[self.support_]
        self.dual_coef_ = self.alpha[self.support_] * y[self.support_]
    # 计算样本点的误差
    def _E(self, i, K, y):
        return np.dot(self.alpha * y, K[:, i]) + self.b - y[i]
    # 计算 L 和 H
    def _compute_L_H(self, alpha_i_old, alpha_j_old, y_i, y_j):
        if y_i != y_j:
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)
        else:
            L = max(0, alpha_j_old + alpha_i_old - self.C)
            H = min(self.C, alpha_j_old + alpha_i_old)
        return L, H
    # 计算 b1 和 b2
    def _compute_b(self, E_i, E_j, K, i, j, alpha_i_old, alpha_j_old, y):
        b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
        b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]
        return b1, b2
    # 更新 b
    def _update_b(self, alpha_i, alpha_j, b1, b2):
        if 0 < alpha_i < self.C:
            return b1
        elif 0 < alpha_j < self.C:
            return b2
        else:
            return (b1 + b2) / 2
    # 进行分类
    def predict(self, X):
        K = self._kernel(X, self.support_vectors_)
        return np.sign(np.dot(K, self.dual_coef_) + self.b)


def Accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# 读取数据
X_train = pd.read_csv('breast_cancer_Xtrain.csv', header=0).values
X_test = pd.read_csv('breast_cancer_Xtest.csv', header=0).values

y_train = pd.read_csv('breast_cancer_Ytrain.csv', header=0).values.flatten()  # Ensure y is 1D
y_test = pd.read_csv('breast_cancer_Ytest.csv', header=0).values.flatten()  # Ensure y is 1D

# 创建并训练SVM模型
classifier = SVM(C=2, kernel='rbf', gamma=0.1)
classifier.fit(X_train, y_train)

# 预测并计算准确率
y_pred = classifier.predict(X_test)
accuracy = Accuracy(y_test, y_pred)
print("准确率：", accuracy)

