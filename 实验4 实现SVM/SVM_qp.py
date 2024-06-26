import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

# 定义硬间隔支持向量机类
class HardMarginSVM:
    def __init__(self):
        self.w = None # 权重向量
        self.b = None # 偏置项

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        y = y.flatten()

        # 定义二次规划问题
        K = np.dot(X, X.T)
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n_samples))
        G = matrix(-np.eye(n_samples))
        h = matrix(np.zeros(n_samples))
        A = matrix(y, (1, n_samples), 'd')
        b = matrix(0.0)

        # 解决二次规划问题
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(solution['x']).flatten()

        # 计算权重向量和偏置项
        self.w = np.sum((alphas * y)[:, None] * X, axis=0)
        support_vector_indices = np.where(alphas > 1e-5)[0]
        self.b = np.mean(y[support_vector_indices] - np.dot(X[support_vector_indices], self.w))

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

def Accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# 读取数据
X_train = pd.read_csv('breast_cancer_Xtrain.csv', header=0).values
X_test = pd.read_csv('breast_cancer_Xtest.csv', header=0).values

y_train = pd.read_csv('breast_cancer_Ytrain.csv', header=0).values.flatten()  # Ensure y is 1D
y_test = pd.read_csv('breast_cancer_Ytest.csv', header=0).values.flatten()  # Ensure y is 1D

# 创建并训练SVM模型
svm = HardMarginSVM()
svm.fit(X_train, y_train)

# 预测并计算准确率
y_pred = svm.predict(X_test)
accuracy = Accuracy(y_test, y_pred)
print("准确率：", accuracy)
