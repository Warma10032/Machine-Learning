import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 读取数据
X_train = pd.read_csv('breast_cancer_Xtrain.csv', header=0).values
X_test = pd.read_csv('breast_cancer_Xtest.csv', header=0).values
y_train = pd.read_csv('breast_cancer_Ytrain.csv', header=0).values.flatten()  # 确保y是一维的
y_test = pd.read_csv('breast_cancer_Ytest.csv', header=0).values.flatten()  # 确保y是一维的

# 创建和训练SVM模型
models = [
    ("线性SVM, C=1", SVC(C=1, kernel='linear')),
    ("线性SVM, C=1000", SVC(C=1000, kernel='linear')),
    ("非线性SVM, C=1, 多项式核, d=2", SVC(C=1, kernel='poly', degree=2)),
    ("非线性SVM, C=1000, 多项式核, d=2", SVC(C=1000, kernel='poly', degree=2))
]

# 评估模型
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} 的准确率：{accuracy:.4f}")
