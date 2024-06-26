import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score

# 读取数据
X_train = pd.read_csv('breast_cancer_Xtrain.csv', header=0).values
X_test = pd.read_csv('breast_cancer_Xtest.csv', header=0).values
y_train = pd.read_csv('breast_cancer_Ytrain.csv', header=0).values.flatten()  # 确保y是一维的
y_test = pd.read_csv('breast_cancer_Ytest.csv', header=0).values.flatten()  # 确保y是一维的

# 定义参数网格
param_grid = [
    {
        'C': [1, 10, 100, 1000],
        'kernel': ['linear']
    },
    {
        'C': [1, 10, 100, 1000],
        'kernel': ['poly'],
        'degree': [2, 3, 4],
        'coef0': [0, 1]  # 偏移量c
    },
    {
        'C': [1, 10, 100, 1000],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10]
    },
    {
        'C': [1, 10, 100, 1000],
        'kernel': ['sigmoid'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
        'coef0': [0, 1]  # 偏移量c
    }
]

# 进行网格搜索
grid_search = GridSearchCV(SVC(), param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1, return_train_score=True)
grid_search.fit(X_train, y_train)

# 得到每个参数组合的交叉验证准确率
results = pd.DataFrame(grid_search.cv_results_)
results = results[['params', 'mean_test_score', 'rank_test_score']]

# 输出最佳参数组合和最佳模型的分类准确率
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Accuracy:", best_score)

# 绘制图表函数
def plot_grid_search(cv_results, param1, param2, param1_name, param2_name, title):
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(param2), len(param1))

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(scores_mean, interpolation='nearest', cmap='viridis')
    plt.xlabel(param1_name)
    plt.ylabel(param2_name)
    plt.colorbar()
    plt.title(title)
    plt.xticks(np.arange(len(param1)), param1, rotation=45)
    plt.yticks(np.arange(len(param2)), param2)
    plt.show()

# 提取参数组合并绘制图表
for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    if kernel == 'linear':
        param1_name = 'C'
        param2_name = 'kernel'
        param1 = [1, 10, 100, 1000]
        param2 = [kernel]
        plot_grid_search(results[results['params'].apply(lambda x: x['kernel'] == kernel)], param1, param2, param1_name, param2_name, f'Grid Search Accuracy for {kernel} kernel')
    elif kernel == 'poly':
        for coef0 in [0, 1]:
            param1_name = 'C'
            param2_name = 'degree'
            param1 = [1, 10, 100, 1000]
            param2 = [2, 3, 4]
            filtered_results = results[results['params'].apply(lambda x: x['kernel'] == kernel and x['coef0'] == coef0)]
            if len(filtered_results) == len(param1) * len(param2):  # Ensure complete data
                plot_grid_search(filtered_results, param1, param2, param1_name, param2_name, f'Grid Search Accuracy for {kernel} kernel with coef0={coef0}')
    elif kernel == 'rbf':
        param1_name = 'C'
        param2_name = 'gamma'
        param1 = [1, 10, 100, 1000]
        param2 = ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10]
        filtered_results = results[results['params'].apply(lambda x: x['kernel'] == kernel)]
        if len(filtered_results) == len(param1) * len(param2):  # Ensure complete data
            plot_grid_search(filtered_results, param1, param2, param1_name, param2_name, f'Grid Search Accuracy for {kernel} kernel')
    elif kernel == 'sigmoid':
        for coef0 in [0, 1]:
            param1_name = 'C'
            param2_name = 'gamma'
            param1 = [1, 10, 100, 1000]
            param2 = ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10]
            filtered_results = results[results['params'].apply(lambda x: x['kernel'] == kernel and x['coef0'] == coef0)]
            if len(filtered_results) == len(param1) * len(param2):  # Ensure complete data
                plot_grid_search(filtered_results, param1, param2, param1_name, param2_name, f'Grid Search Accuracy for {kernel} kernel with coef0={coef0}')

