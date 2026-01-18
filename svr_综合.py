import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from pyswarm import pso
import optuna
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import ParameterGrid
from scipy.optimize import dual_annealing
from sklearn.preprocessing import StandardScaler

# 固定随机种子
np.random.seed(42)
random.seed(42)

# 读取数据
input_file = "数据_半疲劳.xlsx"  # 替换为你的文件名
data = pd.read_excel(input_file, engine='openpyxl')

# 数据处理
y = np.log10(data.iloc[:, -1])
X = data.iloc[:, 1:-1]  # 假设第一列是样本编号
print(X)
print(y)

X = (X - X.mean()) / X.std()  # 标准化

# 转换为 NumPy 数组
X = X.values
y = y.values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# 贝叶斯优化
trials_data_bayesian = []

def bayesian_objective(trial):
    C = trial.suggest_float('C', 0.01, 100)
    epsilon = trial.suggest_float('epsilon', 0.01, 0.1)
    gamma = trial.suggest_float('gamma', 0.0001, 1)
    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])

    svr = SVR(
        C=C,
        epsilon=epsilon,
        gamma=gamma,
        kernel=kernel
    )

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_mse_scores = []

    for train_idx, val_idx in kfold.split(X_train):
        # 转换为 numpy 数组
        X_train_np = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

        # 根据索引进行切分
        X_train_fold, X_val_fold = X_train_np[train_idx], X_train_np[val_idx]
        y_train_fold, y_val_fold = y_train_np[train_idx], y_train_np[val_idx]

        # 训练模型
        svr.fit(X_train_fold, y_train_fold)
        y_val_pred = svr.predict(X_val_fold)
        mse = mean_squared_error(y_val_fold, y_val_pred)
        cv_mse_scores.append(mse)

    avg_mse = np.mean(cv_mse_scores)
    trials_data_bayesian.append(avg_mse)
    return avg_mse

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(bayesian_objective, n_trials=200)
best_params_bayesian = study.best_params

svr_bayesian = SVR(
    **best_params_bayesian
)
svr_bayesian.fit(X_train, y_train)
y_pred_bayesian = svr_bayesian.predict(X_test)
bayesian_mse = mean_squared_error(y_test, y_pred_bayesian)
bayesian_r2 = r2_score(y_test, y_pred_bayesian)

pso_mse_list = []  # 用于存储每次迭代所有粒子中最小的 MSE 值

def pso_objective(parameters):
    C, epsilon, gamma, kernel_idx = parameters
    C = float(C)
    epsilon = float(epsilon)
    gamma = float(gamma)
    kernel = ['linear', 'poly', 'rbf'][int(kernel_idx)]

    svr = SVR(
        C=C,
        epsilon=epsilon,
        gamma=gamma,
        kernel=kernel
    )

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    mse_list = []  # 存储每个折叠的 MSE

    for train_idx, val_idx in kfold.split(X_train):
        # 转换为 numpy 数组
        X_train_np = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

        # 根据索引进行切分
        X_train_fold, X_val_fold = X_train_np[train_idx], X_train_np[val_idx]
        y_train_fold, y_val_fold = y_train_np[train_idx], y_train_np[val_idx]

        svr.fit(X_train_fold, y_train_fold)
        y_val_pred = svr.predict(X_val_fold)
        mse = mean_squared_error(y_val_fold, y_val_pred)
        mse_list.append(mse)

    return np.mean(mse_list)

# 使用 PSO 求解
lb = [0.01, 0.01, 0.0001, 0]  # 参数的下界
ub = [100, 0.1, 1, 2]  # 参数的上界，其中 kernel 使用 0, 1, 2 代表 ['linear', 'poly', 'rbf']

# 修改 collect_pso_mse 以记录每次迭代所有粒子的最小 MSE
def collect_pso_mse(parameters):
    mse = pso_objective(parameters)  # 计算当前粒子的 MSE
    return mse

# 自定义 PSO
def custom_pso(objective, lb, ub, swarmsize=10, maxiter=200):
    global pso_mse_list

    best_params = None
    best_mse = float("inf")

    for iter_num in range(maxiter):
        iteration_mse_list = []

        def wrapper(parameters):
            mse = objective(parameters)
            iteration_mse_list.append(mse)
            return mse

        best_params_iter, _ = pso(wrapper, lb, ub, swarmsize=swarmsize, maxiter=1)

        min_mse = min(iteration_mse_list)
        if min_mse < best_mse:
            best_mse = min_mse
            best_params = best_params_iter

        pso_mse_list.append(min_mse)

    return best_params, best_mse

# 调用自定义 PSO
best_params_pso, best_mse_pso = custom_pso(collect_pso_mse, lb, ub, swarmsize=10, maxiter=200)

svr_pso = SVR(
    C=best_params_pso[0],
    epsilon=best_params_pso[1],
    gamma=best_params_pso[2],
    kernel=['linear', 'poly', 'rbf'][int(best_params_pso[3])]
)
svr_pso.fit(X_train, y_train)
y_pred_pso = svr_pso.predict(X_test)
pso_mse = mean_squared_error(y_test, y_pred_pso)
pso_r2 = r2_score(y_test, y_pred_pso)

# 网格搜索
grid_search_data = []  # 用于存储每种参数组合的 MSE
# 定义网格搜索函数
def grid_search():
    param_grid = {
        'C': np.linspace(0.01, 100, 10),
        'epsilon': np.linspace(0.01, 0.1, 10),
        'gamma': np.linspace(0.0001, 1, 10),
        'kernel': ['linear', 'poly', 'rbf']
    }

    best_params = None
    best_mse = float("inf")

    for C in param_grid['C']:
        for epsilon in param_grid['epsilon']:
            for gamma in param_grid['gamma']:
                for kernel in param_grid['kernel']:
                    svr = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel)

                    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
                    mse_list = []

                    for train_idx, val_idx in kfold.split(X_train):
                        # 转换为 numpy 数组
                        X_train_np = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
                        y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

                        # 根据索引进行切分
                        X_train_fold, X_val_fold = X_train_np[train_idx], X_train_np[val_idx]
                        y_train_fold, y_val_fold = y_train_np[train_idx], y_train_np[val_idx]


                        svr.fit(X_train_fold, y_train_fold)
                        y_val_pred = svr.predict(X_val_fold)
                        mse = mean_squared_error(y_val_fold, y_val_pred)
                        mse_list.append(mse)

                    avg_mse = np.mean(mse_list)
                    grid_search_data.append(avg_mse)

                    if avg_mse < best_mse:
                        best_mse = avg_mse
                        best_params = (C, epsilon, gamma, kernel)

    return best_params, best_mse

# 调用网格搜索函数
best_params_grid, best_mse_grid = grid_search()

# 使用最优参数重新训练模型
svr_grid = SVR(
    C=best_params_grid[0],
    epsilon=best_params_grid[1],
    gamma=best_params_grid[2],
    kernel=best_params_grid[3]
)
svr_grid.fit(X_train, y_train)
y_pred_grid = svr_grid.predict(X_test)
grid_mse = mean_squared_error(y_test, y_pred_grid)
grid_r2 = r2_score(y_test, y_pred_grid)


# 模拟退火
sa_mse_list = []  # 用于存储每次迭代的 MSE

def simulated_annealing_svr():
    bounds = [(0.01, 100), (0.01, 0.1), (0.0001, 1), (0, 2)]

    def sa_objective(params):
        C, epsilon, gamma, kernel_idx = params
        C = float(C)
        epsilon = float(epsilon)
        gamma = float(gamma)
        kernel = ['linear', 'poly', 'rbf'][int(kernel_idx)]

        svr = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel)

        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        mse_list = []

        for train_idx, val_idx in kfold.split(X_train):
            # 转换为 numpy 数组
            X_train_np = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
            y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

            # 根据索引进行切分
            X_train_fold, X_val_fold = X_train_np[train_idx], X_train_np[val_idx]
            y_train_fold, y_val_fold = y_train_np[train_idx], y_train_np[val_idx]

            svr.fit(X_train_fold, y_train_fold)
            y_val_pred = svr.predict(X_val_fold)
            mse = mean_squared_error(y_val_fold, y_val_pred)
            mse_list.append(mse)

        avg_mse = np.mean(mse_list)
        sa_mse_list.append(avg_mse)
        return avg_mse

    result = dual_annealing(sa_objective, bounds, maxiter=200)
    return result.x, result.fun

# 调用模拟退火函数
best_params_sa, best_mse_sa = simulated_annealing_svr()

svr_sa = SVR(
    C=best_params_sa[0],
    epsilon=best_params_sa[1],
    gamma=best_params_sa[2],
    kernel=['linear', 'poly', 'rbf'][int(best_params_sa[3])]
)
svr_sa.fit(X_train, y_train)
y_pred_sa = svr_sa.predict(X_test)
sa_mse = mean_squared_error(y_test, y_pred_sa)
sa_r2 = r2_score(y_test, y_pred_sa)
#
# 默认参数SVR模型
svr_default = SVR()
svr_default.fit(X_train, y_train)
y_pred_default = svr_default.predict(X_test)

default_mse = mean_squared_error(y_test, y_pred_default)
default_r2 = r2_score(y_test, y_pred_default)

# 提取默认参数
default_params = svr_default.get_params()

# 计算分散带内比例
def calc_band_proportion(y_test, y_pred):
    deviation = np.abs(y_pred - y_test)
    factor_1_5 = np.log10(1.5)
    factor_2 = np.log10(2)

    within_1_5_band = np.sum(deviation <= factor_1_5) / len(y_test)
    within_2_band = np.sum((deviation > factor_1_5) & (deviation <= factor_2)) / len(y_test)
    outside_2_band = np.sum(deviation > factor_2) / len(y_test)

    return within_1_5_band, within_2_band, outside_2_band

bayesian_band = calc_band_proportion(y_test, y_pred_bayesian)
pso_band = calc_band_proportion(y_test, y_pred_pso)
grid_band = calc_band_proportion(y_test, y_pred_grid)
sa_band = calc_band_proportion(y_test, y_pred_sa)
default_band = calc_band_proportion(y_test, y_pred_default)

# 神经网络、前馈神经网络、力学性能、形态学参数
# 保存结果到 Excel
output_file = "D:\\python_project\\fatigue\\结果\\final\\合并\\半疲劳\\1\\LCF_svr.xlsx"
results = {
    "Metric": ["MSE", "R²", "Within 1.5 Band", "Within 2 Band", "Outside 2 Band"],
    "Default RF": [default_mse, default_r2, *default_band],
    "Simulated Annealing": [sa_mse, sa_r2, *sa_band],
    "Bayesian Optimization": [bayesian_mse, bayesian_r2, *bayesian_band],
    "PSO": [pso_mse, pso_r2, *pso_band],
    "Grid Search": [grid_mse, grid_r2, *grid_band]
}
params = {
    "Parameter": ["C", "epsilon", "gamma", "kernel"],
    "Default Parameters": [svr_default.C, svr_default.epsilon, svr_default.gamma, svr_default.kernel],
    "Simulated Annealing": [
        best_params_sa[0],
        best_params_sa[1],
        best_params_sa[2],
        ['linear', 'poly', 'rbf'][int(best_params_sa[3])]
    ],
    "Bayesian Optimization": list(best_params_bayesian.values()),  # 确保这个是字典格式
    "PSO": [
        best_params_pso[0],
        best_params_pso[1],
        best_params_pso[2],
        ['linear', 'poly', 'rbf'][int(best_params_pso[3])]
    ],
    "Grid Search": [best_params_grid[0], best_params_grid[1], best_params_grid[2], best_params_grid[3]]  # 直接从元组中提取
}

results_df = pd.DataFrame(results)
params_df = pd.DataFrame(params)

with pd.ExcelWriter(output_file) as writer:
    results_df.to_excel(writer, index=False, sheet_name="Metrics")
    params_df.to_excel(writer, index=False, sheet_name="Parameters")

# 绘制预测散点图
plt.figure(figsize=(8, 6))

# 添加抖动
jitter = 0.03
y_test_jittered = y_test + np.random.uniform(-jitter, jitter, len(y_test))

# 使用不同的点形状和颜色来表示五种方法
plt.scatter(y_test_jittered, y_pred_default, alpha=0.5, c='red', marker='^', label='Default Model', edgecolors='black', linewidths=0.5, s=30)
plt.scatter(y_test_jittered, y_pred_bayesian, alpha=0.5, c='blue', marker='o', label='TPE Optimization', edgecolors='black', linewidths=0.5, s=30)
plt.scatter(y_test_jittered, y_pred_pso, alpha=0.5, c='green', marker='s', label='PSO Optimization', edgecolors='black', linewidths=0.5, s=30)
plt.scatter(y_test_jittered, y_pred_grid, alpha=0.5, c='orange', marker='D', label='Grid Search', edgecolors='black', linewidths=0.5, s=30)
plt.scatter(y_test_jittered, y_pred_sa, alpha=0.5, c='purple', marker='X', label='Simulated Annealing', edgecolors='black', linewidths=0.5, s=30)

# 绘制理想拟合线和误差带
x = np.linspace(min(y_test), max(y_test), 100)
plt.plot(x, x, 'k--', label='Ideal Fit')
plt.plot(x, x + np.log10(1.5), linestyle='--', color='grey', label='1.5x Band')
plt.plot(x, x - np.log10(1.5), linestyle='--', color='grey')
plt.plot(x, x + np.log10(2), linestyle='-.', color='grey', label='2x Band')
plt.plot(x, x - np.log10(2), linestyle='-.', color='grey')

# 设置坐标轴标签和图例
plt.xlabel('Experimental Logarithmic Life')
plt.ylabel('Predicted Logarithmic Life')
plt.legend()

# 去除网络线
plt.grid(False)

# 保存图片
scatter_plot_file = "D:\\python_project\\fatigue\\结果\\final\\合并\\半疲劳\\1\\LCF_svr.png"
plt.savefig(scatter_plot_file, dpi=300)

# # 绘制迭代过程曲线
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(trials_data_bayesian) + 1), trials_data_bayesian, label='Bayesian Optimization', marker='o')
# plt.plot(range(1, len(pso_mse_list) + 1), pso_mse_list, label='PSO (Min MSE per Iteration)', marker='x')
# plt.plot(range(1, len(grid_search_data) + 1), grid_search_data, label='Grid Search', marker='D')
# plt.plot(range(1, len(sa_mse_list) + 1), sa_mse_list, label='Simulated Annealing', marker='X')
# plt.xlabel('Iteration')
# plt.ylabel('Average MSE')
# plt.title('MSE Trend Over Iterations')
# plt.legend()
# plt.grid(True)
#
# iteration_plot_file = "D:\\python_project\\fatigue\\结果\\final\\LCF\\8\\LCF_svr_迭代.png"
# plt.savefig(iteration_plot_file, dpi=300)

print(f"Results saved to {output_file}, scatter plot saved as '{scatter_plot_file}'")


# 假设 y_test 和预测值存储在以下变量中
data = {
    "y_test": y_test,
    "y_pred_default": y_pred_default,
    "y_pred_grid": y_pred_grid,
    "y_pred_bayesian": y_pred_bayesian,
    "y_pred_pso": y_pred_pso,
    "y_pred_sa": y_pred_sa,
}

# 转换为 DataFrame
df = pd.DataFrame(data)

# 指定输出文件路径
output_file = "D:\\python_project\\fatigue\\结果\\final\\合并\\半疲劳\\1\\LCF_svr_predictions.xlsx"

# 保存到 Excel
df.to_excel(output_file, index=False, sheet_name="Predictions")

print(f"Prediction data saved to {output_file}")


#
# # 定义函数保存十折交叉验证结果和平均值
# def cross_validation_results(model, X_train, y_train, method_name):
#     kfold = KFold(n_splits=10, shuffle=True, random_state=42)
#     mse_scores = []  # 存储每折的 MSE
#
#     for train_idx, val_idx in kfold.split(X_train):
#         X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
#
#         # 检查 y_train 类型，选择合适的切分方式
#         if isinstance(y_train, pd.Series):  # 如果 y_train 是 pandas.Series
#             y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
#         else:  # 否则认为 y_train 是 numpy.ndarray
#             y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
#
#         # 训练模型
#         model.fit(X_train_fold, y_train_fold)
#         y_val_pred = model.predict(X_val_fold)
#
#         # 计算 MSE
#         mse = mean_squared_error(y_val_fold, y_val_pred)
#         mse_scores.append(mse)
#
#     # 计算平均值
#     avg_mse = np.mean(mse_scores)
#     results = {
#         "method": method_name,
#         "fold_mse": mse_scores,
#         "average_mse": avg_mse
#     }
#
#     return results
#
#
#
# # 假设模型 svr_bayesian、svr_pso 等已定义，X_train 和 y_train 已准备好
# results_bayesian = cross_validation_results(svr_bayesian, X_train, y_train, "Bayesian Optimization")
# results_pso = cross_validation_results(svr_pso, X_train, y_train, "PSO")
# results_grid = cross_validation_results(svr_grid, X_train, y_train, "Grid Search")
# results_sa = cross_validation_results(svr_sa, X_train, y_train, "Simulated Annealing")
#
# # 将所有结果汇总到一个列表中
# all_results = [results_bayesian, results_pso, results_grid, results_sa]
#
# # 创建一个 Excel 文件保存结果
# with pd.ExcelWriter("D:\\python_project\\fatigue\\结果\\final\\LCF\\8\\LCF_svr_训练集.xlsx") as writer:
#     for result in all_results:
#         method_name = result["method"]
#
#         # 创建一个 DataFrame
#         data = {
#             "Fold": [f"Fold {i + 1}" for i in range(len(result["fold_mse"]))],  # 每一折
#             "MSE": result["fold_mse"],  # 对应 MSE
#         }
#         df = pd.DataFrame(data)
#
#         # 在最后一行添加平均 MSE
#         avg_mse_row = pd.DataFrame({"Fold": ["Average"], "MSE": [result["average_mse"]]})
#         df = pd.concat([df, avg_mse_row], ignore_index=True)
#
#         # 将 DataFrame 写入 Excel，不同方法写入不同的 Sheet
#         df.to_excel(writer, index=False, sheet_name=method_name)
#
# # 提示保存完成
# print("十折交叉验证结果已保存到 cross_validation_results.xlsx 文件中！")



# # 置换重要性
# from sklearn.inspection import permutation_importance
#
# # 定义一个函数来计算特征重要性
# def compute_feature_importance(model, X_train, y_train, X_test, y_test):
#     result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
#     importance = result.importances_mean
#     return importance
#
# # 计算不同优化算法下SVR模型的特征重要性
# models_dict = {
#     "Default": svr_default,
#     "Bayesian": svr_bayesian,
#     "PSO": svr_pso,
#     "Grid Search": svr_grid,
#     "Simulated Annealing": svr_sa
# }
#
# importance_results = {}
#
# for name, model in models_dict.items():
#     importance = compute_feature_importance(model, X_train, y_train, X_test, y_test)
#     importance_results[name] = importance
#
# # 特征名称
# feature_names = [
#     "Power", "Scan Speed", "Elastic Modulus", "Hysteresis Area", "Damping Ratio",
#     "Max Strain", "Max Stress", "Min Strain", "Min Stress", "Secant Stiffness"
# ]
#
# importance_df = pd.DataFrame(importance_results, index=feature_names)
#
#
# # 绘制特征重要性对比图
# importance_df.plot(kind='bar', figsize=(10, 6))
# plt.title('Feature Importance Comparison for Different Optimization Methods')
# plt.ylabel('Feature Importance')
# plt.xlabel('Features')
# plt.legend(title='Optimization Methods')
# plt.tight_layout()
# plt.show()
#
# # 将特征重要性结果保存到 Excel 文件
# importance_df.to_excel('feature_importance_results_42.xlsx', index=True)
#
# print("特征重要性已保存至 'feature_importance_results_42.xlsx'")
#
#
