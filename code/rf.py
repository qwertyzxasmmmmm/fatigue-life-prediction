import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from pyswarm import pso
import optuna
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import ParameterGrid
from scipy.optimize import dual_annealing

# 固定随机种子
np.random.seed(42)
random.seed(42)

# 读取数据
input_file = "早期数据.xlsx"  # 替换为你的文件名
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

# # 高周疲劳，第一列无编号
# encoder = OneHotEncoder(sparse=False)
# encoded_features = encoder.fit_transform(data[['Comments']])
# encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names(['Comments']))
#
# # 合并编码后的特征到原始DataFrame
# test = pd.concat([data.drop(['Comments'], axis=1), encoded_df], axis=1)
# print(test)
# y = np.log10(test.iloc[:, -3])  # 修改为倒数第二列作为因变量
#
# # 除去因变量的所有列都是自变量
# X = test.drop(test.columns[-3], axis=1)  # 删除因变量列，其余为自变量
# print(X)
# print(y)
# X = (X - X.mean()) / X.std()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# 网格搜索
trials_data_grid = []
def grid_search():
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 6, 9, 12],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [0.5, 0.75, 1.0],
        'max_samples': [0.6, 0.8, 0.99]
    }
    best_params = None
    best_mse = float("inf")
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    for params in ParameterGrid(param_grid):
        cv_mse_scores = []
        for train_idx, val_idx in kfold.split(X_train):
            # # 低周疲劳
            # X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            # y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # 高周疲劳
            # 转换为 numpy 数组
            X_train_np = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
            y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

            # 根据索引进行切分
            X_train_fold, X_val_fold = X_train_np[train_idx], X_train_np[val_idx]
            y_train_fold, y_val_fold = y_train_np[train_idx], y_train_np[val_idx]

            rf = RandomForestRegressor(**params, random_state=42)
            rf.fit(X_train_fold, y_train_fold)
            y_val_pred = rf.predict(X_val_fold)
            mse = mean_squared_error(y_val_fold, y_val_pred)
            cv_mse_scores.append(mse)

        avg_mse = np.mean(cv_mse_scores)
        trials_data_grid.append(avg_mse)
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_params = params

    return best_params, best_mse

best_params_grid, best_mse_grid = grid_search()

rf_grid = RandomForestRegressor(**best_params_grid, random_state=42)
rf_grid.fit(X_train, y_train)
y_pred_grid = rf_grid.predict(X_test)
grid_mse = mean_squared_error(y_test, y_pred_grid)
grid_r2 = r2_score(y_test, y_pred_grid)




# 贝叶斯优化
trials_data_bayesian = []
def bayesian_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20, log=True)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    max_features = trial.suggest_float('max_features', 0.1, 1.0)
    max_samples = trial.suggest_float('max_samples', 0.5, 0.99)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_samples=max_samples,
        random_state=42
    )

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_mse_scores = []

    for train_idx, val_idx in kfold.split(X_train):
        # # 低周疲劳
        # X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        # y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # 修复的代码：确保使用 numpy 数组进行索引，高周疲劳
        # 转换为 numpy 数组
        X_train_np = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

        # 根据索引进行切分
        X_train_fold, X_val_fold = X_train_np[train_idx], X_train_np[val_idx]
        y_train_fold, y_val_fold = y_train_np[train_idx], y_train_np[val_idx]

        rf.fit(X_train_fold, y_train_fold)
        y_val_pred = rf.predict(X_val_fold)
        mse = mean_squared_error(y_val_fold, y_val_pred)
        cv_mse_scores.append(mse)

    avg_mse = np.mean(cv_mse_scores)
    trials_data_bayesian.append(avg_mse)
    return avg_mse

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(bayesian_objective, n_trials=200)
best_params_bayesian = study.best_params

rf_bayesian = RandomForestRegressor(
    **best_params_bayesian, random_state=42
)
rf_bayesian.fit(X_train, y_train)
y_pred_bayesian = rf_bayesian.predict(X_test)
bayesian_mse = mean_squared_error(y_test, y_pred_bayesian)
bayesian_r2 = r2_score(y_test, y_pred_bayesian)

pso_mse_list = []  # 用于存储每次迭代所有粒子中最小的 MSE 值

def pso_objective(parameters):
    # 从传入的参数中提取各个超参数
    n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, max_samples = parameters
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)

    # 创建随机森林回归器
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_samples=max_samples,
        random_state=42
    )

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    mse_list = []  # 存储每个折叠的 MSE

    # 进行交叉验证，计算每个折叠的 MSE
    for train_idx, val_idx in kfold.split(X_train):
        # # 低周疲劳
        # X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        # y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # 高周疲劳
        # 转换为 numpy 数组
        X_train_np = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

        # 根据索引进行切分
        X_train_fold, X_val_fold = X_train_np[train_idx], X_train_np[val_idx]
        y_train_fold, y_val_fold = y_train_np[train_idx], y_train_np[val_idx]

        rf.fit(X_train_fold, y_train_fold)
        y_val_pred = rf.predict(X_val_fold)
        mse = mean_squared_error(y_val_fold, y_val_pred)
        mse_list.append(mse)

    # 返回当前粒子的平均 MSE
    return np.mean(mse_list)

# 使用 PSO 求解
lb = [50, 3, 2, 1, 0.1, 0.5]  # 参数的下界
ub = [300, 20, 10, 5, 1.0, 0.99]  # 参数的上界

# 修改 collect_pso_mse 以记录每次迭代所有粒子的最小 MSE
def collect_pso_mse(parameters):
    mse = pso_objective(parameters)  # 计算当前粒子的 MSE
    return mse

# PSO 外部包装，手动实现每次迭代最小 MSE 记录
from pyswarm import pso  # 确认这里引入的是正确的 pso 函数实现

# 自定义 pso，添加记录逻辑
def custom_pso(objective, lb, ub, swarmsize=10, maxiter=200):
    global pso_mse_list  # 用于记录每次迭代的最小 MSE

    best_params = None
    best_mse = float("inf")

    for iter_num in range(maxiter):
        iteration_mse_list = []  # 当前迭代中所有粒子的 MSE

        def wrapper(parameters):
            mse = objective(parameters)
            iteration_mse_list.append(mse)
            return mse

        # 调用原始的 pso 函数，进行一次迭代
        best_params_iter, _ = pso(wrapper, lb, ub, swarmsize=swarmsize, maxiter=1)

        # 更新当前最优
        min_mse = min(iteration_mse_list)
        if min_mse < best_mse:
            best_mse = min_mse
            best_params = best_params_iter

        # 保存当前迭代最小 MSE
        pso_mse_list.append(min_mse)

    return best_params, best_mse

# 调用自定义 PSO 函数
best_params_pso, best_mse_pso = custom_pso(collect_pso_mse, lb, ub, swarmsize=10, maxiter=200)

# 检查最终的 pso_mse_list 长度
print(len(pso_mse_list))  # 应该输出 200
print(pso_mse_list)  # 输出所有迭代中最小的 MSE 值


rf_pso = RandomForestRegressor(
    n_estimators=int(best_params_pso[0]),
    max_depth=int(best_params_pso[1]),
    min_samples_split=int(best_params_pso[2]),
    min_samples_leaf=int(best_params_pso[3]),
    max_features=best_params_pso[4],
    max_samples=best_params_pso[5],
    random_state=42
)
rf_pso.fit(X_train, y_train)
y_pred_pso = rf_pso.predict(X_test)
pso_mse = mean_squared_error(y_test, y_pred_pso)
pso_r2 = r2_score(y_test, y_pred_pso)


# 模拟退火
trials_data_sa = []
def simulated_annealing():
    bounds = [(50, 300), (3, 20), (2, 10), (1, 5), (0.1, 1.0), (0.5, 0.99)]

    def sa_objective(params):
        n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, max_samples = params
        rf = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            max_features=max_features,
            max_samples=max_samples,
            random_state=42
        )

        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        mse_list = []
        for train_idx, val_idx in kfold.split(X_train):
            # # 低周疲劳
            # X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            # y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # 高周疲劳
            # 转换为 numpy 数组
            X_train_np = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
            y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

            # 根据索引进行切分
            X_train_fold, X_val_fold = X_train_np[train_idx], X_train_np[val_idx]
            y_train_fold, y_val_fold = y_train_np[train_idx], y_train_np[val_idx]

            rf.fit(X_train_fold, y_train_fold)
            y_val_pred = rf.predict(X_val_fold)
            mse = mean_squared_error(y_val_fold, y_val_pred)
            mse_list.append(mse)

        avg_mse = np.mean(mse_list)
        trials_data_sa.append(avg_mse)
        return avg_mse

    # 设置最大迭代次数为 200
    result = dual_annealing(sa_objective, bounds, maxiter=200)
    return result.x, result.fun
    # result = dual_annealing(sa_objective, bounds)
    # return result.x, result.fun

best_params_sa, best_mse_sa = simulated_annealing()

rf_sa = RandomForestRegressor(
    n_estimators=int(best_params_sa[0]),
    max_depth=int(best_params_sa[1]),
    min_samples_split=int(best_params_sa[2]),
    min_samples_leaf=int(best_params_sa[3]),
    max_features=best_params_sa[4],
    max_samples=best_params_sa[5],
    random_state=42
)
rf_sa.fit(X_train, y_train)
y_pred_sa = rf_sa.predict(X_test)
sa_mse = mean_squared_error(y_test, y_pred_sa)
sa_r2 = r2_score(y_test, y_pred_sa)



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


# 使用默认参数的随机森林模型
rf_default = RandomForestRegressor(random_state=42)
rf_default.fit(X_train, y_train)
y_pred_default = rf_default.predict(X_test)

default_mse = mean_squared_error(y_test, y_pred_default)
default_r2 = r2_score(y_test, y_pred_default)
default_band = calc_band_proportion(y_test, y_pred_default)


# 提取默认参数
default_params = rf_default.get_params()

# 保存结果到Excel
output_file = "D:\\python_project\\fatigue\\结果\\final\\LCF\\加形态学参数\\3\\LCF_rf.xlsx"
results = {
    "Metric": ["MSE", "R²", "Within 1.5 Band", "Within 2 Band", "Outside 2 Band"],
    "Default RF": [default_mse, default_r2, *default_band],
    "Bayesian Optimization": [bayesian_mse, bayesian_r2, *bayesian_band],
    "PSO": [pso_mse, pso_r2, *pso_band],
    "Grid Search": [grid_mse, grid_r2, *grid_band],
    "Simulated Annealing": [sa_mse, sa_r2, *sa_band]
}
params = {
    "Parameter": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "max_samples"],
    "Default RF": [
        default_params["n_estimators"],
        default_params["max_depth"],
        default_params["min_samples_split"],
        default_params["min_samples_leaf"],
        default_params["max_features"],
        default_params["max_samples"]
    ],
    "Bayesian Optimization": list(best_params_bayesian.values()),
    "PSO": list(best_params_pso),
    "Grid Search": list(best_params_grid.values()),
    "Simulated Annealing": [
        int(best_params_sa[0]),
        int(best_params_sa[1]),
        int(best_params_sa[2]),
        int(best_params_sa[3]),
        best_params_sa[4],
        best_params_sa[5]
    ]
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
x = np.linspace(3.0, 5.3, 100)  # 设置x轴的范围，稍微扩大
plt.plot(x, x, 'k--', label='Ideal Fit')
plt.plot(x, x + np.log10(1.5), linestyle='--', color='grey', label='1.5x Band')
plt.plot(x, x - np.log10(1.5), linestyle='--', color='grey')
plt.plot(x, x + np.log10(2), linestyle='-.', color='grey', label='2x Band')
plt.plot(x, x - np.log10(2), linestyle='-.', color='grey')

# 设置坐标轴标签和图例
plt.xlabel('Experimental Logarithmic Life')
plt.ylabel('Predicted Logarithmic Life')
# 设置坐标轴的范围，稍微增加宽度
plt.xlim(3.0, 5.3)
plt.ylim(3.0, 5.3)

plt.legend()

# 去除网络线
plt.grid(False)

# 保存图片
scatter_plot_file = "D:\\python_project\\fatigue\\结果\\final\\LCF\\加形态学参数\\3\\LCF_rf.png"
plt.savefig(scatter_plot_file, dpi=300)

# 绘制迭代过程曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(trials_data_bayesian) + 1), trials_data_bayesian, label='Bayesian Optimization', marker='o')
plt.plot(range(1, len(pso_mse_list) + 1), pso_mse_list, label='PSO (Min MSE per Iteration)', marker='x')
plt.plot(range(1, len(trials_data_grid) + 1), trials_data_grid, label='Grid Search', marker='D')
plt.plot(range(1, len(trials_data_sa) + 1), trials_data_sa, label='Simulated Annealing', marker='X')
plt.xlabel('Iteration')
plt.ylabel('Average MSE')
plt.title('MSE Trend Over Iterations')
plt.legend()
plt.grid(True)

iteration_plot_file = "D:\\python_project\\fatigue\\结果\\final\\LCF\\加形态学参数\\3\\LCF_rf_迭代.png"
plt.savefig(iteration_plot_file, dpi=300)

print(f"Results saved to {output_file}, scatter plot saved as '{scatter_plot_file}', and iteration trend plot saved as '{iteration_plot_file}'")

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
output_file = "D:\\python_project\\fatigue\\结果\\final\\LCF\\加形态学参数\\3\\LCF_rf_predictions.xlsx"

# 保存到 Excel
df.to_excel(output_file, index=False, sheet_name="Predictions")

print(f"Prediction data saved to {output_file}")


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



# # 假设模型 svr_bayesian、svr_pso 等已定义，X_train 和 y_train 已准备好
# results_bayesian = cross_validation_results(rf_bayesian, X_train, y_train, "Bayesian Optimization")
# results_pso = cross_validation_results(rf_pso, X_train, y_train, "PSO")
# results_grid = cross_validation_results(rf_grid, X_train, y_train, "Grid Search")
# results_sa = cross_validation_results(rf_sa, X_train, y_train, "Simulated Annealing")
#
# # 将所有结果汇总到一个列表中
# all_results = [results_bayesian, results_pso, results_grid, results_sa]
#
# # 创建一个 Excel 文件保存结果
# with pd.ExcelWriter("D:\\python_project\\fatigue\\结果\\final\\LCF\\0\\LCF_rf_训练集.xlsx") as writer:
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



