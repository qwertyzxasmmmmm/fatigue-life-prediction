import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import os
from sklearn.metrics import r2_score, mean_squared_error

# 加载数据集
test = pd.read_excel(r"D:\yyyywv\研一 (2)\研一\code\疲劳寿命\数值\代码与数据\数据汇总-第一版-早期.xlsx", engine='openpyxl')

y = np.log10(test.iloc[:, -1])  # 因变量
x = test.iloc[:, [1, 2, 4, 5, 6, 8]]#早期
print(x)
print(y)
# x=test.iloc[:, [2,3,4,5,7,8]]
# x = test.iloc[:, 1:9]
x = (x - x.mean()) / x.std()  # 标准化

# test = pd.read_excel(r"D:\yyyywv\研一 (2)\研一\code\疲劳寿命\数值\代码与数据\MBNN匹配数据.xlsx", engine='openpyxl')
# correction = test.iloc[:, -1]  # 最后一列为修正值
# y = np.log10(test.iloc[:, -2])  # 倒数第二列为原始对数因变量
# x = test.iloc[:, 1:-2]
# print(x)
# print(correction)
# x = (x - x.mean()) / x.std()

# 初始化结果列表
MSE_train = []
MSE_test = []
R2_train = []
R2_test = []

# 多次随机划分
for j in range(10):
    # 数据划分
    # x_train, x_test, y_train, y_test, corr_train, corr_test = train_test_split(
    #     x, y, correction, test_size=0.2, random_state=j)
    # x_test, x_val, y_test, y_val, corr_test, corr_val = train_test_split(
    #     x_test, y_test, corr_test, test_size=0.5, random_state=j)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=j)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=j)

    # 设置XGBoost模型
    model = xgb.XGBRegressor()
    model.fit(x_train, y_train)

    # # 训练集预测
    # y_train_pred_log = model.predict(x_train)
    # y_train_true_real = 10 ** y_train + corr_train.values
    # y_train_pred_real = 10 ** y_train_pred_log + corr_train.values
    # y_train_true_final = np.log10(y_train_true_real)
    # y_train_pred_final = np.log10(y_train_pred_real)
    #
    # mse_train = mean_squared_error(y_train_true_final, y_train_pred_final)
    # r2_train = r2_score(y_train_true_final, y_train_pred_final)
    # MSE_train.append(mse_train)
    # R2_train.append(r2_train)
    #
    # # 测试集预测
    # y_test_pred_log = model.predict(x_test)
    # y_test_true_real = 10 ** y_test + corr_test.values
    # y_test_pred_real = 10 ** y_test_pred_log + corr_test.values
    # y_test_true_final = np.log10(y_test_true_real)
    # y_test_pred_final = np.log10(y_test_pred_real)
    #
    # mse_test = mean_squared_error(y_test_true_final, y_test_pred_final)
    # r2_test = r2_score(y_test_true_final, y_test_pred_final)
    # MSE_test.append(mse_test)
    # R2_test.append(r2_test)

    # 训练集预测
    y_train_pred = model.predict(x_train)
    # y_train_true_real = 10 ** y_train + corr_train.values
    # y_train_pred_real = 10 ** y_train_pred_log + corr_train.values
    # y_train_true_final = np.log10(y_train_true_real)
    # y_train_pred_final = np.log10(y_train_pred_real)
    mse_train = np.mean((y_train_pred - y_train) ** 2)
    r2_train = r2_score(y_train, y_train_pred)
    MSE_train.append(mse_train)
    R2_train.append(r2_train)
    # mse_train = mean_squared_error(y_train_true_final, y_train_pred_final)
    # r2_train = r2_score(y_train_true_final, y_train_pred_final)
    # MSE_train.append(mse_train)
    # R2_train.append(r2_train)

    # 测试集预测
    y_test_pred = model.predict(x_test)
    # y_test_true_real = 10 ** y_test + corr_test.values
    # y_test_pred_real = 10 ** y_test_pred_log + corr_test.values
    # y_test_true_final = np.log10(y_test_true_real)
    # y_test_pred_final = np.log10(y_test_pred_real)
    #
    # mse_test = mean_squared_error(y_test_true_final, y_test_pred_final)
    # r2_test = r2_score(y_test_true_final, y_test_pred_final)
    mse_test = np.mean((y_test_pred - y_test) ** 2)
    r2_test = r2_score(y_test, y_test_pred)
    MSE_test.append(mse_test)
    R2_test.append(r2_test)

    # 保存预测值（原始尺度）
    result_train_df = pd.DataFrame({
        'y_train': 10 ** y_train,
        'y_train_pred': 10 ** y_train_pred
    })
    result_test_df = pd.DataFrame({
        'y_test': 10 ** y_test,
        'y_test_pred': 10 ** y_test_pred
    })

    # 保存到文件夹
    seed_folder = r"D:\yyyywv\研一 (2)\研一\code\疲劳寿命\数值\代码与数据\合成散点图\Early_XGB_已剔除\SeedResults"
    os.makedirs(seed_folder, exist_ok=True)
    result_train_df.to_excel(os.path.join(seed_folder, f"Seed_{j}_train_results.xlsx"), index=False)
    result_test_df.to_excel(os.path.join(seed_folder, f"Seed_{j}_test_results.xlsx"), index=False)

# 总结结果保存为 Excel
mse_data = pd.DataFrame({
    'Train_MSE': MSE_train,
    'Test_MSE': MSE_test,
    'Train_R2': R2_train,
    'Test_R2': R2_test
})
mse_data.to_excel(
    r'D:\yyyywv\研一 (2)\研一\code\疲劳寿命\数值\代码与数据\所有最新结果\XGB_Early_已剔除_6.3.xlsx',
    index=False
)

# 打印输出
print('Train_MSE:', MSE_train)
print('Test_MSE:', MSE_test)
print('Train_R2:', R2_train)
print('Test_R2:', R2_test)
print('Test MSE 平均值:', np.mean(MSE_test))

# 绘制箱线图
plt.figure(figsize=(8, 6))
plt.boxplot(MSE_test, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))

plt.title("Boxplot of MSE Values", fontsize=16)
plt.ylabel("Mean Squared Error (MSE)", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
