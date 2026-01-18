import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import keras

# 读取数据
file_path = r"C:\yyywv\研一\code\疲劳寿命\数值\代码与数据\数据汇总-第一版-早期.xlsx"
df = pd.read_excel(file_path)

# 提取特征和目标变量
X = df.iloc[:, 1:-2]  # 自变量（去掉第一列和最后一列）
y = df.iloc[:, -1]  # 因变量（最后一列）

# 标准化特征
y = np.log10(y)
X = (X - X.mean()) / X.std()

# 记录MSE结果和剔除的特征
mse_results = []
removed_features = []

feature_list = list(X.columns)
while len(feature_list) > 1:
    initial_mse_list = []
    importance_scores = {}

    for j in range(10):
        x_train, x_test, y_train, y_test = train_test_split(X[feature_list], y, test_size=0.2, random_state=j)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=j)

        # 定义ANN模型
        model = models.Sequential([
            layers.Dense(8, activation='tanh', bias_initializer=tf.ones_initializer()),
            layers.Dense(16, activation='tanh'),
            layers.Dense(32, activation='tanh'),
            layers.Dense(16, activation='tanh'),
            layers.Dense(1)
        ])
        model.compile(optimizer='Adam', loss='mse', metrics=['mae'])

        # 训练模型
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=2, verbose=1)

        # 计算MSE
        y_pred = model.predict(x_test)
        initial_mse_list.append(mean_squared_error(y_test, y_pred))

    # 记录初始MSE
    mse_results.append([len(feature_list), 'All Features'] + initial_mse_list)
    initial_mse = np.mean(initial_mse_list)

    # 逐个剔除特征并计算MSE
    min_mse = float('inf')
    worst_feature = None
    for feature in feature_list:
        temp_features = feature_list.copy()
        temp_features.remove(feature)
        temp_mse_list = []

        for j in range(10):
            x_train, x_test, y_train, y_test = train_test_split(X[temp_features], y, test_size=0.2, random_state=j)
            x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=j)

            model = models.Sequential([
                layers.Dense(8, activation='tanh', bias_initializer=tf.ones_initializer()),
                layers.Dense(16, activation='tanh'),
                layers.Dense(32, activation='tanh'),
                layers.Dense(16, activation='tanh'),
                layers.Dense(1)
            ])
            model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
            model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=2, verbose=1)

            y_pred = model.predict(x_test)
            temp_mse_list.append(mean_squared_error(y_test, y_pred))

        avg_mse = np.mean(temp_mse_list)
        mse_results.append([len(temp_features), feature] + temp_mse_list)
        importance_scores[feature] = initial_mse - avg_mse

        if avg_mse < min_mse:
            min_mse = avg_mse
            worst_feature = feature

    if min_mse > initial_mse:
        break

    # 记录剔除的特征
    removed_features.append(worst_feature)
    feature_list.remove(worst_feature)

# 记录最终保留的特征
final_features = feature_list.copy()

# 保存MSE结果到Excel
mse_columns = ['Num_Features', 'Removed_Feature'] + [f'MSE_Run_{i + 1}' for i in range(10)]
mse_df = pd.DataFrame(mse_results, columns=mse_columns)
output_path = r"C:\yyywv\研一\code\疲劳寿命\数值\代码与数据\所有最后结果\MSE_results_select_early_2.13.xlsx"
mse_df.to_excel(output_path, index=False)

# 保存剔除的特征和最终特征
summary_df = pd.DataFrame({'Step': range(1, len(removed_features) + 1), 'Removed_Feature': removed_features})
summary_df.loc[len(summary_df)] = ['Final_Features', ', '.join(final_features)]
summary_path = r"C:\yyywv\研一\code\疲劳寿命\数值\代码与数据\所有最后结果\Feature_Selection_Summary_early_2.13.xlsx"
summary_df.to_excel(summary_path, index=False)

print(f"MSE results saved to {output_path}")
print(f"Feature selection summary saved to {summary_path}")