import random
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import keras
import pandas as pd
import copy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import os

R2 = []
MSE = []
# 定义模型
model = models.Sequential()
model.add(layers.Dense(8, activation='tanh',bias_initializer=tf.ones_initializer()))
model.add(layers.Dense(16, activation='tanh'))
model.add(layers.Dense(32, activation='tanh'))
#model.add(layers.Dense(128, activation='tanh'))
#model.add(layers.Dropout(0.3))
#model.add(layers.Dense(64, activation='tanh'))
model.add(layers.Dense(16, activation='tanh'))
#model.add(layers.Dropout(0.3))
model.add(layers.Dense(1))
model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
callbacks = [
  keras.callbacks.ModelCheckpoint(
    filepath= "D:\yyyywv\研一 (2)\研一\code\疲劳寿命\数值\代码与数据\ANN_early.keras",
    monitor='val_loss',
    save_best_only = True,),
  ]
#定义tanh函数
def tanh(x):
  return 1-2/(tf.math.exp(2*x)+1)
#数据处理
test=pd.read_excel(r"D:\yyyywv\研一 (2)\研一\code\疲劳寿命\数值\代码与数据\数据汇总-第一版-早期.xlsx",engine='openpyxl')
y=test.iloc[:,-1]
# x=test.iloc[:, [1,4,7,8]]
# x=test.iloc[:, [2,3,4,5,7,8]]#半疲劳
x=test.iloc[:,[1,2,4,5,6,8]]#早期
# x=test.iloc[:,1:9]
print(x)
x=(x-x.mean())/x.std()
y=np.log10(y)
# test = pd.read_excel(r"D:\yyyywv\研一 (2)\研一\code\疲劳寿命\数值\代码与数据\MBNN匹配数据.xlsx", engine='openpyxl')
# correction = test.iloc[:, -1]  # 最后一列为修正值
# y = np.log10(test.iloc[:, -2])  # 倒数第二列为原始对数因变量
# x = test.iloc[:, 1:-2]
# print(x)
# x = (x - x.mean()) / x.std()
# print(y)

MSE_train = []  # 用于存储训练集的MSE
MSE_test = []   # 用于存储测试集的MSE
R2_train = []
R2_test = []

output_dir_for_point = r"D:\yyyywv\研一 (2)\研一\code\疲劳寿命\数值\代码与数据\合成散点图"
output_excel = os.path.join(output_dir_for_point, "Results_for_early_已剔除_6.3_10.xlsx")


with pd.ExcelWriter(output_excel) as writer:
    for j in range(10):
        # 数据集划分
        random.seed(j)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=j)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=j)

        # 模型训练
        history = model.fit(x_train,
                            y_train,
                            epochs=200,
                            batch_size=2,
                            validation_data=(x_val, y_val),
                            callbacks=callbacks)

        model1 = keras.models.load_model(r'D:\yyyywv\研一 (2)\研一\code\疲劳寿命\数值\代码与数据\ANN_early.keras')

        # 训练集预测与评估
        y_train_pred = model1.predict(x_train).ravel()
        mse_train = np.mean((y_train_pred - y_train) ** 2)
        r2_train = r2_score(y_train, y_train_pred)
        MSE_train.append(mse_train)
        R2_train.append(r2_train)

        # 测试集预测与评估
        y_hat = model1.predict(x_test).ravel()
        mse_test = np.mean((y_hat - y_test) ** 2)
        r2 = r2_score(y_test, y_hat)
        MSE_test.append(mse_test)
        R2_test.append(r2)

        # 转换回原始尺度
        y_test_original = 10 ** y_test
        y_hat_test_original = 10 ** y_hat

        # 保存实际值与预测值
        df_results = pd.DataFrame({'Actual': y_test_original, 'Predicted': y_hat_test_original})
        df_results.to_excel(writer, sheet_name=f'Seed_{j}', index=False)

# 将MSE和R2保存到Excel文件
metrics_data = pd.DataFrame({
    'Train_MSE': MSE_train,
    'Test_MSE': MSE_test,
    'Train_R2': R2_train,
    'Test_R2': R2_test
})
metrics_data.to_excel(r'D:\yyyywv\研一 (2)\研一\code\疲劳寿命\数值\代码与数据\所有最新结果\ANN_early已剔除_6.3_10.xlsx', index=False)

print('Train_MSE', MSE_train)
print('Test_MSE',MSE_test)
print(np.mean(MSE_test))
print("R2",R2)

# 绘制箱线图
plt.figure(figsize=(8, 6))
plt.boxplot(MSE_test, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'), medianprops=dict(color='red'))

# 添加标题和标签
plt.title("Boxplot of MSE Values", fontsize=16)
plt.ylabel("Mean Squared Error (MSE)", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示箱线图
plt.show()
