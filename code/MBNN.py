from keras import models
from keras import Model
from keras import layers
import random
from keras import Input
import matplotlib.pyplot as plt
from keras import losses
import os
import numpy as np
import pandas as pd
import copy
import tensorflow as tf
import keras
import h5py
from keras import callbacks
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
#定义tanh函数
# def tanh(x):
#   return 1-2/(tf.math.exp(2*x)+1)
# data=pd.read_excel(r'D:\yyyywv\研一 (2)\研一\code\疲劳寿命\数值\代码与数据\匹配后数据_2.14.xlsx')
data=pd.read_excel(r'D:\yyyywv\研一 (2)\研一\code\疲劳寿命\数值\代码与数据\MBNN匹配数据.xlsx')
data = data.dropna()



#标准化
x = data.iloc[:,1:-2]
print(x)
min_vals = np.min(x, axis=0)
max_vals = np.max(x, axis=0)
x=(x-x.mean())/x.std()
#x = (x - min_vals) / (max_vals - min_vals)
print(f'标准化后的x为:{x}')


#x=(x-x.mean())/x.std()
data.update(x)
data['y'] = np.log10(data['y'])

print(data)
# y = data.iloc[:,-1]
# y=(y-y.mean())/y.std()
# data.update(y)
#定义模型
N1 = Input(shape=(7,))
N2 = Input(shape=(7,))
N3 = Input(shape=(1,))
N4 = Input(shape=(1,))

x1 = layers.Dense(8,activation='tanh',bias_initializer=tf.ones_initializer())(N1)
x1 = layers.Dense(16,activation='tanh')(x1)
x1 = layers.Dense(32,activation='tanh')(x1)
#x1 = layers.Dense(128,activation='tanh')(x1)
#x1 = layers.Dropout(0.3)(x1)
#x1 = layers.Dense(64,activation='tanh')(x1)
#x1 = layers.Dense(32,activation='tanh')(x1)
#x1 = layers.Dropout(0.3)(x1)
x1 = layers.Dense(16,activation='tanh')(x1)

#x1 = layers.Dense(1)(x1)
#x1 = layers.Dropout(0.3)(x1)



x2 = layers.Dense(8,activation='tanh',bias_initializer=tf.ones_initializer())(N2)
x2 = layers.Dense(16,activation='tanh')(x2)
x2 = layers.Dense(32,activation='tanh')(x2)
#x2 = layers.Dense(128,activation='tanh')(x2)
#x2 = layers.Dropout(0.3)(x2)
#x2 = layers.Dense(64,activation='tanh')(x2)
#x2 = layers.Dense(32,activation='tanh')(x2)
#x2 = layers.Dense(8,activation='tanh')(x2)
#x2 = layers.Dropout(0.3)(x2)
x2 = layers.Dense(16,activation='tanh')(x2)
#print("---------x2",x2)
#print("---------x21",x21)
# x2 = layers.Dense(1)(x2)
# x2 = layers.Dropout(0.3)(x2)
# print("#####",x1,x2,N3)
# print("------------",x1.shape,x2.shape,N3.shape)
concatenated = layers.concatenate([x1,x2,N3,N4])

fatige = layers.Dense(8,activation='tanh')(concatenated)
#fatige = layers.Dense(4,activation='tanh')(fatige)
# fatige = layers.Dense(64,activation='tanh')(fatige)
# fatige = layers.Dense(16,activation='tanh')(fatige)
fatige = layers.Dense(1)(fatige)


def smooth_l1_loss(y_true, y_pred):
  """Smooth L1损失函数的实现"""
  delta = 1.0  # 平衡平方差和绝对值误差之间的转换的参数
  diff = tf.abs(y_true - y_pred)
  less_than_delta = 0.5 * (diff ** 2) / delta
  greater_equal_delta = diff - 0.5 * delta
  loss = tf.where(diff < delta, less_than_delta, greater_equal_delta)
  return tf.reduce_mean(loss)

model = Model([N1, N2, N3, N4],[fatige])
#loss_weights = {'dense_19': 1.0, 'dense_15': 1.0}

#model.compile(optimizer='Adam', loss=['mse','binary_crossentropy', ], metrics=['mae','accuracy'],loss_weights=loss_weights)
model.compile(optimizer='Adam', loss=smooth_l1_loss, metrics=['mse'])
MSE = []
R2 =[]

callbacks = [
  keras.callbacks.ModelCheckpoint(
    filepath= "D:\yyyywv\研一 (2)\研一\code\疲劳寿命\数值\代码与数据\model_MBNN.keras",
    monitor='val_loss',
    save_best_only = True,),
  ]

results = pd.DataFrame(columns=["Iteration", "Train_MSE", "Test_MSE", "Train_R2", "Test_R2"])

output_dir_for_point = r"D:\yyyywv\研一 (2)\研一\code\疲劳寿命\数值\代码与数据\合成散点图"
output_excel = os.path.join(output_dir_for_point, "Results_for_MBNN_26.1.12_10.xlsx")
with pd.ExcelWriter(output_excel) as writer:
  for j in range(10):
    # 划分测试集
    random.seed(j)
    labels = data["modulus"]
    labels = list(set(labels))
    random.shuffle(labels)
    test_size = int(len(labels) * 0.1)
    val_size = int(len(labels) * 0.1)
    train_size = len(labels) - test_size - val_size
    test_set = labels[:test_size]
    val_set = labels[test_size:test_size + val_size]
    train_set = labels[test_size + val_size:]
    test = data[data['modulus'].isin(test_set)]
    train = data[data['modulus'].isin(train_set)]
    val = data[data['modulus'].isin(val_set)]
    test = test.groupby('modulus').apply(lambda x: x.sample(n=1, random_state=j)).reset_index(drop=True)
    print(f'训练集为:{train}')
    # val = val.groupby('modulus').apply(lambda x: x.sample(n=5, random_state=1)).reset_index(drop=True)
    y_train1 = train.iloc[:, -2]
    x_train = train.iloc[:, 1:-2]
    x_train1 = x_train.iloc[:, :7]
    x_train2 = x_train.iloc[:, 7:14]
    x_train3 = x_train.iloc[:, 14]#E
    x_train4=x_train.iloc[:,15]#n2-n1
    print(f'训练集中的n2-n1为：{x_train4}')

    y_test1 = test.iloc[:, -2]
    x_test = test.iloc[:, 1:-2]
    x_test1 = x_test.iloc[:, :7]
    x_test2 = x_test.iloc[:, 7:14]
    x_test3 = x_test.iloc[:, 14]#E
    x_test4 = x_test.iloc[:,15]#n2-n1

    y_val1 = val.iloc[:, -2]
    # y_val2 = val.iloc[:, 17]
    x_val = val.iloc[:, 1:-2]
    x_val1 = x_val.iloc[:, :7]
    x_val2 = x_val.iloc[:, 7:14]
    x_val3 = x_val.iloc[:, 14]#E
    x_val4 = x_val.iloc[:,15]#n2-n1

    model.fit([x_train1, x_train2, x_train3, x_train4], [y_train1], epochs=200, batch_size=8,
              validation_data=([x_val1, x_val2, x_val3, x_val4], [y_val1]), callbacks=callbacks)
    model1 = keras.models.load_model(r'D:\yyyywv\研一 (2)\研一\code\疲劳寿命\数值\代码与数据\model_MBNN.keras',
                                        custom_objects={'smooth_l1_loss': smooth_l1_loss})

    # # 计算训练集 MSE 和 R2
    # y_train_pred = model1.predict([x_train1, x_train2, x_train3, x_train4]).ravel()
    # train_mse = np.mean((y_train_pred - y_train1) ** 2)
    # train_r2 = r2_score(y_train1, y_train_pred)
    #
    # # 计算测试集 MSE 和 R2
    # y_test_pred = model1.predict([x_test1, x_test2, x_test3, x_test4]).ravel()
    # test_mse = np.mean((y_test_pred - y_test1) ** 2)
    # test_r2 = r2_score(y_test1, y_test_pred)

    # 模型预测
    y_train_pred = model1.predict([x_train1, x_train2, x_train3, x_train4]).ravel()
    y_test_pred = model1.predict([x_test1, x_test2, x_test3, x_test4]).ravel()

    # 获取原始偏移项（最后一列）
    offset_train = train.iloc[:, -1].values
    offset_test = test.iloc[:, -1].values

    # 恢复为原始尺度：加上偏移项，再取 log10
    y_train_pred_log = np.log10(10 ** y_train_pred + offset_train)
    y_train_true_log = np.log10(10 ** y_train1 + offset_train)

    y_test_pred_log = np.log10(10 ** y_test_pred + offset_test)
    y_test_true_log = np.log10(10 ** y_test1 + offset_test)

    # 计算 MSE 和 R2（在处理后的尺度下）
    train_mse = np.mean((y_train_pred_log - y_train_true_log) ** 2)
    train_r2 = r2_score(y_train_true_log, y_train_pred_log)

    test_mse = np.mean((y_test_pred_log - y_test_true_log) ** 2)
    test_r2 = r2_score(y_test_true_log, y_test_pred_log)

    # 记录结果
    results = pd.concat([results, pd.DataFrame([{
      "Iteration": j + 1,
      "Train_MSE": train_mse,
      "Test_MSE": test_mse,
      "Train_R2": train_r2,
      "Test_R2": test_r2
    }])], ignore_index=True)

    # 转换回原始尺度（对数取反）
    y_test_original = 10 ** y_test_true_log
    y_hat_test_original = 10 ** y_test_pred_log

    # 保存到 Excel 的不同 sheet
    df_results = pd.DataFrame({'Actual': y_test_original, 'Predicted': y_hat_test_original})
    df_results.to_excel(writer, sheet_name=f'Seed_{j}', index=False)


# 保存 MSE 和 R2 到 Excel
excel_path = r'D:\yyyywv\研一 (2)\研一\code\疲劳寿命\数值\代码与数据\所有最新结果\多分支ANN_26.1.12_10.xlsx'
results.to_excel(excel_path, index=False)

print("训练结果已保存至 Excel:", excel_path)
# print("Test set size:", len(test_set))
# print("Validation set size:", len(val_set))
# print("Training set size:", len(train_set))
#
# print(test.shape)
# print(test)
print("MSE",MSE)
print('r2',R2)
