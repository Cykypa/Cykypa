# 导入需要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
# 加载数据
data = pd.read_csv('./data/test_data.csv', header=None, names=['date', 'value'])
ts = data['value'].values

# 拆分数据，使用前80%的数据作为训练集和后20%的数据作为测试集
train_size = int(len(ts) * 0.8)
train, test = ts[0:train_size], ts[train_size:len(ts)]

# 训练模型
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    # 预测下一个值
    model = ARIMA(history, order=(5, 1, 0))  # 这里选择ARIMA(5,1,0)模型
    model_fit = model.fit(disp=0)
    pred = model_fit.forecast()[0]
    predictions.append(pred)
    # 将真实值添加到历史列表中进行训练
    history.append(test[i])
    print('Predicted=%f, Expected=%f' % (pred, test[i]))

# 计算模型的均方根误差
error = mean_squared_error(test, predictions)
print('Test RMSE: %.3f' % np.sqrt(error))

# 可视化结果
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()