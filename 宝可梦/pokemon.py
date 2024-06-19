import numpy
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置字体以支持中文
font_path = "C:/Windows/Fonts/simsun.ttc"  # 使用宋体字体（请确保字体路径正确）
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# 加载数据
data = pd.read_csv("pokemon0820.csv")  # 假设数据已保存为CSV文件

# 数据预处理（处理缺失值和异常值，如果需要）
# 例如：data = data.dropna()  # 删除缺失值

# 特征工程
X = data[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'generation', 'is_legendary']]
y = data['hp'] + data['attack'] + data['defense'] + data['sp_attack'] + data['sp_defense'] + data['speed']

# 划分数据集，前3/4用于训练，后1/4用于测试
split_index = int(len(data) * 0.75)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 初始化随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("均方误差 (MSE):", mse)
print("均方根误差 (RMSE):", rmse)
print("决定系数 (R²):", r2)

# 展示测试数据的预测结果并按预测误差排序
results = pd.DataFrame({
    '实际值': y_test,
    '预测值': y_pred
})
results['预测误差'] = abs(results['实际值'] - results['预测值'])
results_sorted = results.sort_values(by='预测误差', ascending=False)

print("\n测试数据的预测结果 (按预测误差从大到小排序):")
print(results_sorted.head(10))

# 打印测试数据集大小
print("\n测试数据集大小:", len(X_test))

print("\n测试数据的中位数误差:", numpy.median(results_sorted['预测误差']))

print("\n测试数据的平均误差:", numpy.mean(results_sorted['预测误差']))

# 可视化展示排行榜前10名
top_n = 10
results_top_n = results_sorted.head(top_n)
plt.figure(figsize=(12, 8))

plt.barh(results_top_n.index, results_top_n['预测误差'], color='skyblue', edgecolor='black')
plt.xlabel('预测误差')
plt.ylabel('样本索引')
plt.title(f'预测误差排行榜（前{top_n}名）')
plt.gca().invert_yaxis()  # 翻转y轴，使得误差最小的排在上面
plt.show()

# 预测新数据
new_pokemon = pd.DataFrame([[100, 100, 100, 100, 100, 100, 5, 1]],
                           columns=['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'generation',
                                    'is_legendary'])
predicted_score = model.predict(new_pokemon)
print("\n新宝可梦的综合能力预测得分:", predicted_score[0])
