import csv
import random
from collections import deque

# 打开原始CSV文件并读取内容
with open('t2g_all_wo_prompt.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 跳过标题行
    data = list(reader)

# 打乱数据顺序
random.shuffle(data)

# 计算每个数据集的大小
total_size = len(data)
train_size = int(total_size * 0.8)
val_size = int(total_size * 0.1)
test_size = total_size - train_size - val_size

# 划分数据集
train_data = deque(data[:train_size])
val_data = deque(data[train_size:train_size+val_size])
test_data = deque(data[train_size+val_size:])

# 写入训练集
with open('t2g_train_wo_prompt.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['text'])  # 写入标题行
    writer.writerows(train_data)

# 写入验证集
with open('t2g_val_wo_prompt.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['text'])  # 写入标题行
    writer.writerows(val_data)

# 写入测试集
with open('t2g_test_wo_prompt.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['text'])  # 写入标题行
    writer.writerows(test_data)

print("数据划分完成,结果已写入train.csv、val.csv和test.csv文件。")