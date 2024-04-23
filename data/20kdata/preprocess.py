import csv

# 打开原始文件并读取内容
with open('all.txt', 'r') as file:
    lines = file.readlines()

# 创建一个列表来存储转换后的数据
data = []

# 遍历每两行,将它们作为一个元组添加到data列表中
for i in range(0, len(lines), 2):
    input_text = lines[i+1]
    target_text = lines[i]
    data.append((input_text, target_text))

# 打开新的CSV文件并写入数据
with open('g2t_all.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # 写入列名
    writer.writerow(['input', 'target'])
    
    # 写入数据
    writer.writerows(data)

print("处理完成,结果已写入output.csv文件。")