import csv

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data

# 示例用法
file_path = 'new_data.csv'
data = read_csv(file_path)

# print(f"从 {file_path} 读取到的数据:")
# for row in data:
#     print(row)

with open('new_data.txt', 'w', newline='') as file:
    for row in data:
        file.write(row[1]+'\n')
        print(row[1])