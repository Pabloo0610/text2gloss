import re

# 读取文件内容到变量text中
with open('raw_output_ft2.txt', 'r', encoding='utf-8') as file:
    text = file.read()

pattern = r'Assistant:(.*?)(?:output|$)'

results = re.findall(pattern, text, re.DOTALL)
print(len(results))

# 将结果写入新文件
with open('extracted_data_ft2.txt', 'w', encoding='utf-8') as output_file:
    for result in results:
        output_file.write(result.strip() + '\n')