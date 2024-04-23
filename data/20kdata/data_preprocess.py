import re

def process_line(line):
    processed = ""
    skip_char = False
    prev_char = ""

    for char in line:
        if skip_char:
            if char == ")":
                skip_char = False
            continue

        if char == "/":
            processed += "#"
        elif char == "-":
            continue
        # elif char == "+":
        #     continue
        elif char == "(":
            skip_char = True
            continue
        elif char == "1" and prev_char and is_chinese(prev_char):
            continue        
        elif char == "2" and prev_char and is_chinese(prev_char):
            continue
        else:
            processed += char

        prev_char = char
    for i, char in enumerate(processed):
        if char == '#':
            last_slash_index = i
    if last_slash_index != -1:
        processed = processed[:last_slash_index]
    return processed+'。'

def is_chinese(char):
    if '\u4e00' <= char <= '\u9fff':
        return True
    else:
        return False

#line = "我们/要求/(疑惑)供应/商/保证/涂料/、/油1/2/墨/、/胶粘剂/和/去污剂+++/一些2/。"
#print(process_line(line.strip()))
# 示例用法
with open("new_20kdata.txt", "r") as file:
    lines = file.readlines()

with open("123.txt", "w") as file:
    i = 0
    for line in lines:
        if i == 0:
            file.write(line)
            i = 1
        else:
            processed_line = process_line(line.strip())
            file.write(processed_line + "\n")
            i = 0


print("处理完成,结果已写入output.txt")